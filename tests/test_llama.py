import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from torch._tensor import Tensor

FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3 format

def get_fp8_constants() -> Tuple[torch.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if torch.version.hip is not None:
        pt_fp8_dtype = torch.float8_e4m3fnuz
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
    return pt_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12

#from META FP8 gemm 
def quantize_fp8_row(
    a: Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    output_device = a.device

    a_shape = a.shape
    # Get constants.
    pt_dtype, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=-1)[0]
    # Apply clamping.
    row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[:-1]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[..., None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, (1 / a_scale).view(a_shape[:-1])  # pyre-ignore

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (
        (x_view * (FP8_E4M3_MAX / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n),
        (x_amax / FP8_E4M3_MAX).view(m, -1)
    )

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (FP8_E4M3_MAX / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / FP8_E4M3_MAX).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = quantize_fp8_row(x), quantize_fp8_row(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1], rowwise_scaling=True))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM Rowwise:')
    for m in (64, 128, 4096):
        #for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
        for k, n in [(16384, 5120), (5120, 4096), (2048, 5120)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
