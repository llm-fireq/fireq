import torch
import triton
import triton.language as tl

@triton.jit
def dequant_int8_bf16_kernel(
    iweight_ptr,
    scale_ptr,
    out_ptr,
    M, N,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = offs_n < N

    group_idx = offs_n // group_size

    iweight = tl.load(iweight_ptr + offs_m*N + offs_n, mask=mask).to(tl.bfloat16)
    
    scale = tl.load(scale_ptr + offs_m * (N // group_size) + group_idx, mask=mask)
    
    out_fp8 = iweight * scale
    out_fp8 = out_fp8.to(tl.bfloat16)

    tl.store(out_ptr + offs_m * N + offs_n, out_fp8, mask=mask)

def triton_dequantize_int8_bf16(iweight, scale, group_size):
    assert iweight.ndim == 2 and scale.ndim == 2
    M, N = iweight.shape
    assert scale.shape == (M, N // group_size)

    out = torch.empty_like(iweight, dtype=torch.bfloat16)
    BLOCK_N = 128

    grid = (M, triton.cdiv(N, BLOCK_N))
    dequant_int8_bf16_kernel[grid](
        iweight, scale, out,
        M, N,
        group_size=group_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    
    return out


@triton.jit
def dequant_int8_fp16_kernel(
    iweight_ptr,
    scale_ptr,
    out_ptr,
    M, N,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = offs_n < N

    group_idx = offs_n // group_size

    iweight = tl.load(iweight_ptr + offs_m*N + offs_n, mask=mask).to(tl.float16)
    
    scale = tl.load(scale_ptr + offs_m * (N // group_size) + group_idx, mask=mask)
    
    out_fp8 = iweight * scale
    out_fp8 = out_fp8.to(tl.float16)

    tl.store(out_ptr + offs_m * N + offs_n, out_fp8, mask=mask)

def triton_dequantize_int8_fp16(iweight, scale, group_size):
    assert iweight.ndim == 2 and scale.ndim == 2
    M, N = iweight.shape
    assert scale.shape == (M, N // group_size)

    out = torch.empty_like(iweight, dtype=torch.float16)
    BLOCK_N = 128

    grid = (M, triton.cdiv(N, BLOCK_N))
    dequant_int8_fp16_kernel[grid](
        iweight, scale, out,
        M, N,
        group_size=group_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    
    return out

@triton.jit
def dequant_int8_fp8_kernel(
    iweight_ptr,
    scale_ptr,
    out_ptr,
    M, N,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = offs_n < N

    group_idx = offs_n // group_size

    iweight = tl.load(iweight_ptr + offs_m*N + offs_n, mask=mask).to(tl.bfloat16)
    
    scale = tl.load(scale_ptr + offs_m * (N // group_size) + group_idx, mask=mask)
    
    out_fp8 = iweight * scale
    out_fp8 = out_fp8.to(tl.float8e4nv)

    tl.store(out_ptr + offs_m * N + offs_n, out_fp8, mask=mask)

def triton_dequantize_int8_fp8(iweight, scale, group_size):
    assert iweight.ndim == 2 and scale.ndim == 2
    M, N = iweight.shape
    assert scale.shape == (M, N // group_size)

    out = torch.empty_like(iweight, dtype=torch.float8_e4m3fn)
    BLOCK_N = 128

    grid = (M, triton.cdiv(N, BLOCK_N))
    dequant_int8_fp8_kernel[grid](
        iweight, scale, out,
        M, N,
        group_size=group_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    
    return out
