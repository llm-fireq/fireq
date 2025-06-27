import torch
import triton
import triton.language as tl
from typing import Tuple

__all__ = [
  "dequantize_int8_fp8",
  "triton_dequantize_int8_fp8",
  "triton_act_quant",
  "get_activation_fcn",
  "triton_fp8_bf16_quant",
  "triton_int8_bf16_quant",
  "triton_int8_fp16_quant",
  "triton_fp8_fp16_quant"
]


## dequantize

def dequantize_int8_fp8(iweight:torch.Tensor, scale:torch.Tensor, group_size:int, on_dtype=torch.float)->torch.Tensor:

    assert iweight.dim() == 2
    N, K = iweight.shape

    iweight_fp = iweight.to(on_dtype)

    expanded_scale = scale.to(on_dtype).repeat_interleave(group_size, dim=1)
    dequantized_weight = iweight_fp * expanded_scale
    
    return dequantized_weight.to(torch.float8_e4m3fn)

# @triton.jit
# def dequant_int8_fp8_kernel(
#     iweight_ptr,
#     scale_ptr,
#     out_ptr,
#     M, N,
#     group_size: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     offs_m = pid_m
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     mask = offs_n < N

#     group_idx = offs_n // group_size

#     iweight = tl.load(iweight_ptr + offs_m*N + offs_n, mask=mask).to(tl.float32)
#     scale = tl.load(scale_ptr + offs_m * (N // group_size) + group_idx, mask=mask).to(tl.float32)

#     out_fp32 = iweight * scale

#     out_fp8 = out_fp32.to(tl.float8e4nv)
#     tl.store(out_ptr + offs_m * N + offs_n, out_fp8, mask=mask)

# def triton_dequantize_int8_fp8(iweight, scale, group_size):
#     assert iweight.ndim == 2 and scale.ndim == 2
#     M, N = iweight.shape
#     assert scale.shape == (M, N // group_size)

#     out = torch.empty_like(iweight, dtype=torch.float8_e4m3fn)
#     BLOCK_N = 128

#     grid = (M, triton.cdiv(N, BLOCK_N))
#     dequant_int8_fp8_kernel[grid](
#         iweight, scale, out,
#         M, N,
#         group_size=group_size,
#         BLOCK_N=BLOCK_N,
#         num_warps=4,
#         num_stages=2
#     )
    
#     return out

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


@triton.jit
def bf16_to_fp8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_x = pid*N + offs
    mask = offs < N
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    #scale = max_val / 448.0
    scale = tl.where(
        max_val > 0,
        max_val / 448.0, # 448
        1.0
    )

    # Clamp
    # MIN_SCALE = 1e-2
    # MAX_SCALE = 32.0
    # scale = tl.clamp(scale, MIN_SCALE, MAX_SCALE)
    
    x_scaled = x / scale ## nan
    x_scaled = tl.clamp(x_scaled, -448.0, 448.0)
    x_fp8 = tl.cast(x_scaled, tl.float8e4nv)
    
    scale_bf16 = tl.cast(scale, tl.bfloat16)

    tl.store(out_ptr + offs_x, x_fp8, mask=mask)
    tl.store(scale_ptr + pid, scale_bf16)
        
def triton_fp8_bf16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    N = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((*leading_shape,1), dtype=torch.bfloat16, device=x.device)

    bf16_to_fp8_act_dynamic_quant_kernel[(leading_shape.numel(),)](
        x, out, scale, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out, scale



@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def triton_act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def bf16_to_int8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_x = pid*N + offs
    mask = offs < N
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    #scale = max_val / 448.0
    scale = tl.where(
        max_val > 0,
        max_val / 127, # 448
        1.0
    )
    
    x_scaled = x / scale ## nan
    x_scaled = tl.clamp(x_scaled, -128.0, 127.0)
    x_int8 = tl.cast(x_scaled, tl.int8)
    
    scale_bf16 = tl.cast(scale, tl.bfloat16)

    tl.store(out_ptr + offs_x, x_int8, mask=mask)
    tl.store(scale_ptr + pid, scale_bf16)
        
def triton_int8_bf16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    N = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    out = torch.empty_like(x, dtype=torch.int8, device=x.device)
    scale = torch.empty((*leading_shape,1), dtype=torch.bfloat16, device=x.device)

    bf16_to_int8_act_dynamic_quant_kernel[(leading_shape.numel(),)](
        x, out, scale, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out, scale

@triton.jit
def fp16_to_int8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_x = pid*N + offs
    mask = offs < N
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    #scale = max_val / 448.0
    scale = tl.where(
        max_val > 0,
        max_val / 127, # 448
        1.0
    )
    
    x_scaled = x / scale ## nan
    x_scaled = tl.clamp(x_scaled, -128.0, 127.0)
    x_int8 = tl.cast(x_scaled, tl.int8)
    
    scale_fp16 = tl.cast(scale, tl.float16)

    tl.store(out_ptr + offs_x, x_int8, mask=mask)
    tl.store(scale_ptr + pid, scale_fp16)
        
def triton_int8_fp16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    N = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    out = torch.empty_like(x, dtype=torch.int8, device=x.device)
    scale = torch.empty((*leading_shape,1), dtype=torch.float16, device=x.device)

    fp16_to_int8_act_dynamic_quant_kernel[(leading_shape.numel(),)](
        x, out, scale, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out, scale



@triton.jit
def fp16_to_fp8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_x = pid*N + offs
    mask = offs < N
    x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    #scale = max_val / 448.0
    scale = tl.where(
        max_val > 0,
        max_val / 448.0, # 448
        1.0
    )

    # Clamp
    # MIN_SCALE = 1e-2
    # MAX_SCALE = 32.0
    # scale = tl.clamp(scale, MIN_SCALE, MAX_SCALE)
    
    x_scaled = x / scale ## nan
    x_scaled = tl.clamp(x_scaled, -448.0, 448.0)
    x_fp8 = tl.cast(x_scaled, tl.float8e4nv)
    
    scale_fp16 = tl.cast(scale, tl.float16)

    tl.store(out_ptr + offs_x, x_fp8, mask=mask)
    tl.store(scale_ptr + pid, scale_fp16)
        
def triton_fp8_fp16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    N = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((*leading_shape,1), dtype=torch.float16, device=x.device)

    fp16_to_fp8_act_dynamic_quant_kernel[(leading_shape.numel(),)](
        x, out, scale, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out, scale


def get_activation_fcn(qdtype=None, model_dtype=None, *args, **kwargs):

    return triton_fp8_bf16_quant
    


def _get_activation_fcn(qdtype=None, model_dtype=None, *args, **kwargs):
    qdtype = qdtype.upper()
    if qdtype in ["INT4_WITH_FP8", "SCALED_INT4_WITH_FP8", "INT4BF16_WITH_FP8"]:
        if model_dtype in [torch.half, torch.float16]:
            return triton_fp8_fp16_quant
        elif model_dtype in [torch.bfloat16]:
            return triton_fp8_bf16_quant
    elif dtype in ["INT4_WITH_INT8", "INT8_WITH_INT8"]:
        if model_dtype in [torch.half, torch.float16]:
            return triton_int8_fp16_quant
        elif model_dtype in [torch.bfloat16]:
            return triton_int8_bf16_quant
        
    raise NotImplementedError
  

