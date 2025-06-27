import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()


@triton.jit
def _bf16_to_fp8_weight_quant_per_ch_kernel(
    w_ptr, out_ptr, scale_ptr, w_stridem, o_stridem, s_stridem,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_w = pid*w_stridem + offs
    
    mask = offs < N
    w = tl.load(w_ptr + offs_w, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
    abs_w = tl.abs(w)
    max_val = tl.max(abs_w, axis=0)
    
    scale = tl.where(
        max_val > 0,
        max_val / 448.0, # 448
        1.0
    ).to(tl.float32)
    
    w_scaled = w / scale
    w_scaled = tl.clamp(w_scaled, -448.0, 448.0)
    w_fp8 = tl.cast(w_scaled, tl.float8e4nv)
    
    scale_bf16 = tl.cast(scale, tl.bfloat16)

    offs_out = pid*o_stridem + offs
    offs_scale = pid*s_stridem

    tl.store(out_ptr + offs_out, w_fp8, mask=mask)
    tl.store(scale_ptr + offs_scale, scale_bf16)
        

@triton.jit
def _fp16_to_fp8_weight_quant_per_ch_kernel(
    w_ptr, out_ptr, scale_ptr, w_stridem, o_stridem, s_stridem,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    offs_w = pid*w_stridem + offs
    
    mask = offs < N
    w = tl.load(w_ptr + offs_w, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
    abs_w = tl.abs(w)
    max_val = tl.max(abs_w, axis=0)
    
    scale = tl.where(
        max_val > 0,
        max_val / 448.0, # 448
        1.0
    ).to(tl.float32)
    
    w_scaled = w / scale
    w_scaled = tl.clamp(w_scaled, -448.0, 448.0)
    w_fp8 = tl.cast(w_scaled, tl.float8e4nv)
    
    scale_fp16 = tl.cast(scale, tl.float16)

    offs_out = pid*o_stridem + offs
    offs_scale = pid*s_stridem

    tl.store(out_ptr + offs_out, w_fp8, mask=mask)
    tl.store(scale_ptr + offs_scale, scale_fp16)
        
def triton_f16_fp8_weight_quant_per_ch(w):

    M, N = w.shape
    w_dtype = w.dtype
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    out = torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((M,1), dtype=w_dtype, device=w.device)

    grid = lambda meta: (M, )

    if w_dtype == torch.bfloat16:
        _bf16_to_fp8_weight_quant_per_ch_kernel[grid](
            w, out, scale, 
            w.stride(0), out.stride(0), scale.stride(0),
            M, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, 
            num_stages=2,
        )
    elif w_dtype == torch.float16:
        _fp16_to_fp8_weight_quant_per_ch_kernel[grid](
            w, out, scale, 
            w.stride(0), out.stride(0), scale.stride(0),
            M, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, 
            num_stages=2,
        )

    return out, scale

@triton.jit
def _bf16_to_int8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, x_stride, N, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D

    for x_index in tl.range(pid, N, pnum):
        offs_x = x_index*x_stride + offs_d
        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        
        abs_x = tl.abs(x)
        max_val = tl.max(abs_x, axis=0)
        
        scale = tl.where(
            max_val > 0,
            max_val / 127,
            1.0
        )
    
        x_scaled = x / scale ## nan
        x_scaled = tl.clamp(x_scaled, -128.0, 127.0)
        x_int8 = tl.cast(x_scaled, tl.int8)
        
        scale_bf16 = tl.cast(scale, tl.bfloat16)
    
        tl.store(out_ptr + offs_x, x_int8, mask=mask)
        tl.store(scale_ptr + x_index, scale_bf16)
        

@triton.jit
def _fp16_to_int8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, x_stride, N, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D

    for x_index in tl.range(pid, N, pnum):
        offs_x = x_index*x_stride + offs_d
        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        
        abs_x = tl.abs(x)
        max_val = tl.max(abs_x, axis=0)
        
        scale = tl.where(
            max_val > 0,
            max_val / 127,
            1.0
        )
    
        x_scaled = x / scale ## nan
        x_scaled = tl.clamp(x_scaled, -128.0, 127.0)
        x_int8 = tl.cast(x_scaled, tl.int8)
        
        scale_bf16 = tl.cast(scale, tl.bfloat16)
    
        tl.store(out_ptr + offs_x, x_int8, mask=mask)
        tl.store(scale_ptr + x_index, scale_bf16)
        
        
def triton_int8_f16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    x_dtype = x.dtype
    D = x.shape[-1]
    N = leading_shape.numel()
    BLOCK_SIZE = triton.next_power_of_2(D)
    
    out = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((*leading_shape,1), dtype=x_dtype, device=x.device)

    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    kernel = _bf16_to_int8_act_dynamic_quant_kernel.warmup(x, out, scale, D, N, D, BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()

    occupancy = NUM_REGS // (kernel.n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // kernel.metadata.shared)
    
    num_programs = min(NUM_SM * occupancy, N)
    grid = lambda meta: (num_programs,)
    
    if x_dtype == torch.bfloat16:
        _bf16_to_int8_act_dynamic_quant_kernel[grid](
            x, out, scale, D, N, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    elif x_dtype == torch.float16:   
        _fp16_to_int8_act_dynamic_quant_kernel[grid](
            x, out, scale, D, N, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    return out, scale



@triton.jit
def bf16_to_fp8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, x_stride, N, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D
    
    for row_idx in tl.range(pid, N, pnum):
        
        offs_x = row_idx*x_stride + offs_d
        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
        abs_x = tl.abs(x)
        max_val = tl.max(abs_x, axis=0)
    
        #scale = max_val / 448.0
        scale = tl.where(
            max_val > 0,
            max_val / 448.0, # 448
            1.0
        )
        x_scaled = x / scale ## nan
        x_scaled = tl.clamp(x_scaled, -448.0, 448.0)
        x_fp8 = tl.cast(x_scaled, tl.float8e4nv)
    
        scale_bf16 = tl.cast(scale, tl.bfloat16)

        tl.store(out_ptr + offs_x, x_fp8, mask=mask)
        tl.store(scale_ptr + row_idx, scale_bf16)
        
@triton.jit
def fp16_to_fp8_act_dynamic_quant_kernel(
    x_ptr, out_ptr, scale_ptr, x_stride, N, D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D
    
    for row_idx in tl.range(pid, N, pnum):
        
        offs_x = row_idx*x_stride + offs_d
        x = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32) #.to(tl.float32)
    
        abs_x = tl.abs(x)
        max_val = tl.max(abs_x, axis=0)
    
        #scale = max_val / 448.0
        scale = tl.where(
            max_val > 0,
            max_val / 448.0, # 448
            1.0
        )
        x_scaled = x / scale ## nan
        x_scaled = tl.clamp(x_scaled, -448.0, 448.0)
        x_fp8 = tl.cast(x_scaled, tl.float8e4nv)
    
        scale_bf16 = tl.cast(scale, tl.float16)

        tl.store(out_ptr + offs_x, x_fp8, mask=mask)
        tl.store(scale_ptr + row_idx, scale_bf16)
        
def triton_fp8_f16_quant(x):
    #x = x.contiguous() #assumed
    leading_shape = x.shape[:-1]
    N = leading_shape.numel()
    D = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(D)
    x_dtype = x.dtype
    
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((*leading_shape,1), dtype=x_dtype, device=x.device)

    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    kernel = bf16_to_fp8_act_dynamic_quant_kernel.warmup(x, out, scale, D, N, D, BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()

    occupancy = NUM_REGS // (kernel.n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // kernel.metadata.shared)
    
    num_programs = min(NUM_SM * occupancy, N)
    grid = lambda meta: (num_programs,)

    if x_dtype == torch.bfloat16:
        bf16_to_fp8_act_dynamic_quant_kernel[grid](
            x, out, scale, D, N, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        fp16_to_fp8_act_dynamic_quant_kernel[grid](
            x, out, scale, D, N, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    return out, scale

