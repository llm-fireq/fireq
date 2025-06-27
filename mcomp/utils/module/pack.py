import torch
import triton
import triton.language as tl

__all__ = [
  "pack_int4_to_int32",
  "unpack_int32_to_int4",
  "triton_pack_int4_to_int32",
  "triton_unpack_int32_to_int4"    
]

def pack_int4_to_int32(weight:torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.int8
    assert weight.dim() == 2
    assert weight.shape[1] % 8 == 0

    orig_shape = weight.shape

    reshaped = weight.view(*orig_shape[:-1], -1, 8)
    packed = torch.zeros(*reshaped.shape[:-1], dtype=torch.int32, device=weight.device)
    
    for i in range(8):
        val = reshaped[...,i]
        val = val & 0x0F
        shift = i*4
        packed |= (val.to(torch.int32) << shift)

    return packed
    
def unpack_int32_to_int4(packed:torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.int32
    orig_shape = packed.shape
    unpacked = torch.zeros(*orig_shape[:-1], orig_shape[-1]*8, dtype=torch.int8, device=packed.device)

    for i in range(8):
        shift = i*4
        
        # val = (packed >> shift) & 0xF
        # signed_val = val.clone()
        # signed_val[val >= 8] -= 16
        
        signed_val = (packed >> shift) & 0xF
        signed_val[signed_val >= 8] -= 16
        
        unpacked[..., i::8] = signed_val.to(torch.int8)
        
    return unpacked


@triton.jit
def pack_int4_to_int32_kernel(in_ptr, out_ptr, k, stride_in, stride_out, BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    in_row = in_ptr + pid * stride_in
    out_row = out_ptr + pid *stride_out

    offsets = tl.arange(0, BLOCK)
    cols = offsets * 8

    for i in range(BLOCK):
        idx = i * 8
        int4_vals = tl.load(in_row + idx + tl.arange(0,8), mask=idx + tl.arange(0,8) < k, other=0)
        int4_vals = int4_vals & 0xF
        shifts = tl.arange(0,8) * 4
        packed = tl.sum(int4_vals.to(tl.int32) << shifts, axis=0)
        tl.store(out_row+i, packed)

def triton_pack_int4_to_int32(weight:torch.Tensor):
    assert weight.dtype == torch.int8
    assert weight.dim() == 2
    N, K = weight.shape
    assert K % 8 == 0

    packed = torch.empty((N, K//8), dtype=torch.int32, device=weight.device)

    grid = lambda meta: (N, )
    pack_int4_to_int32_kernel[grid](weight, 
                          packed, 
                          K, 
                          weight.stride(0), 
                          packed.stride(0),
                          BLOCK=K//8)
    return packed

@triton.jit
def unpack_int32_to_int4_kernel(in_ptr, out_ptr, k,
                                stride_in, stride_out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    in_row = in_ptr + pid * stride_in
    out_row = out_ptr + pid * stride_out

    for i in range(BLOCK):
        packed = tl.load(in_row + i)
        int4_vals = (packed >> (tl.arange(0,8) * 4)) & 0xF
        int4_vals = int4_vals.to(tl.int8)
        needs_fix = int4_vals >= 8
        int4_vals = tl.where(needs_fix, int4_vals - 16, int4_vals)
        tl.store(out_row + i*8 + tl.arange(0,8),int4_vals)

def triton_unpack_int32_to_int4(packed:torch.Tensor)->torch.Tensor:
    assert packed.dtype == torch.int32
    N, K = packed.shape
    unpacked = torch.empty((N,K*8), dtype=torch.int8, device=packed.device)
    grid = lambda meta: (N, )
    
    unpack_int32_to_int4_kernel[grid](packed, unpacked, K*8, packed.stride(0), 
                                      unpacked.stride(0), BLOCK=K)
    return unpacked

    
