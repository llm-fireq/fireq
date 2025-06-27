import torch
import triton
import triton.language as tl


@triton.jit
def i4bf16_quantize_pack_save_kernel(
    input_ptr, output_ptr, scale_ptr, start_pos,
    B, H, S, max_cache_len,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Decode 1D program ID into (b, h, s)
    total_HS = H * S
    b = pid // total_HS
    hs_idx = pid % total_HS
    h = hs_idx // S
    s = hs_idx % S
    real_s = start_pos + s

    # Pointers offset
    input_offset = ((b * H + h) * S + s) * BLOCK_SIZE
    output_offset = ((b * H + h) * max_cache_len + real_s) * (BLOCK_SIZE // 8)
    scale_offset = ((b * H + h) * max_cache_len + real_s) * (BLOCK_SIZE // group_size)

    # Processing D dimension with BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < BLOCK_SIZE

    # Load bf16 input values as fp32
    x = tl.load(input_ptr + input_offset + offs, mask=mask, other=0.0)

    # Initialize output buffer
    q = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Groupwise quantization
    for g in range(0, BLOCK_SIZE, group_size):
        group_offs = offs + g
        group_mask = group_offs < BLOCK_SIZE
        
        group_vals = tl.load(input_ptr + input_offset + group_offs, mask=group_mask, other=0.0)
        
        abs_vals = tl.abs(group_vals)
        absmax = tl.max(abs_vals, axis=0)

        scale = absmax / 7.0
        scale = tl.where(absmax == 0, 1.0, scale)  # avoid div0

        # save scale
        group_id = g // group_size
        tl.store(scale_ptr + scale_offset + group_id, scale.to(tl.bfloat16))

        # quantize
        q_group = group_vals / scale
        q_group = tl.where(q_group < -8.0, -8.0, q_group)
        q_group = tl.where(q_group > 7.0, 7.0, q_group)
        #q_group = tl.round(q_group)
        q_group = q_group.to(tl.int32)

        q = tl.where(group_offs < BLOCK_SIZE, q_group, q)

    # Now Pack int4
    # each 8 elements -> 1 int32
    offs_packed = tl.arange(0, BLOCK_SIZE // 8)
    q8 = q.reshape([BLOCK_SIZE // 8, 8]) # [Npack, 8]
    
    shifts = tl.arange(0, 8) * 4
    q4 = tl.where(q8 < 0, (q8 + 16) & 0xF, q8 & 0xF)
    
    packed = tl.sum(q4 << shifts[None, :], axis=1)

    # Save packed int32
    tl.store(output_ptr + output_offset + offs_packed, packed)


def triton_i4bf16_quantize_pack_save(
    input_tensor, output_tensor, scale_tensor, start_pos, group_size, max_cache_len
):
    assert input_tensor.dtype == torch.bfloat16

    B, H, S, D = input_tensor.shape

    n_programs = B * H * S

    grid = lambda meta: (n_programs,)

    i4bf16_quantize_pack_save_kernel[grid](
        input_tensor,
        output_tensor,
        scale_tensor,
        start_pos,
        B, H, S, max_cache_len,
        group_size,
        BLOCK_SIZE=128
    )
