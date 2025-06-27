import torch

def construct_module(module, construct_cls):
    for n,m in module.named_modules():
        if isinstance(m, construct_cls):
            m.construct()

def pseudo_group_quantize(weight, n_bit, group_size, scales_dt=None): # symmetric quant
    org_w_shape = weight.shape
    org_dt = weight.dtype
    
    weight = weight.reshape(-1, group_size)
    max_val = weight.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    scales = max_val / max_int
    
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(weight).sum() == 0

    if scales_dt is not None and scales_dt == torch.float8_e4m3fn:

        smallest_scale = 2 ** -6 * (1/8)
        #smallest_scale = torch.finfo(scales_dt).smallest_normal
        scales = scales.to(scales_dt).to(org_dt)
        scales[scales < smallest_scale] = smallest_scale
    
    iweight = torch.clamp(torch.round(weight/scales), min_int, max_int)
    deqweight = iweight*scales
    
    iweight = iweight.reshape(org_w_shape)
    deqweight = deqweight.reshape(org_w_shape)
    scales = scales.view(org_w_shape[0], -1)

    return deqweight, scales, iweight

def pseudo_group_dequantize(weight, scales, on_dtype=None, out_dtype=None):
    
    org_w_shape = weight.shape
    org_s_shape = scales.shape
    group_size = org_w_shape[1] // org_s_shape[1]

    weight = weight.reshape(-1, group_size)
    scales = scales.view(-1,1)

    if on_dtype is not None:
        weight = weight.to(on_dtype)
        scales = scales.to(on_dtype)
        
    deqw = weight*scales

    if out_dtype is not None:
        deqw = deqw.to(out_dtype)

    return deqw.reshape(org_w_shape)

def pseudo_act_quantize_fp8(x, scales=None, on_dtype=torch.bfloat16, out_act_dtype=torch.bfloat16, out_scales_dtype=torch.bfloat16): 

    x = x.to(on_dtype)
    if scales is None:
        x_scale = x.abs().max(dim=-1, keepdim=True).values / 448.0 # torch.finfo(torch.float8_e4m3fn).max
    else:
        x_scale = scales.to(dtype=on_dtype, device=x.device)

    qx = (x / x_scale).to(torch.float8_e4m3fn).to(out_act_dtype)
    x_scale = x_scale.to(out_scales_dtype)

    return qx, x_scale


def pseudo_int_per_ch_quantize(weight, n_bit): # symmetric quant
    org_w_shape = weight.shape
    
    max_val = weight.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    scales = max_val / max_int
    
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(weight).sum() == 0
    
    iweight = torch.clamp(torch.round(weight/scales), min_int, max_int)
    deqweight = iweight*scales
    
    iweight = iweight.reshape(org_w_shape)
    deqweight = deqweight.reshape(org_w_shape)
    scales = scales.view(-1)

    return deqweight, scales, iweight


def pseudo_act_quantize_int(x, scales=None, bits=8, on_dtype=torch.bfloat16, out_act_dtype=torch.bfloat16, out_scales_dtype=torch.bfloat16): 
    #x_dtype = x.dtype
    x_absmax = x.abs().amax(dim=-1, keepdim=True)
    max_int = 2**(bits-1)-1
    min_int = -2**(bits-1)

    if scales is None:
        scales = x_absmax / max_int
    else:
        scales = scales.to(dtype=on_dtype, device=x.device)

    assert torch.isnan(scales).sum() == 0
    qx = torch.round(x/scales).clamp(min=min_int, max=max_int).to(out_act_dtype)
    scales = scales.to(out_scales_dtype)

    return qx, scales


def pseudo_f8_block_quantize(w, block_size, on_dtype=None):
    assert w.ndim == 2
    M, N = w.shape

    orig_wdtype = w.dtype
    if on_dtype is None:
        on_dtype = w.dtype

    w = w.to(torch.float32)

    quantized = torch.empty_like(w, dtype=on_dtype)
    dequantized = torch.empty_like(w, dtype=on_dtype)
    scales = torch.empty((M // block_size, N // block_size), dtype=on_dtype)

    qmax = torch.finfo(torch.float8_e4m3fn).max
    qmin = torch.finfo(torch.float8_e4m3fn).min

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = w[i:i+block_size, j:j+block_size]
            max_val = block.abs().max()
            scale = max_val / qmax if max_val > 0 else 1.0

            q_block = torch.clamp((block/scale), qmin, qmax).to(torch.float8_e4m3fn).to(on_dtype)
            quantized[i:i+block_size, j:j+block_size] = q_block
            scales[i // block_size, j // block_size] = scale.to(on_dtype)
            dequantized[i:i+block_size, j:j+block_size] = q_block*scale

    return dequantized, quantized, scales

