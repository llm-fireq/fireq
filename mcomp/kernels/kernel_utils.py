import torch

from mcomp.utils.module.gemm import (
    fp8_bf16_matmul_,
    fp8_fp16_matmul_,
    fp8_matmul_
)

from mcomp.kernels.triton import (
    
    triton_dequantize_int8_bf16,
    triton_dequantize_int8_fp16,
    triton_dequantize_int8_fp8,
    
    # triton_int8_fp16_quant,
    # triton_int8_bf16_quant,
    
    # triton_fp8_bf16_quant,
    # triton_fp8_fp16_quant,
    
    triton_f16_fp8_weight_quant_per_ch,
    
    triton_int8_f16_quant,
    triton_fp8_f16_quant,
    triton_fp8_matmul_fp8f16xfp8,
    triton_fp8_matmul_fp8f16xfp8f16_per_tok_per_ch,
    triton_int8_matmul_i8f16xi8f16_per_tok_per_ch,
)

def get_matmul_kernel(dtype_key, model_dtype, *args, **kwargs):
    
    dtype_key = dtype_key.upper()
    dtype_key_list = dtype_key.split('_')
    if dtype_key == 'I4_F8':
        return triton_fp8_matmul_fp8f16xfp8f16_per_tok_per_ch
    elif len(dtype_key_list) > 1 and dtype_key_list[-1] == 'F8':
        return triton_fp8_matmul_fp8f16xfp8
    elif dtype_key == 'I4_I8':
        return triton_int8_matmul_i8f16xi8f16_per_tok_per_ch
        # if model_dtype == torch.float16:
        #     return fp8_fp16_matmul_
        # elif model_dtype == torch.bfloat16:
        #     return fp8_bf16_matmul_


def get_act_quantizer_kernel(dtype_key, model_dtype, *args, **kwargs):
    dtype_key = dtype_key.upper()
    dtype_key_list = dtype_key.split('_')
    assert len(dtype_key_list) > 1
    assert dtype_key_list[-1] in ['F8', 'I8']
    if dtype_key_list[-1] == 'F8':
        return triton_fp8_f16_quant
        # if model_dtype == torch.float16:
        #     return triton_fp8_fp16_quant
        # elif model_dtype == torch.bfloat16:
        #     return triton_fp8_bf16_quant
    elif dtype_key_list[-1] == 'I8':
        return triton_int8_f16_quant
        # if model_dtype == torch.float16:
        #     return triton_int8_fp16_quant
        # elif model_dtype == torch.bfloat16:
        #     return triton_int8_bf16_quant 
        
def get_quant_weight_kernel(dtype_key, model_dtype, *args, **kwargs):
    dtype_key = dtype_key.upper()
    assert dtype_key in ['I4_F8', ]
    if dtype_key == 'I4_F8':
        return triton_f16_fp8_weight_quant_per_ch
        # if model_dtype == torch.float16:
        #     return triton_fp16_fp8_weight_per_ch_quant
        # elif model_dtype == torch.bfloat16:
        #     return triton_bf16_fp8_weight_per_ch_quant

def get_dequant_weight_kernel(dtype_key, model_dtype, *args, **kwargs):
    dtype_key = dtype_key.upper()
    dtype_key_list = dtype_key.split('_')
    assert len(dtype_key_list) > 0
    weight_dtype = dtype_key_list[0]
    if weight_dtype == 'SCALED':
        weight_dtype = dtype_key_list[1]
    assert weight_dtype in ['I4', 'I4F8']
    
    if weight_dtype in ['I4F8']:
        return triton_dequantize_int8_fp8
    elif weight_dtype == 'I4':
        if model_dtype == torch.float16:
            return triton_dequantize_int8_fp16
        elif model_dtype == torch.bfloat16:
            return triton_dequantize_int8_bf16