import torch
import torch.nn as nn
from mcomp.config.const import _SCALE_DTYPE_MAP

from mcomp.modules.linears import (
    LinearScaled_I4F8_F8, 
    LinearI4F8_F8, 
    LinearI4_F8, 
    LinearI4,
    LinearI4_I8,
)

from mcomp.kernels.kernel_utils import (
    get_matmul_kernel,
    get_act_quantizer_kernel,
    get_quant_weight_kernel,
    get_dequant_weight_kernel,
)

def get_linear(dtype_key, in_features, out_features, bias=None, group_size=128, model_dtype=torch.bfloat16):

    if dtype_key is None:
        dtype_key = 'PRISTINE' # Assigned when quantizer is declared.
    assert isinstance(dtype_key, str)
    
    dtype_key = dtype_key.upper()
    if dtype_key == 'I4F8_F8':        
        new_linear = LinearI4F8_F8(in_features, out_features, bias=bias, group_size=group_size, dtype=model_dtype)
        new_linear.matmul_kernel = get_matmul_kernel(dtype_key, model_dtype)
        new_linear.act_kernel = get_act_quantizer_kernel(dtype_key, model_dtype) 
        new_linear.weight_dequant_kernel = get_dequant_weight_kernel(dtype_key, model_dtype) 
        return new_linear
    elif dtype_key == 'SCALED_I4F8_F8':
        new_linear = LinearScaled_I4F8_F8(in_features, out_features, bias=bias, group_size=group_size, dtype=model_dtype)
        new_linear.matmul_kernel = get_matmul_kernel(dtype_key, model_dtype)
        new_linear.act_kernel = get_act_quantizer_kernel(dtype_key, model_dtype) 
        new_linear.weight_dequant_kernel = get_dequant_weight_kernel(dtype_key, model_dtype) 
        return new_linear
    elif dtype_key == 'I4_F8':
        new_linear = LinearI4_F8(in_features, out_features, bias=bias, group_size=group_size, dtype=model_dtype)
        new_linear.matmul_kernel = get_matmul_kernel(dtype_key, model_dtype)
        new_linear.act_kernel = get_act_quantizer_kernel(dtype_key, model_dtype) 
        new_linear.weight_quant_kernel = get_quant_weight_kernel(dtype_key, model_dtype)
        new_linear.weight_dequant_kernel = get_dequant_weight_kernel(dtype_key, model_dtype)
        return new_linear
    elif dtype_key == 'I4':
        new_linear =  LinearI4(in_features, out_features, bias=bias, group_size=group_size, dtype=model_dtype)
        new_linear.weight_dequant_kernel = get_dequant_weight_kernel(dtype_key, model_dtype) 
        return new_linear
    elif dtype_key == 'I4_I8':
        new_linear =  LinearI4_I8(in_features, out_features, bias=bias)
        new_linear.act_kernel = get_act_quantizer_kernel(dtype_key, model_dtype) 
        new_linear.matmul_kernel = get_matmul_kernel(dtype_key, model_dtype)
        return new_linear
    else:
        torch_dtype = _SCALE_DTYPE_MAP.get(dtype_key, torch.bfloat16) ## default torch.bfloat16
        return nn.Linear(in_features, out_features, bias, dtype=torch_dtype)

