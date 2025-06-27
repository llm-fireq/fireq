import torch
import torch.nn as nn
from mcomp.quant.modules.linears import (
    PseudoQuantLinearI4_I8, 
    PseudoQuantLinearI4F8_F8, 
    PseudoQuantLinearScaled_I4F8_F8, 
    PseudoQuantLinearI4_F8, 
    PseudoQuantLinearI4,
)
from mcomp.utils.quant import (
    pseudo_act_quantize_fp8, 
    pseudo_act_quantize_int,
)

def get_pseudolinear(dtype_key, linear:nn.Linear, group_size=None, dynamic_act_quant=None, calib_dtype=None, **kwargs):

    if dtype_key is None:
        dtype_key = 'PRISTINE' # Assigned when quantizer is declared.
    assert isinstance(dtype_key, str)
    
    dtype_key = dtype_key.upper()
    if dtype_key == 'I4F8_F8':
        return PseudoQuantLinearI4F8_F8(linear, group_size=group_size, dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
    elif dtype_key == 'SCALED_I4F8_F8':
        return PseudoQuantLinearScaled_I4F8_F8(linear, group_size=group_size, dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
    elif dtype_key == "I4_F8":
        return PseudoQuantLinearI4_F8(linear, group_size=group_size, dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
    elif dtype_key == "I4":
        return PseudoQuantLinearI4(linear, group_size=group_size, dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
    elif dtype_key == 'I4_I8':
        return PseudoQuantLinearI4_I8(linear, dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
    else:
        return linear

def get_pseudo_act_quantizer(dtype_key, *args, **kwargs):
    ## It depends on Attn Implementation.
    if dtype_key is None:
        dtype_key = 'PRISTINE'
    assert isinstance(dtype_key, str)
    
    dtype_key = dtype_key.upper()
    if dtype_key == 'I4F8_F8':
        return pseudo_act_quantize_fp8
    elif dtype_key == 'SCALED_I4F8_F8':
        return pseudo_act_quantize_fp8
    elif dtype_key == "I4_F8":
        return pseudo_act_quantize_fp8
    elif dtype_key == "I4":
        return None
    elif dtype_key == 'I4_I8':
        return pseudo_act_quantize_int 
    else:
        return None