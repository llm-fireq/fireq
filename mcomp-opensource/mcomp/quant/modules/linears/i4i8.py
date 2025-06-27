import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Union, List, Tuple, Dict
import gc
from mcomp.utils.quant import (
    pseudo_int_per_ch_quantize,
    pseudo_act_quantize_int,
    
)

from .base import BaseCalbiratedLinear
from mcomp.config.const import LinearDtype

class RoundSTE(Function):

    @staticmethod
    def forward(ctx, inp, bit_width):
        max_int = 2 ** (bit_width - 1) - 1
        min_int = -(2 ** (bit_width - 1))
        return torch.clamp(torch.round(inp), min_int, max_int)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
        
def ste_round(x, bit_width=4):
    return RoundSTE.apply(x, bit_width)


class PseudoQuantLinearFunctionI4_I8(Function):

    @staticmethod
    def forward(ctx, x, weight, w_scales, bias, dynamic_act_quant, act_scale, on_dtype):

        ctx.save_for_backward(x, weight, w_scales, bias)

        # weight and scale quant
        weight = ste_round(weight, bit_width=4)
        weight = weight.to(on_dtype)
        w_scales = w_scales.to(on_dtype)
        if dynamic_act_quant:
            x_q, x_scales = pseudo_act_quantize_int(x, bits=8, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)
        else:
            assert act_scale is not None
            assert x.shape[:-1] == act_scale.shape[:-1]
            x_q, x_scales = pseudo_act_quantize_int(x, act_scale, bits=8, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)

        out_features, in_features = weight.shape
        
        output = x_scales * torch.matmul(x_q, weight.T) * w_scales
        if bias is not None:
            output = output + bias
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, w_scales, bias = ctx.saved_tensors
        
        out_features, in_features = weight.shape

        grad_x = grad_output @ weight
        
        #grad_weight_deq = grad_output.view(out_features, -1) @ x.view(-1, in_features) # out_features x in_features
        grad_weight_deq = grad_output.view(-1, out_features).t() @ x.view(-1, in_features)  # out_features x in_features
        grad_weight = grad_weight_deq*w_scales #

        grad_scales = (grad_weight_deq * weight).reshape(out_features, -1).sum(dim=-1) #?!

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(torch.arange(grad_output.dim()-1)))
            
        return grad_x, grad_weight, grad_scales, grad_bias, None, None, None
        
        
class PseudoQuantLinearI4_I8(BaseCalbiratedLinear):

    weight_int_quantized = True
    linear_dtype = LinearDtype.I4_I8
    
    def __init__(self, linear:nn.Linear,
                dynamic_act_quant:bool=True, act_scales=None,
                calib_dtype:torch.dtype=torch.float32):
        super().__init__(linear, calib_dtype)
        self.dynamic_act_quant = dynamic_act_quant
        self.act_scales = act_scales
        self.bit_width = 4
        
        self.group_size = linear.weight.data.shape[-1]
        
        #self.weight_scale_dtype = on_dtype

    def forward(self, x):

        return PseudoQuantLinearFunctionI4_I8.apply(
            x, 
            self.pqweight, 
            self.scales, 
            self.bias, 
            self.dynamic_act_quant, 
            self.act_scales,
            self.on_dtype,
            )


    def apply_pseudo_quantize(self):

        _, scales, iweight = pseudo_int_per_ch_quantize(self.pqweight, self.bit_width)

        if self.scales is None:
            del self.scales
            self.scales = nn.Parameter(
                scales.detach().clone().to(
                                    dtype=self.orig_w_dtype,
                                    device=self.org_w_device
                                 ),
            )
        else:
            self.scales.data = scales
            
        self.pqweight.data = iweight.to(self.orig_w_dtype)
