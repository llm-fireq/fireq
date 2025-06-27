import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Union, List, Tuple, Dict
import gc
from mcomp.utils.quant import (
    pseudo_group_quantize,
    pseudo_group_dequantize,
    pseudo_act_quantize_fp8
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


class F8CastSTE(Function):

    @staticmethod
    def forward(ctx, inp):
        return inp.to(torch.float8_e4m3fn)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_fp8_cast(x):
    return F8CastSTE.apply(x)


class PseudoQuantLinearFunctionI4F8_F8(Function):

    @staticmethod
    def forward(ctx, x, weight, w_scales, bias, group_size, dynamic_act_quant, act_scale, on_dtype):

        ctx.save_for_backward(x, weight, w_scales, bias)
        ctx.group_size = group_size

        # weight and scale quant
        weight = ste_round(weight, bit_width=4)
        weight = weight.to(on_dtype)
        w_scales = w_scales.to(torch.float8_e4m3fn).to(on_dtype)  
        
        if dynamic_act_quant:
            x_q, x_scales = pseudo_act_quantize_fp8(x, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)
        else:
            assert act_scale is not None
            assert x.shape[:-1] == act_scale.shape[:-1]
            x_q, x_scales = pseudo_act_quantize_fp8(x, act_scale, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)

        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        scale_expanded = w_scales.repeat_interleave(group_size, dim=1)
        w_deq = weight * scale_expanded
        
        output = x_scales * torch.matmul(x_q, w_deq.T)
        if bias is not None:
            output = output + bias
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, w_scales, bias = ctx.saved_tensors
        group_size = ctx.group_size
        
        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        scale_expanded = w_scales.repeat_interleave(group_size, dim=1)
        w_deq = weight * scale_expanded

        grad_x = grad_output @ w_deq
        
        #grad_weight_deq = grad_output.view(out_features, -1) @ x.view(-1, in_features) # out_features x in_features
        grad_weight_deq = grad_output.view(-1, out_features).t() @ x.view(-1, in_features)  # out_features x in_features
        grad_weight = grad_weight_deq*scale_expanded #

        grad_scales = (grad_weight_deq * weight).reshape(out_features, -1, group_size).sum(dim=-1)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(torch.arange(grad_output.dim()-1)))
            
        return grad_x, grad_weight, grad_scales, grad_bias, None, None, None, None
        
## group-wise - per token
class PseudoQuantLinearI4F8_F8(BaseCalbiratedLinear):

    weight_int_quantized = True
    linear_dtype = LinearDtype.I4F8_F8
    
    def __init__(self, linear:nn.Linear, group_size:int,  
                 dynamic_act_quant:bool=True, act_scales=None,
                calib_dtype:torch.dtype=torch.float32):
        super().__init__(linear, calib_dtype)
        self.group_size = group_size
        self.dynamic_act_quant = dynamic_act_quant
        self.act_scales = act_scales
        self.bit_width = 4
        
        self.weight_scale_dtype = torch.float8_e4m3fn

    def forward(self, x):

        return PseudoQuantLinearFunctionI4F8_F8.apply(
            x, 
            self.pqweight, 
            self.scales, 
            self.bias, 
            self.group_size, 
            self.dynamic_act_quant, 
            self.act_scales,
            self.on_dtype,
            )


    def apply_pseudo_quantize(self):

        _, scales, iweight = pseudo_group_quantize(self.pqweight, self.bit_width, self.group_size, scales_dt=torch.float8_e4m3fn) # torch.float8_e4m3fn

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
            
        self.pqweight.data = iweight


## group-wise - per token
class PseudoQuantLinearScaled_I4F8_F8(BaseCalbiratedLinear):
    
    multiplier = None
    weight_int_quantized = True
    linear_dtype = LinearDtype.SCALED_I4F8_F8

    def __init__(self, linear:nn.Linear, group_size:int,  
                 dynamic_act_quant:bool=True, act_scales=None,
                calib_dtype:torch.dtype=torch.float32):
        super().__init__(linear, calib_dtype)
        self.group_size = group_size
        self.dynamic_act_quant = dynamic_act_quant
        self.act_scales = act_scales
        self.bit_width = 4
        
        self.weight_scale_dtype = torch.float8_e4m3fn
        
        self.per_tensor_scale = nn.Parameter(torch.ones(1).to(
                dtype=self.orig_w_dtype,
                device=self.org_w_device
            ))

    def forward(self, x, restore_scale=True):

        y = PseudoQuantLinearFunctionI4F8_F8.apply(
            x, 
            self.pqweight, 
            self.scales, 
            self.bias, 
            self.group_size, 
            self.dynamic_act_quant, 
            self.act_scales,
            self.on_dtype,
            )
        if restore_scale:
            y = y/self.per_tensor_scale
        return y


    def apply_pseudo_quantize(self):

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        fp8_min = 2**-9
        weight_min = fp8_min * (2**(self.bit_width-1) - 1)
        converge_eps = 1e-3
        initial_test_power = 15
        
        # clamp, fp8 max
        with torch.no_grad():
            self.pqweight.clamp_(max=fp8_max)
        
        @torch.no_grad()
        def slack_sum(w, scale, group_size):
            w = w*scale
            gw = w.view(-1, group_size)
            gwa = gw.abs()
            gwa = gwa.amax(dim=1)
            return (weight_min - gwa[gwa < weight_min]).sum()/weight_min, (gwa[gwa > fp8_max] - fp8_max).sum()/(fp8_max)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        # @torch.no_grad()
        # def binary_search(w, group_size, a, b, pa, pb, iter_num=0, max_iter=100):
        #     m = (a+b)/2
            
        #     print(a,b)
            
        #     if iter_num > 0 and abs(a-pa) == 0 and abs(b-pb) == 0:
        #         return m
        #     if abs(b-a) < converge_eps or iter_num > max_iter:
        #         return m
                
        #     _lower, _upper = slack_sum(w, m, group_size)
        #     if _upper > 0:
        #         return binary_search(w, group_size, a, m, a, b, iter_num+1,)
        #     else:
        #         best_memory.append((m, _lower))
        #         return binary_search(w, group_size, m, b, a, b, iter_num+1,)
        
        best_memory = []    
        initial_list = []
        for i in range(initial_test_power): ## 
            _lower, _upper = slack_sum(self.pqweight, float(2**i), self.group_size)
            if _upper > 0: break
            initial_list.append((torch.tensor(float(2**i), dtype=torch.bfloat16), _lower))
        best_memory.extend(initial_list)

        #binary_search(self.pqweight, self.group_size, 1.0, fp8_max/ self.pqweight.abs().max(), 1.0, fp8_max/ self.pqweight.abs().max())
        if best_memory:
            #assert best_memory, f"maxvalue: {self.pqweight.data.max()}"
            best_memory.sort(key = lambda x: (x[1], x[0]))
            
            #self.per_tensor_scale.data = torch.Tensor([best_memory[0][0]]).to(device=self.org_w_device, dtype=self.orig_w_dtype)
            
            # per-tensor-scaling test
            if getattr(self, 'multiplier', None) is not None and len(best_memory) > self.multiplier:
                self.per_tensor_scale.data = torch.Tensor([best_memory[self.multiplier][0]]).to(device=self.org_w_device, dtype=self.orig_w_dtype)
            else:
                self.per_tensor_scale.data = torch.Tensor([best_memory[0][0]]).to(device=self.org_w_device, dtype=self.orig_w_dtype)
            
        else:
            self.per_tensor_scale.data = torch.Tensor([1.0]).to(device=self.org_w_device, dtype=self.orig_w_dtype)

        _, scales, iweight = pseudo_group_quantize(self.pqweight*self.per_tensor_scale, self.bit_width, self.group_size, scales_dt=torch.float8_e4m3fn) # torch.float8_e4m3fn

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
            
        self.pqweight.data = iweight
        if self.bias is not None:
            self.bias.data = self.bias.data*self.per_tensor_scale


## group-wise - per token
class PseudoQuantLinearFunctionI4_F8(Function):

    @staticmethod
    def forward(ctx, x, weight, w_scales, bias, group_size, dynamic_act_quant, act_scale, on_dtype):

        ctx.save_for_backward(x, weight, w_scales, bias)
        ctx.group_size = group_size

        # weight and scale quant
        weight = ste_round(weight, bit_width=4)
        weight = weight.to(on_dtype)
        w_scales = w_scales.to(torch.bfloat16).to(on_dtype)  
        
        if dynamic_act_quant:
            x_q, x_scales = pseudo_act_quantize_fp8(x, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)
        else:
            assert act_scale is not None
            assert x.shape[:-1] == act_scale.shape[:-1]
            x_q, x_scales = pseudo_act_quantize_fp8(x, act_scale, on_dtype=on_dtype, out_act_dtype=on_dtype, out_scales_dtype=on_dtype)

        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        scale_expanded = w_scales.repeat_interleave(group_size, dim=1)
        w_deq = weight * scale_expanded
        
        output = x_scales * torch.matmul(x_q, w_deq.T)
        if bias is not None:
            output = output + bias
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, w_scales, bias = ctx.saved_tensors
        group_size = ctx.group_size
        
        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        scale_expanded = w_scales.repeat_interleave(group_size, dim=1)
        w_deq = weight * scale_expanded

        grad_x = grad_output @ w_deq
        
        #grad_weight_deq = grad_output.view(out_features, -1) @ x.view(-1, in_features) # out_features x in_features
        grad_weight_deq = grad_output.view(-1, out_features).t() @ x.view(-1, in_features)  # out_features x in_features
        grad_weight = grad_weight_deq*scale_expanded #

        grad_scales = (grad_weight_deq * weight).reshape(out_features, -1, group_size).sum(dim=-1)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(torch.arange(grad_output.dim()-1)))
            
        return grad_x, grad_weight, grad_scales, grad_bias, None, None, None, None
        
## group-wise - per token
class PseudoQuantLinearI4_F8(BaseCalbiratedLinear):

    weight_int_quantized = True
    linear_dtype = LinearDtype.I4_F8

    def __init__(self, linear:nn.Linear, group_size:int,  
                 dynamic_act_quant:bool=True, act_scales=None,
                calib_dtype:torch.dtype=torch.float32):
        super().__init__(linear, calib_dtype)
        self.group_size = group_size
        self.dynamic_act_quant = dynamic_act_quant
        self.act_scales = act_scales
        self.bit_width = 4
        
    def forward(self, x):

        return PseudoQuantLinearFunctionI4_F8.apply(
            x, 
            self.pqweight, 
            self.scales, 
            self.bias, 
            self.group_size, 
            self.dynamic_act_quant, 
            self.act_scales,
            self.on_dtype,
            )


    def apply_pseudo_quantize(self):

        _, scales, iweight = pseudo_group_quantize(self.pqweight, self.bit_width, self.group_size, scales_dt=self.on_dtype)

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
            
        self.pqweight.data = iweight


## group-wise
class PseudoQuantLinearI4(BaseCalbiratedLinear):

    weight_int_quantized = True
    linear_dtype = LinearDtype.I4

    def __init__(self, linear:nn.Linear, group_size:int,  
                 dynamic_act_quant:bool=True, act_scales=None,
                calib_dtype:torch.dtype=torch.float32):
        super().__init__(linear, calib_dtype)
        self.group_size = group_size
        self.dynamic_act_quant = dynamic_act_quant
        self.act_scales = act_scales
        self.bit_width = 4
        
        #self.weight_scale_dtype = on_dtype

    def forward(self, x):
        deq_weight = pseudo_group_dequantize(self.pqweight, self.scales, on_dtype=self.on_dtype, out_dtype=self.on_dtype)

        y = x @ deq_weight.T
        
        if self.bias is not None:
            y = y + self.bias
        return y
        

    def apply_pseudo_quantize(self):

        _, scales, iweight = pseudo_group_quantize(self.pqweight, self.bit_width, self.group_size, scales_dt=self.on_dtype)

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
            
        self.pqweight.data = iweight


## Block-wise - per-token
# class PseudoQuantLinearFunctionF8_F8(Function):

#     @staticmethod
#     def forward(ctx, x, weight, w_scales, bias, dynamic_act_quant, act_scale, on_dtype):

#         ctx.save_for_backward(x, weight, w_scales, bias)
#         ctx.group_size = group_size

#         # weight and scale quant
#         weight = ste_fp8_cast(weight)
#         weight = weight.to(on_dtype)
        
#         if dynamic_act_quant:
#             x_q, x_scales = pseudo_act_quantize_fp8(x)
#         else:
#             assert act_scale is not None
#             assert x.shape[:-1] == act_scale.shape[:-1]
#             x_q, x_scales = pseudo_act_quantize_fp8(x, act_scale)
        
#         output = x_scales * torch.matmul(x_q, weight.T) * w_scales
#         if bias is not None:
#             output = output + bias
            
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, weight, w_scales, bias = ctx.saved_tensors
        
#         out_features, in_features = weight.shape

#         grad_x = grad_output @ weight
        
#         #grad_weight_deq = grad_output.view(out_features, -1) @ x.view(-1, in_features) # out_features x in_features
#         grad_weight = grad_output.view(-1, out_features).t() @ x.view(-1, in_features)  # out_features x in_features

#         grad_scales = (grad_weight * weight).reshape(out_features, -1, group_size).sum(dim=-1)

#         grad_bias = None
#         if bias is not None:
#             grad_bias = grad_output.sum(dim=tuple(torch.arange(grad_output.dim()-1)))
            
#         return grad_x, grad_weight, grad_scales, grad_bias, None, None, None
        
        
# class PseudoQuantLinearF8_F8(BaseCalbiratedLinear):
    
#     def __init__(self, linear:nn.Linear, group_size:int,  
#                  dynamic_act_quant:bool=True, act_scales=None,
#                  calib_dtype:torch.dtype=torch.float32):
#         super().__init__(linear, calib_dtype)
#         self.group_size = group_size
#         self.dynamic_act_quant = dynamic_act_quant
#         self.act_scales = act_scales

#     def forward(self, x):

#         return PseudoQuantLinearFunctionF8_F8.apply(
#             x, 
#             self.pqweight, 
#             self.scales, 
#             self.bias, 
#             self.dynamic_act_quant, 
#             self.act_scales,
#             self.on_dtype,
#             )


#     def apply_pseudo_quantize(self):

#         _, scales, iweight = pseudo_group_quantize(self.pqweight, self.bit_width, self.group_size, scales_dt=self.on_dtype)

#         if self.scales is None:
#             del self.scales
#             self.scales = nn.Parameter(
#                 scales.detach().clone().to(
#                                     dtype=self.orig_w_dtype,
#                                     device=self.org_w_device
#                                  ),
#             )
#         else:
#             self.scales.data = scales
            
#         self.pqweight.data = iweight
