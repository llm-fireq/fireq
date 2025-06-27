import torch
import torch.nn as nn
from torch.autograd import Function
import triton
import triton.language as tl
from .base import BaseLinearModule

from mcomp.utils.module import (
    pack_int4_to_int32,
    triton_unpack_int32_to_int4,
    dequantize_int8_fp8,
    triton_fp8_bf16_quant,
    #triton_dequantize_int8_fp8,
    triton_act_quant,
)

from mcomp.kernels.triton import (
    triton_dequantize_int8_bf16,
    triton_dequantize_int8_fp16,
    triton_dequantize_int8_fp8,
)

#from mcomp.utils.quant import pseudo_act_quantize_fp8
# from mcomp.utils.module.gemm import (
#     fp8_bf16_matmul_,
#     fp8_fp16_matmul_,
#     fp8_matmul_
# )

from mcomp.quant.modules.linears import (
    PseudoQuantLinearI4F8_F8, 
    PseudoQuantLinearScaled_I4F8_F8, 
    PseudoQuantLinearI4_F8, 
    PseudoQuantLinearI4,
)

from mcomp.kernels.kernel_utils import (
    get_matmul_kernel,
    get_act_quantizer_kernel,
    get_quant_weight_kernel,
    get_dequant_weight_kernel,
)

class LinearI4F8_F8(BaseLinearModule):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        self.matmul_kernel = None
        self.act_kernel = None
        self.weight_dequant_kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(torch.float8_e4m3fn))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        ## TODO Branching along with Devices
        x_fp8, x_scale_f16 = self.act_kernel(x)  # triton_fp8_f16_quant

        iweight = triton_unpack_int32_to_int4(self.weight)
        deq_weight = self.weight_dequant_kernel(iweight, self.weight_scale, self.group_size) #triton_dequantize_int8_fp8

        y = self.matmul_kernel(x_fp8, x_scale_f16, deq_weight.T) # triton_fp8_matmul_fp8f16xfp8
        #y = y*x_scale_f16
        if self.bias is not None:
            y += self.bias
            
        return y

    @classmethod
    def from_pseudo(cls, pqlinear, group_size, module_load_only=False):
        
        in_features = pqlinear.in_features
        out_features = pqlinear.out_features
        
        if module_load_only:
            return cls(in_features, out_features, bias=None, group_size=group_size, dtype=pqlinear.pqweight.data.dtype)

        bias = None
        assert isinstance(pqlinear, PseudoQuantLinearI4F8_F8) or isinstance(pqlinear, PseudoQuantLinearScaled_I4F8_F8)
        assert in_features % 8 == 0
        
        qweight = pqlinear.pqweight.data.detach().clone().to(torch.int8)
        scales = pqlinear.scales.data.detach().clone()
        
        ## on cuda
        iweight = pack_int4_to_int32(qweight)
        
        if pqlinear.bias is not None:
            bias = pqlinear.bias.data.detach().clone()

        new_module = cls(in_features, out_features, bias=bias, group_size=group_size, dtype=pqlinear.pqweight.data.dtype)
        
        new_module.weight = iweight
        new_module.weight_scale = scales.to(torch.float8_e4m3fn)
        new_module.bias = bias
        
        new_module.matmul_kernel = get_matmul_kernel('I4F8_F8', pqlinear.pqweight.data.dtype)
        new_module.act_kernel = get_act_quantizer_kernel('I4F8_F8', pqlinear.pqweight.data.dtype) 
        new_module.weight_dequant_kernel = get_dequant_weight_kernel('I4F8_F8', pqlinear.pqweight.data.dtype) 

        return new_module
            
            

class LinearScaled_I4F8_F8(BaseLinearModule):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype
        self.matmul_kernel = None
        self.act_kernel = None
        self.weight_dequant_kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(torch.float8_e4m3fn))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
        self.register_buffer("per_tensor_scale_inv", torch.ones(1, dtype=dtype))

    def forward(self, x, restore_scale=True):
        ## TODO cuda, cpu branch
        x_fp8, x_scale_f16 = self.act_kernel(x)  # triton_fp8_bf16_quant

        iweight = triton_unpack_int32_to_int4(self.weight)
        deq_weight = self.weight_dequant_kernel(iweight, self.weight_scale, self.group_size) # triton_dequantize_int8_fp8

        y = self.matmul_kernel(x_fp8, x_scale_f16, deq_weight.T) # fp8_matmul_
        #y = y*x_scale_f16
        if self.bias is not None:
            y += self.bias
            
        if restore_scale:
            y = y*self.per_tensor_scale_inv
            
        return y

    @classmethod
    def from_pseudo(cls, pqlinear, group_size, module_load_only=False):
        
        in_features = pqlinear.in_features
        out_features = pqlinear.out_features
        orig_dtype = pqlinear.pqweight.data.dtype
        if module_load_only:
            return cls(in_features, out_features, bias=None, group_size=group_size, dtype=orig_dtype)

        bias = None
        assert isinstance(pqlinear, PseudoQuantLinearScaled_I4F8_F8)
        assert in_features % 8 == 0
        
        qweight = pqlinear.pqweight.data.detach().clone().to(torch.int8)
        scales = pqlinear.scales.data.detach().clone()
        per_tensor_scale = pqlinear.per_tensor_scale.detach().clone()
        ## on cuda
        iweight = pack_int4_to_int32(qweight)
        
        if pqlinear.bias is not None:
            bias = pqlinear.bias.data.detach().clone()

        new_module = cls(in_features, out_features, bias=bias, group_size=group_size, dtype=orig_dtype)
        
        new_module.weight = iweight
        new_module.weight_scale = scales.to(torch.float8_e4m3fn)
        new_module.bias = bias
        new_module.per_tensor_scale_inv = (1.0/per_tensor_scale.to(torch.float64)).to(orig_dtype)

        new_module.matmul_kernel = get_matmul_kernel('SCALED_I4F8_F8', orig_dtype)
        new_module.act_kernel = get_act_quantizer_kernel('SCALED_I4F8_F8', orig_dtype) 
        new_module.weight_dequant_kernel = get_dequant_weight_kernel('SCALED_I4F8_F8', pqlinear.pqweight.data.dtype) 

        return new_module
            

class LinearI4_F8(BaseLinearModule):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.matmul_kernel = None
        self.act_kernel = None
        self.weight_dequant_kernel = None
        self.weight_quant_kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        ## TODO cuda, cpu 분할
        x_fp8, x_scale_bf16 = self.act_kernel(x)  # triton_fp8_f16_quant

        iweight = triton_unpack_int32_to_int4(self.weight)
        deq_weight = self.weight_dequant_kernel(iweight, self.weight_scale, self.group_size)  # triton_dequantize_int8_bf16

        w_fp8, w_f16_pch_scale = self.weight_quant_kernel(deq_weight)  # triton_f16_fp8_weight_quant_per_ch
        
        y = self.matmul_kernel(x_fp8, x_scale_bf16, w_fp8.T, w_f16_pch_scale) # triton_fp8_matmul_fp8f16xfp8f16_per_tok_per_ch

        #y = (y*x_scale_bf16) * w_f16_pch_scale.T
        if self.bias is not None:
            y += self.bias
            
        return y

    @classmethod
    def from_pseudo(cls, pqlinear, group_size, module_load_only=False):
        
        in_features = pqlinear.in_features
        out_features = pqlinear.out_features
        orig_dtype = pqlinear.pqweight.data.dtype
        if module_load_only:
            return cls(in_features, out_features, bias=None, group_size=group_size, dtype=orig_dtype)

        bias = None
        assert isinstance(pqlinear, PseudoQuantLinearI4_F8)
        assert in_features % 8 == 0
        
        qweight = pqlinear.pqweight.data.detach().clone().to(torch.int8)
        scales = pqlinear.scales.data.detach().clone()
        
        ## on cuda
        iweight = pack_int4_to_int32(qweight)
        
        if pqlinear.bias is not None:
            bias = pqlinear.bias.data.detach().clone()

        new_module = cls(in_features, out_features, bias=bias, group_size=group_size, dtype=orig_dtype)
        
        new_module.weight = iweight
        new_module.weight_scale = scales
        new_module.bias = bias
        
        new_module.matmul_kernel = get_matmul_kernel('I4_F8', orig_dtype)
        new_module.act_kernel = get_act_quantizer_kernel('I4_F8', orig_dtype) 
        new_module.weight_quant_kernel = get_quant_weight_kernel('I4_F8', orig_dtype)
        new_module.weight_dequant_kernel = get_dequant_weight_kernel('I4_F8', orig_dtype)

        return new_module

class LinearI4(BaseLinearModule):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.weight_dequant_kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):

        iweight = triton_unpack_int32_to_int4(self.weight)
        deq_weight = self.weight_dequant_kernel(iweight, self.weight_scale, self.group_size) # triton_dequantize_int8_bf16

        y = x @ deq_weight.T

        if self.bias is not None:
            y += self.bias
            
        return y

    @classmethod
    def from_pseudo(cls, pqlinear, group_size, module_load_only=False):
        
        in_features = pqlinear.in_features
        out_features = pqlinear.out_features
        
        if module_load_only:
            return cls(in_features, out_features, bias=None, group_size=group_size)

        bias = None
        assert isinstance(pqlinear, PseudoQuantLinearI4)
        assert in_features % 8 == 0
        
        qweight = pqlinear.pqweight.data.detach().clone().to(torch.int8)
        scales = pqlinear.scales.data.detach().clone()
        
        ## on cuda
        iweight = pack_int4_to_int32(qweight)
        
        if pqlinear.bias is not None:
            bias = pqlinear.bias.data.detach().clone()

        new_module = cls(in_features, out_features, bias=bias, group_size=group_size)
        
        new_module.weight = iweight
        new_module.weight_scale = scales
        new_module.bias = bias
        
        new_module.weight_dequant_kernel = get_dequant_weight_kernel('I4', pqlinear.pqweight.data.dtype) 

        return new_module
            
            

