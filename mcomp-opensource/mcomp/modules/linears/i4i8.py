import torch
import torch.nn as nn
from torch.autograd import Function
import triton
import triton.language as tl
from .base import BaseLinearModule

from mcomp.utils.module import (
    pack_int4_to_int32,
    triton_unpack_int32_to_int4,
    triton_int8_bf16_quant,
)

from mcomp.utils.module.gemm import (
    int8_matmul_,
)

from mcomp.quant.modules.linears import PseudoQuantLinearI4_I8
from mcomp.kernels.kernel_utils import (
    get_act_quantizer_kernel,
    get_matmul_kernel,
)

class LinearI4_I8(BaseLinearModule):
    
    def __init__(self, in_features, out_features, bias=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, 1).abs().to(dtype))
        self.act_kernel = None
        self.matmul_kernel = None
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        ## TODO cuda, cpu 분할
        x_i8, x_scale_f16 = self.act_kernel(x) # triton_int8_bf16_quant

        iweight = triton_unpack_int32_to_int4(self.weight)

        #y = fp8_matmul_(x_fp8, deq_weight.T)*x_scale_bf16
        y = self.matmul_kernel(x_i8, x_scale_f16, iweight.T, self.weight_scale)
        #y = (x_scale_f16 * y) * self.weight_scale
        if self.bias is not None:
            y += self.bias
            
        return y

    @classmethod
    def from_pseudo(cls, pqlinear, group_size=None, module_load_only=False):
        
        in_features = pqlinear.in_features
        out_features = pqlinear.out_features
        orig_dtype = pqlinear.pqweight.data.dtype
        if module_load_only:
            return cls(in_features, out_features, bias=None)

        bias = None
        assert isinstance(pqlinear, PseudoQuantLinearI4_I8)
        assert in_features % 8 == 0
        
        qweight = pqlinear.pqweight.data.detach().clone().to(torch.int8)
        scales = pqlinear.scales.data.detach().clone()
        
        ## on cuda
        iweight = pack_int4_to_int32(qweight)
        
        if pqlinear.bias is not None:
            bias = pqlinear.bias.data.detach().clone()

        new_module = cls(in_features, out_features, bias=bias)
        
        new_module.weight = iweight
        new_module.weight_scale = scales
        new_module.bias = bias
        
        new_module.act_kernel = get_act_quantizer_kernel('I4_I8', orig_dtype) 
        new_module.matmul_kernel = get_matmul_kernel('I4_I8', orig_dtype)

        return new_module
            
            