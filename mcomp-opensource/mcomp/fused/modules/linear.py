import torch
import torch.nn as nn
from torch.autograd import Function

from mcomp.utils.module import (
    pack_int4_to_int32,
    triton_unpack_int32_to_int4,
    dequantize_int8_fp8,
    triton_fp8_bf16_quant,
    triton_dequantize_int8_fp8,
    triton_act_quant,
    unpack_int32_to_int4,
)

from mcomp.utils.module.gemm import (
    fp8_bf16_matmul_,
    fp8_fp16_matmul_,
    fp8_matmul_
)

from mcomp.fused.layers import (
    INT4FP8Linear,
    INT4FP8LinearMul,
    INT4FP8LinearSiLU,
    INT4FP8LinearSquareDot
)

class FastLinearI4FP8Function(Function):
    
    @staticmethod
    def forward(ctx, x_fp8, x_scale_bf16, weight_i4, weight_scale_fp8, bias_bf16, group_size):

        iweight = triton_unpack_int32_to_int4(weight_i4)
        deq_weight = triton_dequantize_int8_fp8(iweight, weight_scale_fp8, group_size)

        y = fp8_bf16_matmul_(x_fp8, deq_weight.T)*x_scale_bf16
        if bias_bf16 is not None:
            y += bias_bf16

        return y

class FastLinearI4FP8(nn.Module):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16, scale_dtype=torch.float8_e4m3fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(scale_dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x, x_scale):
        if self.kernel:
            y = self.kernel.forward(x, x_scale, x.shape[0], x.shape[1])
        else:
            y = FastLinearI4FP8Function.apply(x, x_scale, self.weight, self.weight_scale, self.bias, self.group_size)
        return y
        
class FastLinearI4FP8Scaled(nn.Module):
    
    def __init__(self, in_features, out_features, bias=None, group_size=128, dtype=torch.bfloat16, scale_dtype=torch.float8_e4m3fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.kernel = None
        
        self.register_buffer("weight", torch.zeros(out_features, int(in_features /8), dtype=torch.int32))
        self.register_buffer("weight_scale", torch.randn(out_features, int(in_features/group_size)).abs().to(scale_dtype))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
        self.register_buffer("per_tensor_scale_inv", torch.ones(1, dtype=dtype))

    def forward(self, x, x_scale):
        if self.kernel:
            y = self.kernel.forward(x, x_scale, x.shape[0], x.shape[1])
        else:
            y = FastLinearI4FP8Function.apply(x, x_scale, self.weight, self.weight_scale, self.bias, self.group_size)
        y = y*self.per_tensor_scale_inv
        return y
        
        