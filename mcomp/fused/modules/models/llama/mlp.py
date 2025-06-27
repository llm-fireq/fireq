import torch
import torch.nn as nn
from torch.autograd import Function

from transformers.models.llama import LlamaConfig
from transformers.activations import ACT2FN

from typing import Union, List, Tuple, Dict, Optional

from mcomp.utils.module.pack import unpack_int32_to_int4, triton_unpack_int32_to_int4
from mcomp.utils.module.quant import dequantize_int8_fp8, triton_dequantize_int8_fp8
from mcomp.utils.module.gemm import fp8_bf16_matmul_
from mcomp.fused.modules.linear import FastLinearI4FP8

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
)

from mcomp.fused.layers import (
    INT4FP8Linear,
    INT4FP8LinearAdd, 
    INT4FP8LinearMul, 
    INT4FP8LinearSiLU, 
    INT4FP8LinearSquareDot,
    INT4BF16Linear,
    INT4BF16LinearAdd,
    INT4BF16LinearSiLU,
    INT4BF16LinearMul,
) 

import time

class DownProjFunction(Function):

    @staticmethod
    def forward(ctx, x_fp8, x_scale_bf16, weight_i4, weight_scale_fp8, residual, bias_bf16, group_size):

        iweight = triton_unpack_int32_to_int4(weight_i4)
        deq_weight = triton_dequantize_int8_fp8(iweight, weight_scale_fp8, group_size)

        y = fp8_bf16_matmul_(x_fp8, deq_weight.T)*x_scale_bf16 + residual
        if bias_bf16 is not None:
            y += bias_bf16

        return y

class DownProj(nn.Module):

    def __init__(self, linear, group_size):
        super().__init__()
        self.linear = linear
        self.group_size = group_size

    def forward(self, qx, qxscale, residual):
        if self.linear.kernel:
            # TODO: Need to check residual dimension of INT4FP8LinearAdd kernel foward function
            output = self.linear.kernel.forward(qx, qxscale, residual, qx.shape[0], qx.shape[1])
        else:
            output = DownProjFunction.apply(qx, qxscale, self.linear.weight, self.linear.weight_scale, residual, self.linear.bias, self.group_size)
        return output


class UpProjFunction(Function):

    @staticmethod
    def forward(ctx, linear, x_fp8, x_scale_bf16, weight_i4, weight_scale_fp8, gate_score, bias_bf16, group_size, batch_size, seq_len):
        
        iweight = triton_unpack_int32_to_int4(weight_i4)
        deq_weight = triton_dequantize_int8_fp8(iweight, weight_scale_fp8, group_size)

        y = fp8_bf16_matmul_(x_fp8, deq_weight.T)*x_scale_bf16
        if bias_bf16 is not None:
            y += bias_bf16

        return y*gate_score

class UpProj(nn.Module):

    def __init__(self, linear, group_size):
        super().__init__()
        self.linear = linear
        self.group_size = group_size

    def forward(self, qx, qxscale, gate_score):
        # If our CUTLASS kernel is defined
        if self.linear.kernel:
            # x, x_scale, batch_size, seq_len
            output = self.linear.kernel.forward(qx, qxscale, gate_score, qx.shape[0], qx.shape[1])
        else:
            output = UpProjFunction.apply(self.linear, qx, qxscale, self.linear.weight, self.linear.weight_scale, gate_score, self.linear.bias, self.group_size)
        return output

class GateProjFunction(Function):

    @staticmethod
    def forward(ctx, x_fp8, x_scale_bf16, weight_i4, weight_scale_fp8, act_fn, bias_bf16, group_size):
        iweight = triton_unpack_int32_to_int4(weight_i4)
        deq_weight = triton_dequantize_int8_fp8(iweight, weight_scale_fp8, group_size)

        y = fp8_bf16_matmul_(x_fp8, deq_weight.T)*x_scale_bf16
        if bias_bf16 is not None:
            y += bias_bf16

        return act_fn(y)

class GateProj(nn.Module):

    def __init__(self, linear, act_fn, group_size):
        super().__init__()
        self.linear = linear
        self.act_fn = act_fn
        self.group_size = group_size

    def forward(self, qx, qxscale):
        if self.linear.kernel:
            output = self.linear.kernel.forward(qx, qxscale, qx.shape[0], qx.shape[1])
        else:
            output = GateProjFunction.apply(qx, qxscale, self.linear.weight, self.linear.weight_scale, self.act_fn, self.linear.bias, self.group_size)
        return output


class FusedLlamaMLPI4FP8withLN(nn.Module):
    def __init__(self, config: LlamaConfig, ln, act_quant_fn, group_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.group_size = group_size
        self.act_quant_fn = act_quant_fn

        self.act_fn = ACT2FN[config.hidden_act]
        self.post_attention_layernorm = ln

        gate_linear = FastLinearI4FP8(self.hidden_size, self.intermediate_size, group_size=self.group_size)
        self.gate_proj = GateProj(gate_linear, self.act_fn, self.group_size)
        
        up_linear = FastLinearI4FP8(self.hidden_size, self.intermediate_size, group_size=self.group_size)
        self.up_proj = UpProj(up_linear, self.group_size)
        
        down_linear = FastLinearI4FP8(self.intermediate_size, self.hidden_size, group_size=self.group_size)
        self.down_proj = DownProj(down_linear, self.group_size)


    def forward(self, x, residual):
        
        #residual = x
        #x = self.post_attention_layernorm(x)
        
        if isinstance(self.gate_proj.linear.kernel, INT4BF16LinearSiLU):
            _, qxscale = self.act_quant_fn(x)
            gate_score = self.gate_proj(x, qxscale)
            intermediate = self.up_proj(x, qxscale, gate_score)
            out = self.down_proj(intermediate, qxscale, residual)
        else:
            qx, qxscale = self.act_quant_fn(x)
            gate_score = self.gate_proj(qx, qxscale)
            intermediate = self.up_proj(qx, qxscale, gate_score)
            qx, qxscale = self.act_quant_fn(intermediate)
            out = self.down_proj(qx, qxscale, residual)
        return out