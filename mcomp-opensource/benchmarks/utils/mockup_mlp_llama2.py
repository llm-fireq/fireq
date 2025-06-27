import torch
from torch import nn
from mcomp.fused.modules.linear import FastLinearI4FP8Scaled
from mcomp.fused.modules.models.llama.mlp import GateProj, UpProj, DownProj
from mcomp.utils.module.quant import triton_fp8_bf16_quant
from transformers.activations import ACT2FN
from mcomp.utils import timing_utils
from mcomp.fused.layers import (
    INT4FP8Linear,
    INT4FP8LinearAdd, 
    INT4FP8LinearMul, 
    INT4FP8LinearSiLU, 
    INT4FP8LinearSquareDot,
    INT4BF16Linear,
    INT4BF16LinearAdd,
    INT4BF16LinearMul,
    INT4BF16LinearSiLU,
    INT4BF16LinearSquareDot,
) 

class MLPINT4FP8Llama2(nn.Module):
    def __init__(self, hidden_size, intermediate_size, is_bf16, group_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.group_size = group_size
        self.act_quant_fn = triton_fp8_bf16_quant
        self.act_fn = ACT2FN['silu']
        self.is_bf16 = is_bf16
        
        self.gate_proj_weight = torch.randint(-7, 8, (intermediate_size, hidden_size // 2), dtype=torch.int8).cuda()
        self.gate_proj_weight_scale = torch.randn(intermediate_size, (hidden_size + group_size - 1) // group_size).abs().to(torch.float8_e4m3fn).cuda()
        
        self.up_proj_weight = torch.randint(-7, 8, (intermediate_size, hidden_size // 2), dtype=torch.int8).cuda()
        self.up_proj_weight_scale = torch.randn(intermediate_size, (hidden_size + group_size - 1) // group_size).abs().to(torch.float8_e4m3fn).cuda()
        
        self.down_proj_weight = torch.randint(-7, 8, (hidden_size, intermediate_size // 2), dtype=torch.int8).cuda()
        self.down_proj_weight_scale = torch.randn(hidden_size, (intermediate_size + group_size - 1) // group_size).abs().to(torch.float8_e4m3fn).cuda()

        if self.is_bf16:
            self.gate_proj_weight_scale = self.gate_proj_weight_scale.to(torch.bfloat16)
            self.up_proj_weight_scale = self.up_proj_weight_scale.to(torch.bfloat16)
            self.down_proj_weight_scale = self.down_proj_weight_scale.to(torch.bfloat16)
            self.gate_proj_kernel = INT4BF16LinearSiLU(self.gate_proj_weight,
                                                            self.gate_proj_weight_scale,
                                                            hidden_size, intermediate_size, group_size)
            
            self.up_proj_kernel = INT4BF16LinearMul(self.up_proj_weight,
                                                    self.up_proj_weight_scale,
                                                    hidden_size, intermediate_size, group_size)
            
            self.down_proj_kernel = INT4BF16LinearAdd(self.down_proj_weight,
                                                    self.down_proj_weight_scale,
                                                    hidden_size, intermediate_size, group_size)
        else:
            self.gate_proj_kernel = INT4FP8LinearSiLU(self.gate_proj_weight,
                                                            self.gate_proj_weight_scale,
                                                            hidden_size, intermediate_size, group_size)
            
            self.up_proj_kernel = INT4FP8LinearMul(self.up_proj_weight,
                                                    self.up_proj_weight_scale,
                                                    hidden_size, intermediate_size, group_size)
            
            self.down_proj_kernel = INT4FP8LinearAdd(self.down_proj_weight,
                                                    self.down_proj_weight_scale,
                                                    intermediate_size, hidden_size, group_size)
    
    def forward(self, x, residual):
        if isinstance(self.gate_proj_kernel, INT4BF16LinearSiLU):
            _, qxscale = self.act_quant_fn(x)
            gate_score = self.gate_proj_kernel.forward(x, qxscale, x.shape[0], x.shape[1])
            intermediate = self.up_proj_kernel.forward(x, qxscale, gate_score, x.shape[0], x.shape[1])
            out = self.down_proj_kernel.forward(intermediate, qxscale, residual, x.shape[0], x.shape[1])
        else:
            qx, qxscale = self.act_quant_fn(x)
            gate_score = self.gate_proj_kernel.forward(qx, qxscale, qx.shape[0], qx.shape[1])
            intermediate = self.up_proj_kernel.forward(qx, qxscale, gate_score, qx.shape[0], qx.shape[1])
            qx, qxscale = self.act_quant_fn(intermediate)
            out = self.down_proj_kernel.forward(qx, qxscale, residual, qx.shape[0], qx.shape[1])
        return out