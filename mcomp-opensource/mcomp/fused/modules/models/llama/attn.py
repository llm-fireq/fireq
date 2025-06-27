import torch
import torch.nn as nn
from torch.autograd import Function

from typing import Union, List, Tuple, Dict, Optional, Callable
from transformers.models.llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from mcomp.utils.module.pack import unpack_int32_to_int4, triton_unpack_int32_to_int4
from mcomp.utils.module.quant import dequantize_int8_fp8, triton_dequantize_int8_fp8
from mcomp.utils.module.gemm import fp8_bf16_matmul_
from mcomp.fused.modules.linear import FastLinearI4FP8

from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
)

from mcomp.fused.layers import FP8FP8FmhaFwd
from mcomp.utils import timing_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class FastPostSmoothLayerFunction(Function):
    
    @staticmethod
    def forward(ctx, x):
        pass

class FastPostSmoothLayer(nn.Module):

    def __init__(self, hidden_dim, top_k, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer('smooth',torch.ones(2*hidden_dim*top_k).to(dtype))
        self.register_buffer('top_k_indices',torch.ones(2*hidden_dim*top_k).to(torch.int32))
        self.top_k = top_k
        self.dtype = dtype

    def forward(self, x):
        # B,H,S,D = x.shape
        # x=x.transpose(1,2).reshape(B,S,-1)
        
        # x[:,:,self.top_k_indices] = x[:,:,self.top_k_indices]*self.smooth
        # x = x.reshape(B, S, H, D).transpose(1,2)        
        
        return FastPostSmoothLayerFunction.apply(x)
    

class FastLlamaQKVSupply(nn.Module):
    
    def __init__(self, config: LlamaConfig, layer_idx, layer_quant_config, act_quant_fn, _apply_rotary_pos_emb=None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.layer_quant_config = layer_quant_config
        
        self.act_quant_fn = act_quant_fn
        self.attn_ops_config = layer_quant_config.attn_ops

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        self.post_k_smooth = FastPostSmoothLayer(self.num_key_value_heads, self.top_k)
        self.post_q_smooth = FastPostSmoothLayer(self.num_key_value_heads*self.num_key_value_groups, self.top_k)

        # self.attn_ops_config
        self.attn_implementation = self.attn_ops_config.attn_implementation if self.attn_ops_config is not None else 'bf16'
        self.q_act_fcn = act_quant_fn # TODO change from q_qdtype

    def forward(self, 
                q, 
                k, 
                v, 
                position_embeddings,
                past_key_value,
                cache_position,
                **kwargs
               ):
        
        cos, sin = position_embeddings
        
        input_shape = q.shape[:-1] # [B S]  # q -> [B S hidden_size]
        hidden_shape = (*input_shape, -1, self.head_dim) # [B S -1, D]
            
        q = q.view(hidden_shape) # [B H, S, D] # no transpose
        k = k.view(hidden_shape)
        v = v.view(hidden_shape)
        
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin) # custom apply_rotary
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        return self.act_quant_fn(q), k, v
        

class AttentionImplementation(Function):
    
    @staticmethod
    def forward(ctx,
                attention_interface,
                module,
                q,
                k,
                v,
                attention_mask,
                scaling,
                dropout,
                kwargs
                ):
        
        attn_output, attn_weights = attention_interface(
            module,
            q,
            k,
            v,
            attention_mask,
            scaling,
            dropout,
            **kwargs
        )
        return attn_output, attn_weights

class OutProjFunction(Function):

    @staticmethod
    def forward(ctx, x_fp8, x_scale_bf16, weight_i4, weight_scale_fp8, residual, bias_bf16, group_size):

        iweight = unpack_int32_to_int4(weight_i4)
        deq_weight = dequantize_int8_fp8(iweight, weight_scale_fp8, group_size)

        y = fp8_bf16_matmul_(x_fp8, deq_weight.T)*x_scale_bf16 + residual
        if bias_bf16 is not None:
            y += bias_bf16

        return y

class OutProj(nn.Module):

    def __init__(self, linear, group_size):
        super().__init__()
        self.linear = linear
        self.group_size = group_size

    def forward(self, qx, qxscale, residual):
        if self.linear.kernel:
            output = self.linear.kernel.forward(qx, qxscale, residual, qx.shape[0], qx.shape[1])
        else:
            output = OutProjFunction.apply(qx, qxscale, self.linear.weight, self.linear.weight_scale, residual, self.linear.bias, self.group_size)
        return output


class FusedLlamaAttentionI4FP8withLN(nn.Module):
    # def __init__(self, config, layer_quant_config, act_quant_fn, group_size):
    def __init__(self, config: LlamaConfig, layer_idx:int, ln, act_quant_fn, qkv_supply, group_size):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.act_quant_fn = act_quant_fn
        self.attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        self.group_size = group_size

        self.input_layernorm = ln
        
        o_proj_linear = FastLinearI4FP8(config.num_attention_heads * self.head_dim, config.hidden_size, group_size=self.group_size)
        self.o_proj = OutProj(o_proj_linear, self.group_size)

        self.q_proj = FastLinearI4FP8(config.hidden_size, config.num_attention_heads * self.head_dim, group_size=self.group_size)
        self.k_proj = FastLinearI4FP8(config.hidden_size, config.num_key_value_heads * self.head_dim, group_size=self.group_size)
        self.v_proj = FastLinearI4FP8(config.hidden_size, config.num_key_value_heads * self.head_dim, group_size=self.group_size)
        
        self.qkv_supply = qkv_supply

        self.attn_kernel = FP8FP8FmhaFwd(self.num_key_value_groups, config.num_key_value_heads, self.head_dim)
                
    def forward(self, 
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs # Unpack[FlashAttentionKwargs](transformers.processing_utils)
               ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        input_shape = hidden_states.shape[:-1]
        
        qx, qx_scale = self.act_quant_fn(hidden_states)
        
        q = self.q_proj(qx, qx_scale)
        k = self.k_proj(qx, qx_scale)
        v = self.v_proj(qx, qx_scale)
        
        q,k,v = self.qkv_supply(q, k, v, position_embeddings, past_key_value, cache_position, **kwargs)
        
        dropout=0.0 if not self.training else self.attention_dropout
        scaling=self.scaling
        
        if self.attn_kernel:
            attn_output = self.attn_kernel.forward(q, k, v, qx.shape[0], qx.shape[1])
        else:
            attn_output, attn_weights = AttentionImplementation.apply(
                self,
                q,
                k,
                v,
                attention_mask,
                scaling,
                dropout,
                kwargs, ## **kwargs
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        qx, qx_scale = self.act_quant_fn(attn_output)
        torch.cuda.synchronize()
        attn_output = self.o_proj(qx, qx_scale, residual)

        if kwargs['output_attentions']:
            return attn_output, attn_weights
        else:
            return attn_output, None
    
