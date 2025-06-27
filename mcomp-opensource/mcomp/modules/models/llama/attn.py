import torch
import torch.nn as nn
from torch.autograd import Function
import triton
import triton.language as tl

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,

)
from transformers.models.llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from ...calib import (
  OnlineCalibratorInfer
)

from mcomp.utils import (
  find_module_by_exactmatching,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from mcomp.utils.config import get_type_matching_dict

from ...linears import (
    LinearScaled_I4F8_F8, 
    LinearI4F8_F8, 
    LinearI4_F8, 
    LinearI4,
    LinearI4_I8
)
from mcomp.quant.modules.linears import (
    PseudoQuantLinearI4F8_F8, 
    PseudoQuantLinearScaled_I4F8_F8, 
    PseudoQuantLinearI4_F8, 
    PseudoQuantLinearI4,
    PseudoQuantLinearI4_I8,
)
from mcomp.quant.modules.post_rope import PostSmoothLayer
from typing import Union, List, Tuple, Dict, Optional, Callable
from mcomp.modules.module_utils import get_linear

class LlamaQKVSupplyInfer(nn.Module): # definitive

    def __init__(self, config: LlamaConfig, layer_idx, layer_quant_config, act_quant_fn, _apply_rotary_pos_emb=None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.layer_quant_config = layer_quant_config
        
        self.kv_quant:bool = layer_quant_config.kv_quant

        self.qk_calib_config = layer_quant_config.qk_calib
        self.qk_calib:bool = True if self.qk_calib_config is not None else False
        
        self.rope_type:str = layer_quant_config.rope.type
        self.scaled_rope:bool = layer_quant_config.rope.scaled
        self.post_rope_smooth:bool = layer_quant_config.rope.post_rope_smooth
        self.top_k:int = layer_quant_config.rope.top_k

        self.k_calib_config = layer_quant_config.rope.k_calib
        self.k_calib:bool = True if self.k_calib_config is not None else False
        self.post_rope_smooth = False if self.rope_type == "pre" else self.post_rope_smooth
        
        self.act_quant_fn = act_quant_fn
        self.attn_ops_config = layer_quant_config.attn_ops

        assert self.rope_type in ["post", "pre"]
        self.k_calib = False if self.rope_type == "post" else self.k_calib

        self.is_post_rope = True if self.rope_type == "post" else False

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.parameterized_rotary_pos_emb = False
        if _apply_rotary_pos_emb is not None:
            self._apply_rotary_pos_emb = _apply_rotary_pos_emb
            self.parameterized_rotary_pos_emb = True
        else:
            self._apply_rotary_pos_emb = apply_rotary_pos_emb

        if self.post_rope_smooth:
            self.post_k_smooth = PostSmoothLayer(self.num_key_value_heads, self.top_k)
            self.post_q_smooth = PostSmoothLayer(self.num_key_value_heads*self.num_key_value_groups, self.top_k)
        
        ## Validation Logic on kv_quant and attn_implementation etc.
        if self.qk_calib:
            from_name = self.layer_quant_config.qk_calib.from_name
            from_name = 'post_rope_query_side_onlinecalib' if from_name is None else from_name
            self.qk_calib_from_name = from_name
            
            to_name = self.layer_quant_config.qk_calib.to_name
            to_name = 'post_rope_key_side_onlinecalib' if to_name is None else to_name
            self.qk_calib_to_name = to_name
            
            setattr(self, from_name, OnlineCalibratorInfer(self.qk_calib_config, self.head_dim, torch.bfloat16) )
            setattr(self, to_name, OnlineCalibratorInfer(self.qk_calib_config, self.head_dim, torch.bfloat16) )
        
        if self.k_calib:
            from_name = self.layer_quant_config.rope.k_calib.from_name
            from_name = 'pre_rope_key_onlinecalib' if from_name is None else from_name
            self.k_calib_from_name = from_name
            
            to_name = self.layer_quant_config.rope.k_calib.to_name
            to_name = 'pre_rope_key_counter_onlinecalib' if to_name is None else to_name
            self.k_calib_to_name = to_name
            
            setattr(self, from_name, OnlineCalibratorInfer(self.k_calib_config, self.head_dim*config.num_key_value_heads, torch.bfloat16) )
            setattr(self, to_name, OnlineCalibratorInfer(self.k_calib_config, self.head_dim*config.num_key_value_heads, torch.bfloat16) )
            
            
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

        if self.kv_quant: 
            assert past_key_value is not None, "When using Quantized KVCache you should provide the Specific quantizing Cache"

        # if isinstance(past_key_value, IFPQuantizedCache): ## TODO baseQuantizedCache
        #     pass

        if self.is_post_rope:
            q = q.view(hidden_shape).transpose(1, 2) # [B H, S, D]
            k = k.view(hidden_shape).transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)
            
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            
            if self.post_rope_smooth:
                q = self.post_q_smooth(q)
                k = self.post_k_smooth(k)
            
            if self.qk_calib:
                q = getattr(self, self.qk_calib_from_name)(q)
                k = getattr(self, self.qk_calib_to_name)(k)

            ### q quantize
            if self.kv_quant:
                if self.attn_implementation in ['fp8fp16_i4fp8', 'fp8']:
                    qq, qqs = self.act_quant_fn(q) # TODO callable factory w.r.t qdtype   # not pseudo
                    q = (qq, qqs)
                else:
                    # no quant on q
                    pass
            
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        else:
            k_orig_shape = k.shape
            if self.k_calib:
                k = getattr(self, self.k_calib_from_name)(k) # [B S hidden_size] -> [B S hidden_size] hidden_size transform

            q = q.view(hidden_shape).transpose(1, 2)
            k = k.view(hidden_shape).transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

            if self.k_calib:
                k = k.transpose(1, 2).reshape(k_orig_shape)
                k = getattr(self, self.k_calib_to_name)(k) # [B S hidden_size] -> [B S hidden_size] hidden_size transform
                k = k.view(hidden_shape).transpose(1,2)
                
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            
            if self.qk_calib:
                q = getattr(self, self.qk_calib_from_name)(q)
                k = getattr(self, self.qk_calib_to_name)(k)

            # q quantize, k quantize
            if self.kv_quant:
                if self.attn_implementation in ['fp8fp16_i4fp8', 'fp8']:
                    qq, qqs = self.act_quant_fn(q) # TODO callable factory w.r.t qdtype   # not pseudo
                    q = (qq, qqs)
                else:
                    # no quant on q
                    pass
                if self.attn_implementation in ['fp8fp16_i4fp8', 'fp8']:
                    
                    #ikey, ikey_scale = past_key_value._group_quantize_for_kv(k, past_key_value.nbits, past_key_value.group_size, scales_dt=past_key_value.scale_dtype)
                    k = past_key_value._group_quantize_for_kv(k, past_key_value.nbits, past_key_value.group_size, scales_dt=past_key_value.scale_dtype)
            
        return q, k, v
        
        


class LlamaAttentionInfer(nn.Module):
    
    def __init__(self, config: LlamaConfig, layer_idx:int, layer_quant_config, group_size, act_quant_fn, _apply_rotary_pos_emb=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.layer_quant_config = layer_quant_config
        self.act_quant_fn = act_quant_fn
        self.group_size = group_size

        self.type_matching_dict = get_type_matching_dict(layer_quant_config)

        o_proj_type = self.type_matching_dict.get('o_proj', None)
        self.o_proj = get_linear(o_proj_type, config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, group_size=group_size)

        q_proj_type = self.type_matching_dict.get('q_proj', None)
        self.q_proj = get_linear(q_proj_type, config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, group_size=group_size)

        k_proj_type = self.type_matching_dict.get('k_proj', None)
        self.k_proj = get_linear(k_proj_type, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, group_size=group_size)
        
        v_proj_type = self.type_matching_dict.get('v_proj', None)
        self.v_proj = get_linear(v_proj_type, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, group_size=group_size)

        self.qkv_supply = LlamaQKVSupplyInfer(config, layer_idx, layer_quant_config, act_quant_fn)

                
    def forward(self, 
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs # Unpack[FlashAttentionKwargs](transformers.processing_utils)
               ):
        
        input_shape = hidden_states.shape[:-1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q,k,v = self.qkv_supply(q, k, v, position_embeddings, past_key_value, cache_position, **kwargs)
        
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation] ## TODO attention_interface setter
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


    @classmethod
    def from_pseudo(cls, pqattn, act_quant_fn, module_load_only=False):
        
        config = pqattn.config
        layer_quant_config = pqattn.layer_quant_config
        layer_idx = pqattn.layer_idx
        group_size = pqattn.group_size

        new_module = cls(config, layer_idx, layer_quant_config, group_size, act_quant_fn) #_apply_rotary_pos_emb
        if module_load_only:
            return new_module

        # per_tensor_scale이 1.0이면 그냥 LinearI4FP8로
        for proj_name in ['o_proj', 'q_proj', 'k_proj', 'v_proj']:
            proj = find_module_by_exactmatching(pqattn, proj_name)
            if isinstance(proj, nn.Linear):
                setattr(new_module, proj_name, proj)
            else:            
            
                if isinstance(proj, PseudoQuantLinearI4F8_F8):
                    setattr(new_module, proj_name, LinearI4F8_F8.from_pseudo(proj, group_size)) 
                # TMP
                #setattr(new_module, proj_name, LinearI4FP8Scaled.from_pseudo(proj, group_size))
                elif isinstance(proj, PseudoQuantLinearI4_I8):
                    setattr(new_module, proj_name, LinearI4_I8.from_pseudo(proj)) 
                elif isinstance(proj, PseudoQuantLinearI4_F8):
                    setattr(new_module, proj_name, LinearI4_F8.from_pseudo(proj, group_size))
                elif isinstance(proj, PseudoQuantLinearI4):
                    setattr(new_module, proj_name, LinearI4.from_pseudo(proj, group_size))
                elif isinstance(proj, PseudoQuantLinearScaled_I4F8_F8):
                    setattr(new_module, proj_name, LinearScaled_I4F8_F8.from_pseudo(proj, group_size)) 
                else:
                    raise NotImplementedError
                # if proj_name in ['q_proj', 'k_proj' ,'v_proj'] and isinstance(proj, PseudoQuantLinearI4FP8Scaled): # v_proj temp
                #     setattr(new_module, proj_name, LinearI4FP8Scaled.from_pseudo(proj, group_size)) 
                # else:
                #     setattr(new_module, proj_name, LinearI4FP8.from_pseudo(proj, group_size))  ## TODO cls-type-dependent

        # qk_calib
        if new_module.qkv_supply.qk_calib and pqattn.qkv_supply.qk_calib:
            
            setattr(new_module.qkv_supply, 
                    new_module.qkv_supply.qk_calib_from_name, 
                    OnlineCalibratorInfer.from_pseudo(getattr(pqattn.qkv_supply,
                                                              pqattn.qkv_supply.qk_calib_from_name))
                    )
            setattr(new_module.qkv_supply, 
                    new_module.qkv_supply.qk_calib_to_name, 
                    OnlineCalibratorInfer.from_pseudo(getattr(pqattn.qkv_supply,
                                                              pqattn.qkv_supply.qk_calib_to_name))
                    )
        # k_calib
        if new_module.qkv_supply.k_calib and pqattn.qkv_supply.k_calib:
            
            setattr(new_module.qkv_supply, 
                    new_module.qkv_supply.q_calib_from_name, 
                    OnlineCalibratorInfer.from_pseudo(getattr(pqattn.qkv_supply,
                                                              pqattn.qkv_supply.k_calib_from_name))
                    )
            setattr(new_module.qkv_supply, 
                    new_module.qkv_supply.k_calib_to_name, 
                    OnlineCalibratorInfer.from_pseudo(getattr(pqattn.qkv_supply,
                                                              pqattn.qkv_supply.k_calib_to_name))
                    )

        # post rope smooth
        if new_module.qkv_supply.post_rope_smooth and pqattn.qkv_supply.post_rope_smooth:
            
            ## Todo Create InferClass
            setattr(new_module.qkv_supply, 
                    "post_k_smooth", 
                    getattr(pqattn.qkv_supply,"post_k_smooth"))
                    
            setattr(new_module.qkv_supply, 
                    "post_q_smooth", 
                    getattr(pqattn.qkv_supply,"post_q_smooth"))
                    

        return new_module