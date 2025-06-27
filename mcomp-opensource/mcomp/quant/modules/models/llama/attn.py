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
from transformers import Cache
from ...calib import OnlineCalibrator
from ...cache import BasePseudoQuantCache
from ...pair import CalibrationPair
from ...post_rope import PostSmoothLayer

from typing import Union, List, Tuple, Dict, Optional, Callable

from mcomp.config import MappingElementConfig 
from mcomp.utils.quant import pseudo_act_quantize_fp8

from mcomp.utils import (
  find_module_by_exactmatching,
)

from mcomp.quant.modules.pseudo_module_utils import get_pseudolinear, get_pseudo_act_quantizer
from mcomp.utils.config import get_qdtype_from_layer_quant_config
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

## attention
"""
 - input [B S hidden_dim]
 - hidden_shape [B S -1 head_dim]

   ex) W_q [4096 * 4096], W_k [4096 * 1024], W_v [4096 * 1024] (head_dim: 128, head_num: 32)
   
   Q -> [B S total_query_dim] -> [B S -1 head_dim]    -1 -> 32    -> [ B 32 S head_dim ]
   K -> [B S total_key_dim] -> [B S -1 head_dim]       -1 -> 8    -> [ B  8 S head_dim ]
   V -> [B S total_value_dim] -> [B S -1 head_dim]     -1 -> 8    -> [ B  8 S head_dim ]

   cos, sin = position_embeddings
   
   
"""

class PseudoLlamaQKVSupply(nn.Module):

    def __init__(self, config, layer_idx, layer_quant_config=None, on_dtype=torch.bfloat16, calib_dtype=torch.bfloat16, _apply_rotary_pos_emb=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_quant_config = layer_quant_config
        self.on_dtype = on_dtype
        self.calib_dtype = calib_dtype

        self.kv_quant:bool = layer_quant_config.kv_quant
        
        self.qk_calib_config = layer_quant_config.qk_calib
        self.qk_calib:bool = True if self.qk_calib_config is not None else False
        
        self.rope_type:str = layer_quant_config.rope.type
        self.scaled_rope:bool = layer_quant_config.rope.scaled
        self.post_rope_smooth:bool = layer_quant_config.rope.post_rope_smooth
        self.top_k:int = layer_quant_config.rope.top_k

        self.k_calib_config = layer_quant_config.rope.k_calib
        self.k_calib:bool = True if self.k_calib_config is not None else False

        assert self.rope_type in ["post", "pre"]
        self.k_calib = False if self.rope_type == "post" else self.k_calib
        self.post_rope_smooth = False if self.rope_type == "pre" else self.post_rope_smooth

        self.is_post_rope = True if self.rope_type == "post" else False

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads

        if _apply_rotary_pos_emb is not None:
            self._apply_rotary_pos_emb = _apply_rotary_pos_emb
        else:
            self._apply_rotary_pos_emb = apply_rotary_pos_emb
        
        if self.post_rope_smooth:
            self.post_k_smooth = PostSmoothLayer(self.num_key_value_heads, self.top_k)
            self.post_q_smooth = PostSmoothLayer(self.num_key_value_heads*self.num_key_value_groups, self.top_k)
        
        if self.qk_calib:
            from_name = self.layer_quant_config.qk_calib.from_name
            from_name = 'post_rope_query_side_onlinecalib' if from_name is None else from_name
            self.qk_calib_from_name = from_name
            
            to_name = self.layer_quant_config.qk_calib.to_name
            to_name = 'post_rope_key_side_onlinecalib' if to_name is None else to_name
            self.qk_calib_to_name = to_name
            
            setattr(self, from_name, OnlineCalibrator(self.head_dim, on_dtype=on_dtype, calib_dtype=calib_dtype) )
            setattr(self, to_name, OnlineCalibrator(self.head_dim, on_dtype=on_dtype, calib_dtype=calib_dtype) )
        
        if self.k_calib:
            from_name = self.layer_quant_config.rope.k_calib.from_name
            from_name = 'pre_rope_key_onlinecalib' if from_name is None else from_name
            self.k_calib_from_name = from_name
            
            to_name = self.layer_quant_config.rope.k_calib.to_name
            to_name = 'pre_rope_key_counter_onlinecalib' if to_name is None else to_name
            self.k_calib_to_name = to_name
            
            setattr(self, from_name, OnlineCalibrator(self.head_dim*config.num_key_value_heads, on_dtype=on_dtype, calib_dtype=calib_dtype) )
            setattr(self, to_name, OnlineCalibrator(self.head_dim*config.num_key_value_heads, on_dtype=on_dtype, calib_dtype=calib_dtype) )

        q_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'q_proj')
        self.q_act_fcn = get_pseudo_act_quantizer(q_proj_qdtype.types) # TODO change from q_qdtype
        
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
        
        input_shape = q.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        is_pseudocache = False
        if isinstance(past_key_value, BasePseudoQuantCache):
            is_pseudocache = True

        if self.kv_quant:
            assert is_pseudocache

        if self.is_post_rope:
            q_orig_shape = q.shape
            k_orig_shape = k.shape
            
            q = q.view(hidden_shape).transpose(1, 2)
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
            if is_pseudocache:
                qq, qqs = self.q_act_fcn(q, on_dtype=self.on_dtype, out_act_dtype=self.on_dtype, out_scales_dtype=self.on_dtype)
                q = qq*qqs
            
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        else:
            k_orig_shape = k.shape
            if self.k_calib:
                k = getattr(self, self.k_calib_from_name)(k)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

            if self.k_calib:
                k = k.transpose(1, 2).reshape(k_orig_shape)
                k = getattr(self, self.k_calib_to_name)(k)
                k = k.view(hidden_shape).transpose(1,2)

            q = q.view(hidden_shape).transpose(1, 2)
            k = k.view(hidden_shape).transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)
            
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            
            if self.qk_calib:
                q = getattr(self, self.qk_calib_from_name)(q)
                k = getattr(self, self.qk_calib_to_name)(k)

            # q quantize, k quantize
            if is_pseudocache:
                qq, qqs = self.q_act_fcn(q, on_dtype=self.on_dtype, out_act_dtype=self.on_dtype, out_scales_dtype=self.on_dtype)
                q = qq*qqs

                ikey, ikey_scale = past_key_value._group_quantize_for_kv(k, past_key_value.nbits, past_key_value.group_size, scales_dt=past_key_value.scale_dtype)
                k = past_key_value._group_dequantize(ikey,
                                                     ikey_scale,
                                                     on_dtype=self.on_dtype,
                                                     out_dtype=self.on_dtype)
            
        return q, k, v

    
class PseudoLlamaAttention(nn.Module):
    def __init__(self, attn: LlamaAttention, config: LlamaConfig, layer_quant_config, on_dtype, calib_dtype, dynamic_act_quant:bool=True, _apply_rotary_pos_emb=None):
        super().__init__()
        self.config = config
        ## change model_config attn_implementation to eager (if it is one of others)
        self.layer_quant_config = layer_quant_config
        self.layer_idx = attn.layer_idx
        self.head_dim = attn.head_dim
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal
        
        self.group_size = max(self.head_dim, 128) 
            
        q_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'q_proj')
        if q_proj_qdtype is None or q_proj_qdtype.types.lower() == 'pristine':
            self.q_proj = attn.q_proj
        else:
            self.q_proj = get_pseudolinear(q_proj_qdtype.types.lower(), attn.q_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)


        k_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'k_proj')
        if k_proj_qdtype is None or k_proj_qdtype.types.lower() == 'pristine':
            self.k_proj = attn.k_proj
        else:
            self.k_proj = get_pseudolinear(k_proj_qdtype.types.lower(), attn.k_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
            

        v_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'v_proj')
        if v_proj_qdtype is None or v_proj_qdtype.types.lower() == 'pristine':
            self.v_proj = attn.v_proj
        else:
            self.v_proj = get_pseudolinear(v_proj_qdtype.types.lower(), attn.v_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
            

        o_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'o_proj')
        if o_proj_qdtype is None or o_proj_qdtype.types == 'pristine':
            self.o_proj = attn.o_proj
        else:
            self.o_proj = get_pseudolinear(o_proj_qdtype.types.lower(), attn.o_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
            
        
        self.qkv_supply = PseudoLlamaQKVSupply(config, self.layer_idx, layer_quant_config, on_dtype, calib_dtype, _apply_rotary_pos_emb)
                
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
        
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation] ## TODO
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

    def get_calib_pair(self)-> list:
        ret_list = []
        if self.qkv_supply.qk_calib:
            
            from_name = self.layer_quant_config.qk_calib.from_name
            from_name = 'post_rope_query_side_onlinecalib' if from_name is None else from_name
            
            to_name = self.layer_quant_config.qk_calib.to_name
            to_name = 'post_rope_key_side_onlinecalib' if to_name is None else to_name
            
            me = MappingElementConfig(index=99, 
                                     block='attn',
                                     alias='qk_calib',
                                     names=[from_name],
                                     orientation=0,
                                     matching_names=[to_name],
                                     matching_orientation=0,
                                    )
            
            ret_list.append(CalibrationPair(self, me))
            
        if self.qkv_supply.k_calib:
            
            from_name = self.layer_quant_config.rope.k_calib.from_name
            from_name = 'pre_rope_key_onlinecalib' if from_name is None else from_name
            
            to_name = self.layer_quant_config.rope.k_calib.to_name
            to_name = 'pre_rope_key_counter_onlinecalib' if to_name is None else to_name
            
            me = MappingElementConfig(index=99, 
                                     block='attn',
                                     alias='k_calib',
                                     names=[from_name],
                                     orientation=0,
                                     matching_names=[to_name],
                                     matching_orientation=0,
                                    )
            
            ret_list.append(CalibrationPair(self, me))
        
        return ret_list



            





            
        
            