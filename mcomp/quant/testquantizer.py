import torch
import gc
from .modules.models.llama import PseudoLlamaQKVSupply, PseudoLlamaAttention, PseudoLlamaMLP
from .modules.cache import PseudoIFPQuantDynamicCache

from mcomp.utils import (
  find_module_by_exactmatching,
  get_parent_by_exactmatching
)
from mcomp.utils.config import (
  get_layer_type_map,
)

from mcomp.utils.pair import (
  find_by_allowed_calib,
  find_by_adjacent_names,
  find_by_from_or_to_name,
  find_by_to_name,
  find_by_from_name,
)

from mcomp.utils.quant import pseudo_act_quantize_fp8

from .base import BaseQuantizer
from typing import Tuple, List, Dict
from mcomp.utils.calib import hadamard
from copy import deepcopy
import numpy as np
import random
import tqdm

from mcomp.modules.models.llama import (
  LlamaAttentionInfer, 
  LlamaMLPInfer, 
)


class Llama3TestQuantizer(BaseQuantizer):

    def __init__(self, model, tokenizer, config, device):
        super().__init__(model, tokenizer, config, device)

        ### TODO Change to Dynamic Assigning or Quantizer Segregation
        self.mlp_cls = PseudoLlamaMLP
        self.attn_cls = PseudoLlamaAttention
        self.use_normalization_unified_layer = False
        self.mlp_inf_cls = LlamaMLPInfer
        self.attn_inf_cls = LlamaAttentionInfer
        
        setattr(model.config, '_attn_implementation_internal', 'sdpa')

    def get_attn_kwargs(self, x):
        
        # cache_position = torch.arange(
        #                 0, 0 + x.shape[1], device=x.device
        #             )
        # position_ids = cache_position.unsqueeze(0)
        # causal_mask = model.model._update_causal_mask(
        #             None, x, cache_position, None, None
        #         )
        # position_embeddings = model.model.rotary_emb(x, position_ids)
        
        attn_kwargs = self.activation_getter._inp.attn_kwargs
        x_device = x.device
        causal_mask = self.model.model._update_causal_mask(
            None, x, attn_kwargs['cache_position'].to(x_device), None, None
        )
        return {
            "position_embeddings":attn_kwargs['position_embeddings'],
            "attention_mask": causal_mask,
        }

    @torch.no_grad()
    def get_smooth_with_gamma(self, w, gamma):
        assert isinstance(w, torch.Tensor)
        # x is assumed in the shape of [B, S, D]
        hidden_dim = w.shape[-1]
        # smooth = 1 / gamma
        smooth = w.reshape(-1, hidden_dim).amax(dim=0) / gamma
        # smooth = x.reshape(-1, hidden_dim).abs().mean(dim=0)
        return smooth
    
    @torch.no_grad()
    def get_smooth_with_alpha(self, x, alpha, weight_scale=None):
        assert isinstance(x, torch.Tensor)
        # x is assumed in the shape of [B, S, D]
        hidden_dim = x.shape[-1]
        x_dtype = x.dtype
        x = x.to(self.on_device)
        x_scale = x.reshape(-1, hidden_dim).to(torch.float).abs().mean(dim=0)
        
        if weight_scale is None:
            smooth = x_scale.pow(alpha)
        else:
            weight_scale = weight_scale.to(torch.float).to(self.on_device)
            smooth = x_scale.pow(alpha)/weight_scale.pow(1-alpha)
        smooth = smooth.to(x_dtype)
        
        return smooth

    #region quantize
    @torch.no_grad()
    def quantize(self, qts, pairs, activations:List[torch.Tensor], layer_data:List[Dict[str, torch.Tensor]]):
        
        ## Block-wise manner for now
        
        #region # 1. Rotation into all allowed.
        
        ### For attn, v_proj - o_proj
        if self.config.quant.vo_rotate or self.config.quant.mlp_intermediate_rotate or self.config.quant.qk_calib_rotate:
            ### prepare rotation matrix            
            if self.calib_rotation is not None:
                rotation = self.calib_rotation
                rotation_scale = self.calib_rotation_scale
                rotation = rotation / rotation_scale
                block_size = self.calib_rotation_block_size if self.calib_rotation_block_size is not None else None
            else:
                block_size = self.model_config.head_dim
                assert (block_size & (block_size-1)) == 0 and block_size > 0, "rotation block must be power of 2"
                
                rotation = hadamard(block_size)
                rotation_scale = np.sqrt(block_size).item()
                
                self.calib_rotation = rotation
                self.calib_rotation_block_size = block_size
                self.calib_rotation_scale = rotation_scale
                
                rotation = rotation / rotation_scale
        
        
        if self.config.quant.vo_rotate:
            print("Add Rotation between V-O")
            rotation_allowed_pairs = find_by_allowed_calib(pairs, ['rotated'])

            v_o_pair = find_by_from_or_to_name(rotation_allowed_pairs, 'o_proj')

            for _p in v_o_pair:
                _p.set_rotation(rotation, block_size=block_size)
        
        if self.config.quant.qk_calib_rotate:
            print("Add qk-calib rotate")
            
        
        #endregion
        
        self.construct_linear(qts)

        #region Scaled Linear
        if self.config.quant.chwise_scale:
            print("Applying Channel-wise Smoothing on Linears")
            gamma = self.config.quant.gamma
          
            ## activation mean accumulation for ln smooth
            ret_dict = self.get_batched_mean_act(['q_proj', 'up_proj'])
            
            smoothed_allowed_pairs = find_by_allowed_calib(pairs, ['smoothed'])
           
            for pair_name, proj_name in [('input_layernorm','q_proj'), ('post_attention_layernorm','up_proj')]:
                pair = find_by_from_or_to_name(smoothed_allowed_pairs, pair_name)

                block_idx = self.cur_index // 2
                proj = find_module_by_exactmatching(self.model.model.layers[block_idx], proj_name)
                weight = proj.weight.data
                
                initial_scale = ret_dict[proj_name]
                self.apply_chwise_scaling_factor(qts, pair[0], weight, initial_scale, gamma)


        if self.config.quant.awq_smoothing:
            print("Applying AWQ-based Smoothing on Linears")
            ## activation mean accumulation for ln smooth
            ret_dict = self.get_batched_mean_act(['q_proj', 'up_proj', 'o_proj', 'down_proj'])
            
            smoothed_allowed_pairs = find_by_allowed_calib(pairs, ['smoothed'])

            ## lns에만 smoothed에 붙이면서 확인
            attn_ln_pair = find_by_from_or_to_name(smoothed_allowed_pairs, 'input_layernorm')
            mlp_ln_pair = find_by_from_or_to_name(smoothed_allowed_pairs, 'post_attention_layernorm')
            attn_v_o_pair = find_by_from_or_to_name(smoothed_allowed_pairs, 'o_proj')
            mlp_up_down_pair = find_by_from_or_to_name(smoothed_allowed_pairs, 'down_proj')
            
            ## Add weight scale
            qkv_weight_scale = self.get_weight_scale(['q_proj', 'k_proj', 'v_proj'])
            up_gate_weight_scale = self.get_weight_scale(['up_proj', 'gate_proj'])
            down_weight_scale = self.get_weight_scale(['down_proj'])
            
            
            self.apply_awq_scaling_factor(qts, 
                                          attn_ln_pair[0], 
                                          ret_dict['q_proj'], 
                                          iter([i/10 for i in range(-1, 11, 1)]), 
                                          weight_scale=qkv_weight_scale)
            
            self.apply_awq_scaling_factor(qts, 
                                          mlp_ln_pair[0], 
                                          ret_dict['up_proj'], 
                                          iter([i/10 for i in range(-1, 11, 1)]),
                                          weight_scale=up_gate_weight_scale)
            
            #self.apply_awq_scaling_factor(qts, attn_v_o_pair[0], ret_dict['o_proj'], iter([i/10 for i in range(0, 10, 3)]) )
            
            self.apply_awq_scaling_factor(qts, 
                                          mlp_up_down_pair[0], 
                                          ret_dict['down_proj'], 
                                          iter([i/10 for i in range(-1, 11, 1)]),
                                          weight_scale=down_weight_scale)
                        
                        
        #endregion Scaled Linear   
                                  
        
        #region Pre, Post-ROPE Scales
        
        qkv_supply = find_module_by_exactmatching(qts, 'qkv_supply')
        rope_config = qkv_supply.layer_quant_config.rope
        
        alpha = rope_config.alpha
        beta = rope_config.beta
        top_k = rope_config.top_k
        is_pre_rope_smooth = rope_config.pre_rope_smooth
        is_post_rope_smooth = rope_config.post_rope_smooth
        
        if is_pre_rope_smooth or is_post_rope_smooth:
            q_k_pre_scales, q_k_post_values_and_indices = self.get_batched_rope_aware_scales(alpha, beta, top_k, is_pre_rope_scale_applied=is_pre_rope_smooth)
        
        if is_pre_rope_smooth:
            print("Applying pre-rope normalization")
            self.apply_pre_rope_norm(qts, q_k_pre_scales)
            
        if is_post_rope_smooth:
            print("Applying post-rope smoothing")
            self.apply_post_rope_smooth(qts, *q_k_post_values_and_indices)
        
        ## using pair_utils
        """
        find_by_allowed_calib,
        find_by_adjacent_names,
        find_by_from_or_to_name,
        find_by_to_name,
        find_by_from_name,
        """
    #endregion quantize

    def provide_interested_name_list(self)-> Tuple[List[str], List[Tuple[str, str]]]:
        name_list = ['q_proj', 'o_proj', 'up_proj', 'down_proj', 'v_proj', 'gate_proj'] ## same activation treatment for weight clipping
        return name_list, [('q_proj', 'q_proj_out'), ('k_proj', 'k_proj_out'),]
        
    
    def get_batched_mean_act(self, keys:List[str]):
        ret_dict = {}
        for key in keys:
            ret_dict[key] = None
        
        total_batch_size = 0
        for _acts, _ldatas in self.get_gen_for_batched_data(batch_size=4):
            
            for key in keys:
                acts = _ldatas[-1].get(key, None)
                assert acts is not None
                act_dim = acts.shape[-1]
                acts = acts.view(-1, act_dim).abs()
                total_batch_size += acts.shape[0]

                if ret_dict[key] is None:
                    ret_dict[key] = acts.to(torch.float32).sum(dim=0)
                else:
                    ret_dict[key] = torch.vstack([ret_dict[key],acts.to(torch.float32).sum(dim=0)]).sum(dim=0)
                
        assert total_batch_size > 0
        for key in keys:
            ret_dict[key] = ret_dict[key]/total_batch_size
                
        return ret_dict
    
    def get_weight_scale(self, proj_names:list):
        if not proj_names: return
        block_idx = self.cur_index // 2
        decoder_layer = self.model.model.layers[block_idx]
        
        total_weight = None
        for proj_name in proj_names:
            module = find_module_by_exactmatching(decoder_layer, proj_name)
            
            if total_weight is None:
                total_weight = module.weight.data.clone()
            else:
                weight = module.weight.data.clone()
                assert weight.shape[-1] == total_weight.shape[-1]
                
                total_weight = torch.vstack([total_weight, weight])
        
        weight_scale = total_weight.abs().mean(dim=0)
        
        return weight_scale

    def apply_chwise_scaling_factor(self, qts, the_pair, weight, initial_scale, gamma):

        assert isinstance(weight, torch.Tensor)
        # x is assumed in the shape of [B, S, D]
        hidden_dim = weight.shape[-1]
        # smooth = 1 / gamma
        w_ch_absmax = weight.reshape(-1, hidden_dim).amax(dim=0)
        # smooth = x.reshape(-1, hidden_dim).abs().mean(dim=0)
        
        scale = (torch.finfo(torch.float8_e4m3fn).max / w_ch_absmax.to(float))
        
        print(f"Gamma: {gamma}, scales: {scale}")
        the_pair.set_smooth(scale)


    def apply_awq_scaling_factor(self, qts, the_pair, initial_scale, ratio_iter, weight_scale=None):
        best_loss = None
        best_ratio = None
        
        print(f"init_scale max: {initial_scale.max()}, init_scale min: {initial_scale.min()}, init_scale mean: {initial_scale.mean()}, ")
        
        for ratio in ratio_iter:
            
            scale = self.get_smooth_with_alpha(initial_scale, ratio, weight_scale)
            
            the_pair.set_smooth(scale)

            self.construct_linear(qts)
            
            with torch.no_grad():
                error = 0
                for _acts, _ldatas in self.get_gen_for_batched_data(batch_size=4):
                    rx = _acts[0]
                    ry = _acts[-1]
                    attn_kwargs = self.get_attn_kwargs(rx)
                    
                    past_key_value = None
                    if self.kvcache_config:
                        past_key_value = PseudoIFPQuantDynamicCache(self.kvcache_config)
                    
                    ty = qts(rx, past_key_value=past_key_value, **attn_kwargs)
                    error = error + (ty - ry).pow(2).sum()
    
            if best_loss is None:
                best_loss = error
                best_ratio = ratio
            else:
                if best_loss > error:
                    best_loss = error
                    best_ratio = ratio

        # TODO LOG        
        print("best smooth loss&ratio")
        print(best_loss)
        print(best_ratio)
        
        best_scale = self.get_smooth_with_alpha(initial_scale, best_ratio, weight_scale)
        
        print(f"best_scale max: {best_scale.max()}, best_scale min: {best_scale.min()}, best_scale mean: {best_scale.mean()}, best_ratio: {best_ratio}")
        
        the_pair.set_smooth(best_scale)
        
        best_loss = None
        best_ratio = None
        
    def get_batched_rope_aware_scales(self, alpha, beta, top_k, is_pre_rope_scale_applied=True):
        
        ag = self.activation_getter
        initial_kv_data_flag = ag.kv_data
        ag.kv_data = True
        
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        key_head_size = self.model_config.num_key_value_heads
        num_key_value_group = self.model_config.num_attention_heads // self.model_config.num_key_value_heads
            
        max_post_keys = torch.zeros([1,key_head_size,head_dim])
        max_pre_keys = torch.zeros([1,key_head_size,head_dim//2])    
        
        
        for _acts, _ldatas in self.get_gen_for_batched_data(batch_size=4):
            
            # pre_scale
            k_out = _ldatas[0]['k_proj_out']
            D = k_out.shape[-1]
            keys = k_out.reshape(-1, D)
            key_abs = keys.abs()
            k_tmp = key_abs.reshape(key_abs.shape[0],-1, 2, head_dim//2)
            k_tmp_scales = (k_tmp[:,:,0,:].pow(2) + k_tmp[:,:,1,:].pow(2)).sqrt()
            k_scales = k_tmp_scales.amax(dim=0, keepdim=True)
            device = k_scales.device
            max_pre_keys = torch.vstack([max_pre_keys.to(device), k_scales])
            max_pre_keys = max_pre_keys.amax(dim=0, keepdim=True)
            
            # post_scale
            kv_cache = _ldatas[0]['past_key_value']
            k_cache = kv_cache.key_cache[-1]
            
            keys = k_cache.transpose(1,2).reshape(-1, key_head_size, head_dim)
            new_keys = keys.abs().amax(dim=0, keepdim=True)
            device = new_keys.device
            max_post_keys = torch.vstack([max_post_keys.to(device), new_keys])
            max_post_keys = max_post_keys.amax(dim=0, keepdim=True)
            
        ag.kv_data = initial_kv_data_flag
        
        ## qk_rope_pre_scale
        k_scales = max_pre_keys.amax(dim=0, keepdim=False)
        k_scales = k_scales.unsqueeze(1).expand(-1, 2, head_dim//2)
        k_scales = alpha / k_scales    

        q_scales = torch.cat([1/k_scales for _ in range(num_key_value_group)], dim=1).reshape(-1)
        k_scales = k_scales.reshape(-1)


        ## qk_rope_post_scale
        k_max = max_post_keys.reshape(-1, head_dim)
        k_topk = torch.topk(k_max, top_k)
        device = k_topk.values.device
        
        half_hdim = head_dim // 2
        k_topk_pair_indices = torch.cat((k_topk.indices, torch.where(k_topk.indices >= half_hdim, k_topk.indices - half_hdim, k_topk.indices + half_hdim)), dim=1)
        k_indices = ((torch.arange(key_head_size)*head_dim).reshape(-1,1).to(device) + k_topk_pair_indices).view(-1)
        
        row_idx = torch.arange(k_max.size(0)).unsqueeze(1)
        k_values = k_max[row_idx, k_topk_pair_indices]
        k_values = beta / k_values

        tmp_k = (torch.arange(0,key_head_size)*head_dim*num_key_value_group).reshape(-1,1).to(device) + k_topk_pair_indices
        q_indices = torch.cat([tmp_k + i*head_dim for i in range(num_key_value_group)], dim=1).reshape(-1)
        q_values = torch.cat([1/k_values for _ in range(num_key_value_group)], dim=1).reshape(-1)
        k_values = k_values.reshape(-1)
        
        if is_pre_rope_scale_applied:
            k_scales = k_scales.to(device)
            q_scales = q_scales.to(device)
            
            half_head_dim = head_dim // 2
            
            _k_indices = k_indices.to('cpu').clone()
            _q_indices = q_indices.to('cpu').clone()
            
            doubled_k_indices_list = []
            doubled_q_indices_list = []
            
            for val in _k_indices:
                bottom = val // head_dim
                head_wise_idx = val%head_dim
                
                if head_wise_idx >= half_head_dim:
                    tg = head_wise_idx - half_head_dim
                else:
                    tg = head_wise_idx + half_head_dim
                
                doubled_k_indices_list.append(bottom*head_dim + tg)
            
            _k_indices = torch.hstack([_k_indices, torch.Tensor(doubled_k_indices_list).to(torch.int32)]).reshape(-1)
            
            for val in _q_indices:
                bottom = val // head_dim
                head_wise_idx = val%head_dim
                
                if head_wise_idx >= half_head_dim:
                    tg = head_wise_idx - half_head_dim
                else:
                    tg = head_wise_idx + half_head_dim
                
                doubled_q_indices_list.append(bottom*head_dim + tg)
                
            _q_indices = torch.hstack([_q_indices, torch.Tensor(doubled_q_indices_list).to(torch.int32)]).reshape(-1)
            
            _k_indices = _k_indices.to(device)
            _q_indices = _q_indices.to(device)
            
            k_scales[_k_indices] = 1.0
            q_scales[_q_indices] = 1.0
        
                
        return (q_scales, k_scales), ((q_values, k_values), (q_indices, k_indices))
        
    def apply_post_rope_smooth(self, qts, q_and_k_post_rope_scales, q_and_k_top_k_indices):
        attn_name = 'self_attn'
        attn = find_module_by_exactmatching(qts, attn_name)
        
        q_values, k_values = q_and_k_post_rope_scales
        q_top_k, k_top_k = q_and_k_top_k_indices

        q_smooth_layer = find_module_by_exactmatching(attn, 'post_q_smooth')
        dtype = q_smooth_layer.dtype
        q_smooth_layer.smooth = q_values.to(dtype)
        q_smooth_layer.top_k_indices = q_top_k.to(torch.int32)

        k_smooth_layer = find_module_by_exactmatching(attn, 'post_k_smooth')
        dtype = k_smooth_layer.dtype
        k_smooth_layer.smooth = k_values.to(dtype)
        k_smooth_layer.top_k_indices = k_top_k.to(torch.int32)
    
    def apply_pre_rope_norm(self, qts, q_and_k_pre_rope_smooth_scales):
        
        q_proj = find_module_by_exactmatching(qts, 'q_proj')
        k_proj = find_module_by_exactmatching(qts, 'k_proj')  
        
        q_scale, k_scale = q_and_k_pre_rope_smooth_scales
        
        q_proj.smooth_out = q_scale
        k_proj.smooth_out = k_scale

    def __pre_quantize__(self):
        
        pass
    