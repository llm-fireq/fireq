import torch
import gc
from mcomp.fused.base_fuser import BaseTransformerFuser
from .mlp import FusedLlamaMLPI4FP8withLN
from .attn import FusedLlamaAttentionI4FP8withLN
from .block import FusedLlamaDecoderLayerLNImported
from transformers.models.llama import LlamaConfig

from mcomp.utils import (
    find_module_by_exactmatching,
)
from mcomp.modules.linears import (
#    LinearI4FP8, LinearI4FP8Scaled, LinearI4BF16FP8 (LinearI4_F8)
    LinearScaled_I4F8_F8, 
    LinearI4F8_F8, 
    LinearI4_F8, 
    LinearI4,
    LinearI4_I8,
)


from mcomp.fused.modules.linear import FastLinearI4FP8, FastLinearI4FP8Scaled

from mcomp.utils.module.quant import get_activation_fcn

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

class QKVSupplyAdaptor:
    
    def __new__(self, attn):
        
        
        
        return attn.qkv_supply
        

class LlamaFuser(BaseTransformerFuser):
    
    def __init__(self,
                 model, 
                 config,
                 ):
        super().__init__(model, config)
        
        self.layer_idx=0
        self.attn_name = 'self_attn'
        self.mlp_name = 'mlp'
        self.attn_ln_name = 'input_layernorm'
        self.mlp_ln_name = 'post_attention_layernorm'
        
    def get_layers(self, model):
        return model.model.layers
    
    def is_fusable(self, layer):
        return True
        
    def get_fused_attn(self, layer):
        
        ## get layer_norm
        attn_ln = find_module_by_exactmatching(layer, self.attn_ln_name)
        
        ## get attn
        attn = find_module_by_exactmatching(layer, self.attn_name)
        assert hasattr(attn, 'group_size')
        group_size = attn.group_size
        layer_idx = attn.layer_idx
        
        # qkv
        qkv_supply = QKVSupplyAdaptor(attn)
        
        act_quant_fn = get_activation_fcn()
        
        fused_attn = FusedLlamaAttentionI4FP8withLN(
                                            self.model_config, 
                                            layer_idx, 
                                            attn_ln,
                                            act_quant_fn,
                                            qkv_supply,
                                            group_size,
                                                    )    
        
        for proj_key in ['q_proj', 'k_proj','v_proj', 'o_proj']:
            proj = getattr(attn, proj_key)
            assert isinstance(proj, LinearI4F8_F8) or isinstance(proj, LinearScaled_I4F8_F8) or isinstance(proj, LinearI4_F8)
            fused_proj  = getattr(fused_attn, proj_key)
            
            scaled_linear_flag = True if isinstance(proj, LinearScaled_I4F8_F8) else False
            
            if isinstance(fused_proj, FastLinearI4FP8):
                _in_features = fused_proj.in_features
                _out_features = fused_proj.out_features
                _group_size = fused_proj.group_size
                if scaled_linear_flag:
                    delattr(fused_attn, proj_key)
                    setattr(fused_attn, proj_key, FastLinearI4FP8Scaled(_in_features, _out_features, _group_size))
                    fused_proj = getattr(fused_attn, proj_key)
                    fused_proj.per_tensor_scale_inv = proj.per_tensor_scale_inv
                    
                fused_proj.weight = proj.weight
                fused_proj.weight_scale = proj.weight_scale
                
                # Attach FireQ CUTLASS kernel
                if proj_key == 'q_proj' or proj_key == "k_proj" or proj_key == "v_proj":
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.kernel = INT4BF16Linear(fused_proj.weight,
                                                                    fused_proj.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                    else:
                        fused_proj.kernel = INT4FP8Linear(fused_proj.weight,
                                                                    fused_proj.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                        

                elif proj_key == 'o_proj':
                    # Potentially INT4FP8LinearAdd
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.kernel = INT4BF16LinearAdd(fused_proj.weight,
                                                                    fused_proj.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                        print("fused_proj.linear.kernel = INT4BF16LinearAdd")
                    else:
                        fused_proj.kernel = INT4FP8LinearAdd(fused_proj.weight,
                                                                    fused_proj.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                        print("fused_proj.linear.kernel = INT4FP8LinearAdd")
                
                if proj.bias is not None:
                    setattr(fused_proj, 'bias', proj.bias)
                    
                if isinstance(proj, LinearI4_F8):
                    fused_proj.scale_dtype =  proj.weight_scale.dtype
            else:
                _in_features = fused_proj.linear.in_features
                _out_features = fused_proj.linear.out_features
                _group_size = fused_proj.linear.group_size
                if scaled_linear_flag:
                    
                    delattr(fused_proj, "linear")
                    setattr(fused_proj, "linear", FastLinearI4FP8Scaled(_in_features, _out_features, _group_size))
                    fused_proj = getattr(fused_attn, proj_key)
                    fused_proj.linear.per_tensor_scale_inv = proj.per_tensor_scale_inv
                
                fused_proj.linear.weight = proj.weight
                fused_proj.linear.weight_scale = proj.weight_scale

                
                # Attach FireQ CUTLASS kernel
                if proj_key == 'q_proj' or proj_key == "k_proj" or proj_key == "v_proj":
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.linear.kernel = INT4BF16Linear(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                    else:
                        fused_proj.linear.kernel = INT4FP8Linear(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                elif proj_key == 'o_proj':
                    # Potentially INT4FP8LinearAdd
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.linear.kernel = INT4BF16LinearAdd(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                    else:
                        fused_proj.linear.kernel = INT4FP8LinearAdd(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)

                if proj.bias is not None:
                    setattr(fused_proj.linear, 'bias', proj.bias)
                    
                if isinstance(proj, LinearI4_F8):
                    fused_proj.scale_dtype =  proj.weight_scale.dtype
    
        return fused_attn,
    
    def get_fused_mlp(self, layer):
        
        ## get layer_norm
        mlp_ln = find_module_by_exactmatching(layer, self.mlp_ln_name)
        
        ## get mlp
        mlp = find_module_by_exactmatching(layer, self.mlp_name)
        assert hasattr(mlp, 'group_size')
        group_size = mlp.group_size
        
        # get act_quant_fn
        act_quant_fn = get_activation_fcn()
        fused_mlp = FusedLlamaMLPI4FP8withLN(
            self.model_config,
            mlp_ln,
            act_quant_fn,
            group_size,
        )
        
        for proj_key in ['gate_proj', 'up_proj','down_proj']:
            proj = getattr(mlp, proj_key)
            assert isinstance(proj, LinearI4F8_F8) or isinstance(proj, LinearScaled_I4F8_F8) or isinstance(proj, LinearI4_F8)
            fused_proj  = getattr(fused_mlp, proj_key)
            
            scaled_linear_flag = True if isinstance(proj, LinearScaled_I4F8_F8) else False
            
            if isinstance(fused_proj, FastLinearI4FP8):
                _in_features = fused_proj.linear.in_features
                _out_features = fused_proj.linear.out_features
                _group_size = fused_proj.linear.group_size
                if scaled_linear_flag:
                    delattr(fused_mlp, proj_key)
                    setattr(fused_mlp, proj_key, FastLinearI4FP8Scaled(_in_features, _out_features, _group_size))
                    fused_proj = getattr(fused_mlp, proj_key)
                    fused_proj.per_tensor_scale_inv = proj.per_tensor_scale_inv
                    
                fused_proj.weight = proj.weight
                fused_proj.weight_scale = proj.weight_scale

                if proj.bias is not None:
                    setattr(fused_proj, 'bias', proj.bias)
                
                if isinstance(proj, LinearI4_F8):
                    fused_proj.scale_dtype =  proj.weight_scale.dtype
            else:
                _in_features = fused_proj.linear.in_features
                _out_features = fused_proj.linear.out_features
                _group_size = fused_proj.linear.group_size
                if scaled_linear_flag:
                    
                    delattr(fused_proj, "linear")
                    setattr(fused_proj, "linear", FastLinearI4FP8Scaled(_in_features, _out_features, _group_size))
                    fused_proj = getattr(fused_mlp, proj_key)
                    fused_proj.linear.per_tensor_scale_inv = proj.per_tensor_scale_inv
                
                fused_proj.linear.weight = proj.weight
                fused_proj.linear.weight_scale = proj.weight_scale
                
                # Attach FireQ CUTLASS kernel
                if proj_key == 'down_proj':
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.linear.kernel = INT4BF16LinearAdd(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                    else:
                        fused_proj.linear.kernel = INT4FP8LinearAdd(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                elif proj_key == 'gate_proj':
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.linear.kernel = INT4BF16LinearSiLU(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                    else:
                        fused_proj.linear.kernel = INT4FP8LinearSiLU(fused_proj.linear.weight,
                                                                    fused_proj.linear.weight_scale,
                                                                    _in_features, _out_features, _group_size)
                elif proj_key == 'up_proj':
                    if isinstance(proj, LinearI4_F8):
                        fused_proj.linear.kernel = INT4BF16LinearMul(fused_proj.linear.weight, 
                                                                fused_proj.linear.weight_scale, 
                                                                _in_features, _out_features, _group_size)
                    else:
                        fused_proj.linear.kernel = INT4FP8LinearMul(fused_proj.linear.weight, 
                                                                fused_proj.linear.weight_scale, 
                                                                _in_features, _out_features, _group_size)
                    
                if proj.bias is not None:
                    setattr(fused_proj.linear, 'bias', proj.bias)
                    
                if isinstance(proj, LinearI4_F8):
                    fused_proj.scale_dtype =  proj.weight_scale.dtype
        
        return fused_mlp,
    
    def get_fused_block(self, layer_idx, fused_attn, fused_mlp, *args):

        assert fused_attn is not None
        assert fused_mlp is not None
        
        fused_block = FusedLlamaDecoderLayerLNImported(
                                                       self.model_config, 
                                                       fused_attn, 
                                                       fused_mlp
                                                       )
        
        self.layers[layer_idx] = None
        gc.collect()
        torch.cuda.empty_cache()
        self.layers[layer_idx] = fused_block

    
    