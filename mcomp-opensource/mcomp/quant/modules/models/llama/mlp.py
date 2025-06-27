import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
)
from transformers.models.llama import LlamaConfig
from transformers.activations import ACT2FN
from typing import Union, List, Tuple, Dict, Optional
from mcomp.quant.modules.pseudo_module_utils import get_pseudolinear
from mcomp.utils.config import get_qdtype_from_layer_quant_config

class PseudoLlamaMLP(nn.Module):
    def __init__(self, mlp: LlamaMLP, config: LlamaConfig, layer_quant_config, on_dtype, calib_dtype, group_size=128, post_rope:bool=True, dynamic_act_quant:bool=True):
        super().__init__()
        self.config = config
        self.layer_quant_config = layer_quant_config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.group_size = group_size

        gate_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'gate_proj')
        if gate_proj_qdtype is None or gate_proj_qdtype.types.lower() == 'pristine':
            self.gate_proj = mlp.gate_proj
        else:
            
            self.gate_proj = get_pseudolinear(gate_proj_qdtype.types.lower(), mlp.gate_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
                        
        up_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'up_proj')
        if up_proj_qdtype is None or up_proj_qdtype.types.lower() == 'pristine':
            self.up_proj = mlp.up_proj
        else:
            self.up_proj = get_pseudolinear(up_proj_qdtype.types.lower(), mlp.up_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
            
        down_proj_qdtype = get_qdtype_from_layer_quant_config(layer_quant_config, 'down_proj')
        if down_proj_qdtype is None or down_proj_qdtype.types.lower() == 'pristine':
            self.down_proj = mlp.down_proj
        else:
            self.down_proj = get_pseudolinear(down_proj_qdtype.types.lower(), mlp.down_proj, group_size=self.group_size, 
                            dynamic_act_quant=dynamic_act_quant, calib_dtype=calib_dtype)
            
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, *args, **kwargs):
        
        x = self.up_proj(x) * self.act_fn(self.gate_proj(x))
        return self.down_proj(x)

