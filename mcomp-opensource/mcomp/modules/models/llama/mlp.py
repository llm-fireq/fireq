import torch
import torch.nn as nn
from torch.autograd import Function
import triton
import triton.language as tl


from mcomp.utils import (
  find_module_by_exactmatching,
)

from mcomp.utils.config import get_type_matching_dict

from ...linears import (
    LinearScaled_I4F8_F8, 
    LinearI4F8_F8, 
    LinearI4_F8, 
    LinearI4,
    LinearI4_I8,
)
from mcomp.quant.modules.linears import (
    PseudoQuantLinearI4F8_F8, 
    PseudoQuantLinearScaled_I4F8_F8, 
    PseudoQuantLinearI4_F8, 
    PseudoQuantLinearI4,
    PseudoQuantLinearI4_I8,
)

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
)
from transformers.models.llama import LlamaConfig
from transformers.activations import ACT2FN

from typing import Union, List, Tuple, Dict, Optional
from mcomp.modules.module_utils import get_linear

class LlamaMLPInfer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_quant_config, group_size, act_quant_fn):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # layer_quant_config.pristine = ture should be filtered out before MLPInfer.__init__() ?!
        self.group_size = group_size
        self.act_quant_fn = act_quant_fn

        self.type_matching_dict = get_type_matching_dict(layer_quant_config)

        ## TODO Valididation Logic

        up_proj_type = self.type_matching_dict.get('up_proj', None)
        self.up_proj = get_linear(up_proj_type, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, group_size=group_size)

        gate_proj_type = self.type_matching_dict.get('gate_proj', None)
        self.gate_proj = get_linear(gate_proj_type, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, group_size=group_size)
        
        down_proj_type = self.type_matching_dict.get('down_proj', None)
        self.down_proj = get_linear(down_proj_type, self.intermediate_size, self.hidden_size, bias=config.mlp_bias, group_size=group_size )
        
        self.act_fn = ACT2FN[config.hidden_act]
                
    def forward(self, x):

        gate_score = self.act_fn(self.gate_proj(x))
        intermediate = gate_score * self.up_proj(x)
        
        return self.down_proj(intermediate)
    
    @classmethod
    def from_pseudo(cls, pqmlp, act_quant_fn, module_load_only=False):

        config = pqmlp.config
        layer_quant_config = pqmlp.layer_quant_config
        group_size = pqmlp.group_size

        new_module = cls(config, layer_quant_config, group_size, act_quant_fn)
        if module_load_only:
            return new_module

        for proj_name in ['up_proj', 'gate_proj', 'down_proj']:
            proj = find_module_by_exactmatching(pqmlp, proj_name)
            if isinstance(proj, nn.Linear):
                setattr(new_module, proj_name, proj)
            else:
                # TMP
                #setattr(new_module, proj_name, LinearI4FP8Scaled.from_pseudo(proj, group_size)) 
                
                if isinstance(proj, PseudoQuantLinearI4F8_F8):
                    setattr(new_module, proj_name, LinearI4F8_F8.from_pseudo(proj, group_size)) 
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
                
                # if proj_name in ['gate_proj', 'up_proj'] and isinstance(proj, PseudoQuantLinearI4FP8Scaled):
                #     setattr(new_module, proj_name, LinearI4FP8Scaled.from_pseudo(proj, group_size)) 
                # else:
                #     setattr(new_module, proj_name, LinearI4FP8.from_pseudo(proj, group_size))  ## TODO type-dependent 

        return new_module

                