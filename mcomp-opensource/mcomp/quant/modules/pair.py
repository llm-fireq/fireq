import torch
import torch.nn as nn

from .calib import OnlineCalibrator
from .target import QuantTargetSequence, ResidualBlock
from mcomp.config.configs import MappingElementConfig
from mcomp.utils import find_module_by_suffix
from typing import Union

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from .linears.base import BaseCalibratedModule, BaseCalbiratedLinear

class CalibrationPair:

    _CALIBRATION_PAIR_REGISTRY = {}
    
    def __init__(self, ts: Union[torch.nn.Sequential, ResidualBlock], 
                 me: MappingElementConfig, 
                 name_mapper=None):

        self.ts = ts
        self.index = me.index
        self.block = me.block
        self.alias = me.alias
        self._from = me.names
        self._from_orient:int = me.orientation
        self._to = me.matching_names
        self._to_orient:int = me.matching_orientation
        self.not_allowed = me.not_allowed
        self.norm_contained = False

        self._from_modules = []
        self._to_modules = []
        self._from_preserved_tensor = []
        self._to_preserved_tensor = []
        self.name_mapper = name_mapper
        self.exist_not_mapped = False
        self.invalid_pair = False
        
        self.smoothed = False
        self.rotated = False
        self.reordered = False
        self.rotation_block_size = None
        self.reordering_block_size = None

         # 2 implies input side, for 1 output side
        assert self._from_orient in [0, 1, 2]
        assert self._to_orient in [0, 1, 2]

        self.set_up_pairs()
        CalibrationPair._CALIBRATION_PAIR_REGISTRY[self.index] = self

    def get_pseudo_module_name(self, name):
        if self.name_mapper is None:
            return name

        rname = self.name_mapper.get(name, None)
        rname = rname if rname is not None else name
        return rname
    
    def set_up_pairs(self):
        for name in self._from:
            module = find_module_by_suffix(self.ts, self.get_pseudo_module_name(name))
            if type(module) in ALL_LAYERNORM_LAYERS:
                self.norm_contained = True
                self._from_preserved_tensor.append(module.weight.data.clone())
            else:
                self._from_preserved_tensor.append(None)
            if module is None:
                self.exist_not_mapped = True
            if not (isinstance(module, BaseCalibratedModule) or type(module) in ALL_LAYERNORM_LAYERS):
                self.invalid_pair = True
            self._from_modules.append(module)
        for name in self._to:
            module = find_module_by_suffix(self.ts, self.get_pseudo_module_name(name))
            if type(module) in ALL_LAYERNORM_LAYERS:
                self.norm_contained = True
                self._to_preserved_tensor.append(module.weight.data.clone())
            else:
                self._to_preserved_tensor.append(None)
            if module is None:
                self.exist_not_mapped = True
            if not (isinstance(module, BaseCalibratedModule) or type(module) in ALL_LAYERNORM_LAYERS):
                self.invalid_pair = True
            self._to_modules.append(module)

    def set_rotation(self, rotation:torch.Tensor, block_size=None):
        assert not self.invalid_pair
        assert not self.norm_contained, "This pair has a Normalization Layer."
        if rotation is not None:
            assert isinstance(rotation, torch.Tensor)
            assert rotation.dim() == 2
            assert rotation.shape[0] == rotation.shape[1]
            if block_size is not None:
                assert rotation.shape[0] == block_size
            
            self.rotated = True
            self.rotation_block_size = block_size
        else:
            self.rotated = False
            self.rotation_block_size = None

        attr_names = ["rotation", "rotation_out", "rotation_in"]

        if rotation is None:
            rotation_T = None
        else:
            rotation_T = rotation.T.contiguous()
        
        for module in self._from_modules:
            assert isinstance(module, BaseCalibratedModule)
            
            orient = self._from_orient if isinstance(module, BaseCalbiratedLinear) else 0
            setattr(module, attr_names[orient], (rotation, block_size))
        
        for module in self._to_modules:
            assert isinstance(module, BaseCalibratedModule)
            
            orient = self._to_orient if  isinstance(module, BaseCalbiratedLinear) else 0
            setattr(module, attr_names[orient], (rotation_T, block_size))

    
    def set_smooth(self, smooth):
        assert not self.invalid_pair
        if smooth is not None:
            assert isinstance(smooth, torch.Tensor)
            assert smooth.dim() == 1
            
            self.smoothed = True
        else:
            self.smoothed = False

        attr_names = ["smooth", "smooth_out", "smooth_in"]

        if smooth is None:
            smooth_inv = None
        else:
            orig_dtype = smooth.dtype
            smooth_inv = (1/smooth.to(torch.float32)).to(orig_dtype)
        
            assert smooth_inv.isnan().sum() == 0
            
        for idx, module in enumerate(self._from_modules):
            assert isinstance(module, BaseCalibratedModule) or type(module) in ALL_LAYERNORM_LAYERS
            
            orient = self._from_orient if isinstance(module, BaseCalbiratedLinear) else 0
            if self.norm_contained and type(module) in ALL_LAYERNORM_LAYERS:
                w_device = module.weight.data.device
                w_dtype = module.weight.data.dtype
                module.weight.data = self._from_preserved_tensor[idx].to(device=w_device, dtype=w_dtype)
                if smooth is None:
                    pass
                else:
                    module.weight.data = module.weight.data*smooth.to(device=w_device, dtype=w_dtype)   
            else:
                setattr(module, attr_names[orient], smooth)
        
        for idx, module in enumerate(self._to_modules):
            assert isinstance(module, BaseCalibratedModule) or type(module) in ALL_LAYERNORM_LAYERS
            
            orient = self._to_orient if isinstance(module, BaseCalbiratedLinear) else 0
            if self.norm_contained and type(module) in ALL_LAYERNORM_LAYERS:
                w_device = module.weight.data.device
                w_dtype = module.weight.data.dtype
                module.weight.data = self._to_preserved_tensor[idx].to(device=w_device, dtype=w_dtype)
                if smooth_inv is None:
                    pass # TODO
                else:
                    module.weight.data = module.weight.data*smooth_inv.to(device=w_device, dtype=w_dtype)
            else:
                setattr(module, attr_names[orient], smooth_inv)

    def set_reordering(self, permutation_idx, block_size=None):
        assert not self.invalid_pair
        assert not self.norm_contained, "This pair has a Normalization Layer."
        attr_names = ["reordering", "reordering_out", "reordering_in"]
        
        if permutation_idx is not None:
            if block_size is not None:
                if isinstance(permutation_idx, list):
                    assert len(permutation_idx) ==  block_size
                elif isinstance(permutation_idx, torch.Tensor):
                    assert permutation_idx.dim() == 1
                    assert permutation_idx.shape[0] == block_size
                    
                self.reordering_block_size = block_size
            else:
                self.reordering_block_size = None
            self.reordered = True
        else:
            self.reordered = False
            self.reordering_block_size = None  

        for module in self._from_modules:
            assert isinstance(module, BaseCalibratedModule)
            
            orient = self._from_orient if isinstance(module, BaseCalbiratedLinear) else 0
            setattr(module, attr_names[orient], (permutation_idx, block_size))
        
        for module in self._to_modules:
            assert isinstance(module, BaseCalibratedModule)
            
            orient = self._to_orient if isinstance(module, BaseCalbiratedLinear) else 0
            setattr(module, attr_names[orient], (permutation_idx, block_size))

    def set_priorities(self, smooth:int=-1, reordering:int=0, rotation:int=1):

        attr_names = ["set_priorities", "set_out_priorities", "set_in_priorities"]
        for module in self._from_modules:
            
            if type(module) in ALL_LAYERNORM_LAYERS: continue
            orient = self._from_orient if isinstance(module, BaseCalbiratedLinear) else 0
            
            getattr(module, attr_names[orient])(smooth=smooth,
                                               reordering=reordering,
                                               rotation=rotation)
        
        for module in self._to_modules:
            
            if type(module) in ALL_LAYERNORM_LAYERS: continue
            orient = self._to_orient if isinstance(module, BaseCalbiratedLinear) else 0
            getattr(module, attr_names[orient])(smooth=smooth,
                                               reordering=reordering,
                                               rotation=rotation)
    
    def construct(self):
        for module in self._from_modules:
            if isinstance(module, BaseCalibratedModule):
                module.construct()
        
        for module in self._to_modules:
            if isinstance(module, BaseCalibratedModule):
                module.construct()

    