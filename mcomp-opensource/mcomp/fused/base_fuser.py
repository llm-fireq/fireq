from abc import ABC, abstractmethod
from mcomp.utils.config import (
  get_layer_type_map, 
)
import tqdm


class BaseTransformerFuser(ABC):
    
    def __init__(self, model, config):
        self.model = model
        self.model_config = model.config
        self.config = config
        self.layer_type_map = get_layer_type_map(config)
        self.layers = self.get_layers(self.model)
    
    @abstractmethod
    def get_fused_attn(self, layer):
        raise NotImplementedError
    
    @abstractmethod
    def get_fused_mlp(self, layer):
        raise NotImplementedError
    
    @abstractmethod
    def get_fused_block(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_layers(self, model):
        raise NotImplementedError
    
    @abstractmethod
    def is_fusable(self, layer):
        raise NotImplementedError
    
    def fuse(self):
        layers = self.get_layers(self.model)
        for layer_idx, layer in tqdm.tqdm(enumerate(layers), total=len(layers)):
            if self.is_fusable(layer):
                attn_args = self.get_fused_attn(layer)
                mlp_args = self.get_fused_mlp(layer)
                self.get_fused_block(layer_idx, *attn_args, *mlp_args)
    
    