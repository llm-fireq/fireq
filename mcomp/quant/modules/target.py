import torch.nn as nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from collections import OrderedDict

class ConservativeSequantial(nn.Sequential):

    def forward(self, x, *args, **kwargs):
        for module in self._modules.values():
            if type(module) in ALL_LAYERNORM_LAYERS:
                x = module(x)
            else:
                x = module(x, *args, **kwargs)
            if type(x) in [tuple, list]:
                x = x[0]
        return x

class ResidualBlock(nn.Sequential):

    def forward(self, x, *args, **kwargs):
        residual = x
        for module in self._modules.values():
            if type(module) in ALL_LAYERNORM_LAYERS:
                x = module(x)
            else:
                x = module(x, *args, **kwargs)
            if type(x) in [tuple, list]:
                x = x[0]
        return x + residual


class QuantTargetSequence:

    def __init__(self, layer_list, config, quant_config):
        pass

    @staticmethod
    def build_target_sequential(layers, residual_group_size=2, names=None):
        assert len(layers) % residual_group_size == 0
        if names is not None:
            assert len(names) == len(layers)

        blocks = []
        for i in range(0, len(layers), residual_group_size):
            if names is None:
                layer_list = []
                for j in range(residual_group_size):
                    layer_list.append(layers[i+j])
                block = ResidualBlock(*layer_list)
            else:
                name_list = []
                layer_list = []
                for j in range(residual_group_size):
                    layer_list.append(layers[i+j])
                    name_list.append(names[i+j])
                block = ResidualBlock(OrderedDict(list(zip(name_list, layer_list))))
            
            blocks.append(block)
            
        return ConservativeSequantial(*blocks)
    
    def forward(self, x, *args, **kwargs):
        
        return self.module(x, *args, **kwargs)

