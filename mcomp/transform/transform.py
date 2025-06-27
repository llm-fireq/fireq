import torch
import torch.nn as nn
from typing import Iterable
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
)
import tqdm
from mcomp.utils import (
    find_module_by_suffix,
    get_parent_by_suffix,
    get_parent_by_exactmatching,
    find_module_by_exactmatching,
)

from mcomp.utils.config import (
    get_layer_type_map,
)

__all__ = [
    'apply_offline_rotation',
    'transform_norm_and_linear',
    '_supply_kwdict_for_transform_norm_and_linear',
    'transform_norm_and_linear_through_model',
    'default_offline_rotation_through_model',
]

def apply_rotation_in(linear, rotation, block_size=None, rotation_scale=None):

    weight = linear.weight.data
    w_device = weight.device
    assert weight.dim() == 2
    assert rotation.dim() == 2
    org_dtype = weight.dtype
    weight = weight.to(torch.float64)
    rotation = rotation.to(torch.float64).to(w_device)
    #rotation_t = rotation.T.contiguous()

    if rotation_scale is not None:
        if not isinstance(rotation_scale, torch.Tensor):
            rotation_scale = torch.tensor(rotation_scale)
        rotation_scale = rotation_scale.to(torch.float64).to(w_device)
        rotation = rotation / rotation_scale
    
    if block_size is None:
        weight = torch.matmul(weight, rotation).contiguous()    
    else:
        out_dimension, in_dimension = weight.shape
        
        rot_kron = torch.kron(torch.eye(in_dimension//block_size).to(w_device), rotation).to(torch.float64).to(w_device)
        weight = torch.matmul(weight, rot_kron).contiguous()     
        # tmp_weight = weight.T.contiguous().view(-1,block_size,out_dimension).transpose(1,2)
        
        # tmp_weight = torch.matmul(tmp_weight, rotation)
        # weight = tmp_weight.transpose(0,1).reshape(weight_shape)

    linear.weight.data = weight.to(org_dtype)

def apply_rotation_out(linear, rotation, block_size=None, rotation_scale=None):

    weight = linear.weight.data
    w_device = weight.device
    bias = None
    if linear.bias is not None:
        bias = linear.bias.data
        bias = bias.to(torch.float64)
    assert weight.dim() == 2
    assert rotation.dim() == 2
    org_dtype = weight.dtype
    weight = weight.to(torch.float64)
    rotation = rotation.to(torch.float64).to(w_device)
    

    if rotation_scale is not None:
        if not isinstance(rotation_scale, torch.Tensor):
            rotation_scale = torch.tensor(rotation_scale)
        rotation_scale = rotation_scale.to(torch.float64).to(w_device)
        rotation = rotation / rotation_scale
        
    rotation_t = rotation.T.contiguous()
    if block_size is None:
        weight = torch.matmul(rotation_t, weight).contiguous()  
        if bias is not None:
            bias = (bias @ rotation).contiguous() 
    else:
        out_dimension, in_dimension = weight.shape
        
        rot_kron_t = torch.kron(torch.eye(out_dimension//block_size).to(w_device), rotation_t).to(torch.float64).to(w_device)
        rot_kron = torch.kron(torch.eye(out_dimension//block_size).to(w_device), rotation).to(torch.float64).to(w_device)
        weight = torch.matmul(rot_kron_t, weight).contiguous() 
        if bias is not None:
            bias = (bias @ rot_kron).contiguous() 
            #bias = (bias.view(-1, block_size) @ rotation).view(-1)
           
        # weight_shape = weight.shape
        
        # tmp_weight = weight.reshape(-1, block_size, weight_shape[1])
        # weight = torch.matmul(rotation, tmp_weight).reshape(weight_shape)
        # if bias is not None:
        #     bias = (bias.view(-1, block_size) @ rotation).view(-1)
            
    linear.weight.data = weight.to(org_dtype)
    if bias is not None:
        linear.bias.data = bias.to(org_dtype)

# from deepcompressor
def transform_rms_norm_and_linear(norm: nn.LayerNorm | LlamaRMSNorm, next_modules: Iterable[nn.Linear]) -> None:
    """Fuse the weight multiplication of rms norm into the next adjacent linear modules.

    Args:
        norm (`nn.LayerNorm` or `RMSNorm`):
            normalization module.
        next_modules (`Iterable[nn.Linear]`):
            modules after the normalization module.
    """
    ln_w = norm.weight.data.to(dtype=torch.float64)
    norm.weight.data = torch.ones_like(norm.weight.data)
    if hasattr(norm, "bias") and norm.bias is not None:
        ln_b = norm.bias.data.to(dtype=torch.float64)
        norm.bias = None
    else:
        ln_b = None
    for linear in next_modules:
        assert isinstance(linear, nn.Linear)
        dtype = linear.weight.dtype
        fc_w = linear.weight.data.to(dtype=torch.float64)
        ln_w = ln_w.to(fc_w.device)
        linear.weight.data = (fc_w * ln_w).to(dtype=dtype)
        if ln_b is not None:
            ln_b = ln_b.to(fc_w.device)
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.to(dtype=torch.float64) + torch.matmul(fc_w, ln_b)).to(dtype=dtype)

def transform_layer_norm_to_rms_norm(
    norm: nn.LayerNorm | LlamaRMSNorm,
    prev_modules: Iterable[nn.Linear],
    prev_out_channels_dims: int | Iterable[int] = 0,
    norm_parent=None, norm_name=None) -> None:
    """Transform LayerNorm to LlamaRMSNorm.

    Args:
        norm (`nn.LayerNorm` or `RMSNorm`):
            normalization module.
        prev_modules (`Iterable[nn.Linear]`):
            Previous adjacent linear modules.
        prev_out_channels_dims (`int` or `Iterable[int]`, *optional*, defaults to `0`):
            Output channels dimension of the previous modules' weights.
    """

    assert isinstance(norm, nn.LayerNorm)
    assert norm_parent is not None
    assert norm_name is not None
    assert len(norm.normalized_shape) == 1, f"LayerNorm's #dims must be 1, got {len(norm.normalized_shape)}"
    assert norm.bias is None, "LayerNorm's bias must be None, please call `transform_rms_norm_and_linear` in advance"
    # region move substract mean to the previous linear modules
    assert len(prev_modules) > 0, "No previous modules found"
    if isinstance(prev_out_channels_dims, int):
        prev_out_channels_dims = [prev_out_channels_dims] * len(prev_modules)
    for module, dim in zip(prev_modules, prev_out_channels_dims, strict=True):
        if isinstance(module, nn.LayerNorm):
            module.bias = None
        else:
            if isinstance(module, nn.Linear):
                assert dim == 0, "Linear module's output channels dimension is 0"
            elif isinstance(module, nn.Embedding):
                assert dim == 1, "Embedding module's output channels dimension is 1"
            dtype = module.weight.dtype
            w = module.weight.data.to(dtype=torch.float64)
            module.weight.data = w.sub_(w.mean(dim=dim, keepdim=True)).to(dtype=dtype)
            if hasattr(module, "bias") and module.bias is not None:
                b = module.bias.data.to(dtype=torch.float64)
                module.bias.data = b.sub_(b.mean()).to(dtype=dtype)
    # endregion
    # region replace LayerNorm with RMSNorm
    rms = LlamaRMSNorm(hidden_size=norm.normalized_shape[0], eps=norm.eps)
    rms.weight.data = norm.weight.data
    setattr(norm_parent, norm_name, rms)

def transform_norm_and_linear(
    norm: nn.LayerNorm | LlamaRMSNorm,
    next_modules: Iterable[nn.Linear],
    prev_modules: Iterable[nn.Linear] | None = None,
    prev_out_channels_dims: int | Iterable[int] = 0,
    norm_parent=None, norm_name=None,
):
    """Transform the normalization module and the next adjacent linear modules.

    Args:
        next_modules (tp.Iterable[nn.Linear]): Next adjacent linear modules.
        prev_modules (tp.Iterable[nn.Linear]): Previous adjacent linear modules.
        prev_out_channels_dims (int | tp.Iterable[int], optional): Output channels dimension of the previous modules.
            Defaults to ``0``.
    """

    transform_rms_norm_and_linear(norm, next_modules)
    if isinstance(norm, nn.LayerNorm):
        transform_layer_norm_to_rms_norm(norm, prev_modules, prev_out_channels_dims, norm_parent, norm_name)

# region Apply Rotation
def apply_offline_rotation(
    out_ch_layers: Iterable[nn.Linear],
    in_ch_layers: Iterable[nn.Linear],
    rotation: torch.Tensor, block_size: int = None, rotation_scale: torch.Tensor | float = None, transposed = False):
    """Apply Off-line Rotation.

    Args:
        out_ch_layers (`Iterable[nn.Linear]`):
            Previous adjacent linear modules. pristine rotation defaultly will be applied into out_ch_modules
        in_ch_layers (`Iterable[nn.Linear]`):
            later adjacent linear modules. transposed rotation defaultly will be applied into in_ch_modules
        rotation (torch.Tensor):
            Rotation Marix, it must be orthonormal In addition to this, throughout Model layers, the same rotation must be taken.
        block_size (int) : 
            either row or column dimension, It implies block-wise diagonal rotation
        rotation_scale ('torch.Tensor' or 'float'): 
            If rotation_scale given, They will be divided with this at the last time.
        transposed (bool default=False):
            If it is True, transposed rotation will be applied into in_ch_modules vice versa for out_ch_modules
            
    """
    orig_rotation_dtype = rotation.dtype
    rotation = rotation.to(torch.float64)
    if transposed:
        rotation = rotation.T.contiguous()
    
    for layer in out_ch_layers:
        if isinstance(layer, nn.Embedding):
            raise ValueError("Embedding must be into in_ch_layers")
        apply_rotation_out(layer, rotation, block_size=block_size, rotation_scale=rotation_scale)

    for layer in in_ch_layers:
        if isinstance(layer, nn.Embedding):
            apply_rotation_in(layer, rotation, block_size=block_size, rotation_scale=rotation_scale)
        else:
            apply_rotation_in(layer, rotation, block_size=block_size, rotation_scale=rotation_scale)

# endregion

def _supply_kwdict_for_transform_norm_and_linear(model, model_config, config, layer_idx, key='attn'):

    assert layer_idx < model_config.num_hidden_layers

    layer_type_map = get_layer_type_map(config)
    layer = model.model.layers[layer_idx] # TODO get layers
    
    ## head
    if layer_idx == model_config.num_hidden_layers-1 and key == 'head':
        norm = find_module_by_exactmatching(model, config.model_type.model_norm_name)
        head = find_module_by_exactmatching(model, config.model_type.head_name)
        ln_name = config.model_type.model_norm_name
        norm_parent = get_parent_by_exactmatching(model, config.model_type.model_norm_name)
        
        key_set = set(layer_type_map.keys())
        last_layer_key = key_set.intersection({'mlp', 'moe'}).pop()
        last_layer_type = layer_type_map[last_layer_key]
        proj_out_names = last_layer_type['proj_out']
        
        prev_modules = []
        prev_out_channels_dims = []
        
        for name in proj_out_names:
            
            proj = find_module_by_suffix(layer, name)
            prev_modules.append(proj)
            #if isinstance(proj, nn.Linear):
            prev_out_channels_dims.append(0) # out_channel
        
        return {
            "norm": norm,
            "next_modules": [head],
            "prev_modules": prev_modules,
            "prev_out_channels_dims": prev_out_channels_dims,
            "norm_parent": norm_parent,
            "norm_name": ln_name
        }
    

    layer_type = layer_type_map[key]
    key_set = set(layer_type_map.keys())
    
    assert len(key_set) == 2
    
    key_set.remove(key)
    other_key = key_set.pop()
    
    ln_name = layer_type['ln_name']
    proj_in_names = layer_type['proj_in']
    proj_out_names = layer_type['proj_out']

    norm = find_module_by_suffix(layer, ln_name)
    next_modules = []
    for name in proj_in_names:
        proj = find_module_by_suffix(layer, name)
        next_modules.append(proj)

    prev_modules = []
    prev_out_channels_dims = []
    if layer_idx == 0 and key == 'attn':
        # embed and 
        emb = find_module_by_exactmatching(model, config.model_type.embed_name)
        prev_modules.append(emb)
        prev_out_channels_dims.append(1)
    else:
        other_layer_type = layer_type_map[other_key]
        other_proj_out_names = other_layer_type['proj_out']
        prev_layer = model.model.layers[layer_idx-1]
        for name in other_proj_out_names:
            
            proj = find_module_by_suffix(prev_layer, name)
            prev_modules.append(proj)
            #if isinstance(proj, nn.Linear):
            prev_out_channels_dims.append(0) # out_channel

    norm_parent = get_parent_by_suffix(layer, ln_name)
    return {
        "norm": norm,
        "next_modules": next_modules,
        "prev_modules": prev_modules,
        "prev_out_channels_dims": prev_out_channels_dims,
        "norm_parent": norm_parent,
        "norm_name": ln_name
    }
        
def transform_norm_and_linear_through_model(model, model_config, config):

    layer_type_map = get_layer_type_map(config)
    keys = list(layer_type_map.keys())
    keys.sort()

    print("Transforming norm and linear on decoder blocks")
    for layer_idx in tqdm.tqdm(range(0, model_config.num_hidden_layers), total=model_config.num_hidden_layers):
        for key in keys:
            kwdict = _supply_kwdict_for_transform_norm_and_linear(model, model_config, config, layer_idx, key)
            transform_norm_and_linear(**kwdict)

    print("Transforming norm and linear on the head")
    layer_idx = model_config.num_hidden_layers-1
    key = 'head'
    kwdict = _supply_kwdict_for_transform_norm_and_linear(model, model_config, config, layer_idx, key)
    transform_norm_and_linear(**kwdict)
    print("Transforming norm and linear has done")


def _supply_kwdict_for_offline_rotation_transformation(model, 
                                                      model_config, 
                                                      config, 
                                                      layer_idx, 
                                                      rotation, 
                                                      block_size=None, 
                                                      rotation_scale=None, 
                                                      transposed=False, 
                                                      key='attn'):
    
    assert layer_idx < model_config.num_hidden_layers

    layer_type_map = get_layer_type_map(config)
    layer = model.model.layers[layer_idx] # TODO get layers
    
    ## head
    if layer_idx == model_config.num_hidden_layers-1 and key == 'head':
        head = find_module_by_exactmatching(model, config.model_type.head_name)
        
        key_set = set(layer_type_map.keys())
        last_layer_key = key_set.intersection({'mlp', 'moe'}).pop()
        last_layer_type = layer_type_map[last_layer_key]
        proj_out_names = last_layer_type['proj_out']
        
        out_ch_layers = []
        
        for name in proj_out_names:
            
            proj = find_module_by_suffix(layer, name)
            out_ch_layers.append(proj)
            #if isinstance(proj, nn.Linear):

        return {
            "in_ch_layers": [head],
            "out_ch_layers": out_ch_layers,
            "rotation": rotation,
            "block_size": block_size,
            "rotation_scale": rotation_scale,
            "transposed": transposed,
        }

    layer_type = layer_type_map[key]
    key_set = set(layer_type_map.keys())
    
    assert len(key_set) == 2
    
    key_set.remove(key)
    other_key = key_set.pop()
    
    proj_in_names = layer_type['proj_in']
    proj_out_names = layer_type['proj_out']

    in_ch_layers = []
    for name in proj_in_names:
        proj = find_module_by_suffix(layer, name)
        in_ch_layers.append(proj)

    out_ch_layers = []
    if layer_idx == 0 and key == 'attn':
        # embed and 
        emb = find_module_by_exactmatching(model, config.model_type.embed_name)
        in_ch_layers.append(emb)
    else:
        other_layer_type = layer_type_map[other_key]
        other_proj_out_names = other_layer_type['proj_out']
        prev_layer = model.model.layers[layer_idx-1]
        for name in other_proj_out_names:
            
            proj = find_module_by_suffix(prev_layer, name)
            out_ch_layers.append(proj)
            #if isinstance(proj, nn.Linear):

    return {
        "in_ch_layers": in_ch_layers,
        "out_ch_layers": out_ch_layers,
        "rotation": rotation,
        "block_size": block_size,
        "rotation_scale": rotation_scale,
        "transposed": transposed,
    }

def get_original_devcies_with_cuda_assign(linears):
    devices = []
    for linear in linears:
        _dev = next(linear.parameters()).device
        devices.append(_dev)
        
        linear.weight.data  = linear.weight.data.to('cuda')
        if hasattr(linear, 'bias') and linear.bias is not None:
            linear.bias.data  = linear.bias.data.to('cuda')
    return devices

def restore_original_devcies(linears, device_list):

    assert len(linears) == len(device_list)
    for idx, linear in enumerate(linears):
        orig_device = device_list[idx]
        
        linear.weight.data = linear.weight.data.to(orig_device)
        if hasattr(linear, 'bias') and linear.bias is not None:
            linear.bias.data = linear.bias.data.to(orig_device)

    

def default_offline_rotation_through_model(model, 
                                           model_config, 
                                           config,
                                           rotation, 
                                           block_size=None, 
                                           rotation_scale=None, 
                                           transposed=False, 
                                           ):

    layer_type_map = get_layer_type_map(config)
    keys = list(layer_type_map.keys())
    keys.sort()

    ## Logging
    print("Applying offline rotation on decoder blocks")
    for layer_idx in tqdm.tqdm(range(0, model_config.num_hidden_layers), total=model_config.num_hidden_layers):
        
        for key in keys:
            kwdict = _supply_kwdict_for_offline_rotation_transformation(model, model_config, config, layer_idx, rotation, block_size, rotation_scale, transposed, key)
            
            if torch.cuda.is_available():
                in_layer_dev_list = get_original_devcies_with_cuda_assign(kwdict['in_ch_layers'])
                out_layer_dev_list = get_original_devcies_with_cuda_assign(kwdict['out_ch_layers'])
                
            apply_offline_rotation(**kwdict)
            
            if torch.cuda.is_available():
                restore_original_devcies(kwdict['in_ch_layers'], in_layer_dev_list)
                restore_original_devcies(kwdict['out_ch_layers'], out_layer_dev_list)

    print("Applying offline rotation on the head")
    layer_idx = model_config.num_hidden_layers-1
    key = 'head'
    kwdict = _supply_kwdict_for_offline_rotation_transformation(model, model_config, config, layer_idx, rotation, block_size, rotation_scale, transposed, key)
    
    if torch.cuda.is_available():
        in_layer_dev_list = get_original_devcies_with_cuda_assign(kwdict['in_ch_layers'])
        out_layer_dev_list = get_original_devcies_with_cuda_assign(kwdict['out_ch_layers'])
        
    apply_offline_rotation(**kwdict)

    if torch.cuda.is_available():
        restore_original_devcies(kwdict['in_ch_layers'], in_layer_dev_list)
        restore_original_devcies(kwdict['out_ch_layers'], out_layer_dev_list)
    print("Applying offline rotation has done")

