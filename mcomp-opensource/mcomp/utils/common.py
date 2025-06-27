import torch
import torch.nn as nn
import os
from safetensors.torch import save_file, load_file


def find_module_by_suffix(module, name):
    found = None
    for n, m in module.named_children():
        if n.endswith(name):
            return m
        if isinstance(m, nn.Module):
            found = find_module_by_suffix(m, name)
            if found is not None:
                return found
    return found

def find_module_by_exactmatching(module, name):
    found = None
    for n, m in module.named_children():
        if n == name:
            return m
        if isinstance(m, nn.Module):
            found = find_module_by_exactmatching(m, name)
            if found is not None:
                return found
    return found

def get_parent_by_suffix(module, suffix):
    found = None
    for n, m in module.named_children():
        if n.endswith(suffix):
            return module
        if isinstance(m, nn.Module):
            found = get_parent_by_suffix(m, suffix)
            if found is not None:
                return found
    return found

def get_parent_by_exactmatching(module, name):
    found = None
    for n, m in module.named_children():
        if n == name:
            return module
        if isinstance(m, nn.Module):
            found = get_parent_by_exactmatching(m, name)
            if found is not None:
                return found
    return found

def get_self_module_name(parent_module, self_module):
    for n,m in parent_module.named_modules():
        if m == self_module:
            return n.split('.')[-1]

def shard_state_dict(state_dict, num_shards=4):
    items = list(state_dict.items())
    shard_size = len(items) // num_shards

    shards = []
    for i in range(num_shards):
        start = i*shard_size
        end = None if i == num_shards -1 else (i + 1) * shard_size
        shard = dict(items[start:end])
        shards.append(shard)
    return shards

def get_base_path(save_path):
    hf_path = os.environ.get('HF_HOME','')
    if hf_path:
        hf_path = os.path.join(hf_path,'hub')
    base_path = os.path.join(hf_path, save_path)
    return base_path

def load_shards(shard_dir, pattern="model_safetensor{}.safetensors", num_shards=4):
    full_state_dict = {}
    hf_path = os.environ.get('HF_HOME','')
    for i in range(num_shards):
        shard = load_file(os.path.join(hf_path, shard_dir, pattern.format(i)))
        full_state_dict.update(shard)
    return full_state_dict

def save_state_dict_with_shard(shard_dir, state_dict, num_shards=4):
    hf_path = os.environ.get('HF_HOME','')
    shard_dir = os.path.join(hf_path, shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    
    shards = shard_state_dict(state_dict, num_shards)
    for i, shard in enumerate(shards):
        save_file(shard, os.path.join(shard_dir,f"model_safetensor{i}.safetensors"))
