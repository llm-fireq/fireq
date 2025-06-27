# KVCaches
import torch
import torch.nn as nn
from mcomp.quant.modules.cache import PseudoQuantStaticCache
from mcomp.utils import find_module_by_exactmatching
from transformers import DynamicCache
import gc
from dataclasses import dataclass
import os
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math
from tqdm import tqdm
from mcomp.utils.config import dump_config_to_yaml

@dataclass
class TestConfig:
    kv_quant:bool = False
    i4fp8_linear:bool = False
    group_size:int = 128
    qk_pre_scale:bool = False
    alpha:float = 7.5
    qk_post_scale:bool = False
    beta:float = 7.5
    chwise_scale:bool = False
    scale_type:int = 0
    qkv_scale:float = 0.0
    o_scale:float = 0.0
    ug_scale:float = 0.0
    d_scale:float = 0.0
    v_mul:float = 1.0
    top_k:int = 3
    model_path:str = 'meta-llama/Llama-3.1-8B'
    attn_name:str = 'self_attn'
    mlp_name:str = 'mlp'
    calib_data:str = 'mit-han-lab/pile-val-backup'
    split:str = 'validation'
    text_col:str = 'text'
    batch:int = 16
    seq_len:int = 1024
    orthogonal:bool = False
    awq_smoothing:bool = False
    no_scaled_linear:bool = False
    quant_yaml:str = None
    kv_cache_key:str = None
    max_cache_len:int = None
    task:str = 'wikitext'

    def to_dict(self):
        ret_dict = {}
        for key in TestConfig.__dataclass_fields__.keys():
            if hasattr(self, key):
                ret_dict[key] = getattr(self, key)
        return ret_dict
    
class NoneKVCache:

    def __new__(self):
        return None

_KV_CACHE_REGISTRY = {
    'base': DynamicCache,
    'i4fp8pseudo': PseudoQuantStaticCache,
    'i4fp8': NoneKVCache,
    'i4f16pseudo': PseudoQuantStaticCache,
}

def empty_kwargs_kvcache(*args, **kwargs):
    return {}

def kwargs_i4fp8pseudo(model_config, 
                       batch_size, 
                       max_cache_len, 
                       dtype, 
                       *args,
                       nbits=4, 
                       scale_dtype=torch.float8_e4m3fn, 
                       group_size=128, 
                       device=None, 
                       layer_device_map=None,
                      **kwargs):
    return {
        'model_config': model_config, 
        'batch_size': batch_size,
        'max_cache_len': max_cache_len,
        'dtype': dtype,
        'nbits': nbits,
        'scale_dtype': scale_dtype,
        'group_size':group_size,
        'device': device,
        'layer_device_map': layer_device_map,
    }

def kwargs_i4f16pseudo(model_config, 
                       batch_size, 
                       max_cache_len, 
                       dtype, 
                       *args,
                       nbits=4, 
                       scale_dtype=torch.bfloat16, 
                       group_size=128, 
                       device=None, 
                       layer_device_map=None,
                      **kwargs):
    return {
        'model_config': model_config, 
        'batch_size': batch_size,
        'max_cache_len': max_cache_len,
        'dtype': dtype,
        'nbits': nbits,
        'scale_dtype': scale_dtype,
        'group_size':group_size,
        'device': device,
        'layer_device_map': layer_device_map,
    }


_KV_CACHE_KWARGS_FCN_REGISTRY = {
    'base': empty_kwargs_kvcache,
    'i4fp8pseudo': kwargs_i4fp8pseudo,
    'i4fp8': empty_kwargs_kvcache,
    'i4f16pseudo': kwargs_i4f16pseudo,
}


def get_kvcache_layer_device_map_from_model_and_attn_name(model, attn_name):
    kvcache_layer_device_map = [0] * model.config.num_hidden_layers

    for idx, decoder_layer in enumerate(model.model.layers): # TODO model layers Factory
        attn = find_module_by_exactmatching(decoder_layer, attn_name)
        try:
            device = next(attn.buffers()).device
        except StopIteration:
            device = next(attn.parameters()).device

        kvcache_layer_device_map[idx] = device
    return kvcache_layer_device_map

def get_kvcache_layer_device_map_from_model_hf_device_map(model):
    kvcache_layer_device_map = [0] * model.config.num_hidden_layers
    for k,v in model.hf_device_map.items():
        is_layer_idx = k.split('.')[-1]
        if is_layer_idx.isdigit():
            layer_idx = int(is_layer_idx)
            kvcache_layer_device_map[layer_idx] = v
    return kvcache_layer_device_map
        
        
class CacheClassFactory:

    def __new__(self, *args, **kwargs):
        kv_cache_key = kwargs.get('kv_cache_key', '')
        kv_cache_cls = _KV_CACHE_REGISTRY.get(kv_cache_key, NoneKVCache)
        return kv_cache_cls

class CacheKwargsFactory:

    def __new__(self, *args, **kwargs):
        kv_cache_key = kwargs.get('kv_cache_key', '')
        kv_cache_kwargs = _KV_CACHE_KWARGS_FCN_REGISTRY.get(kv_cache_key, empty_kwargs_kvcache)(*args, **kwargs)
        return kv_cache_kwargs


class CausalModelWrapperForKVCache(nn.Module):

    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model
        self.config = model.config
        self.device = model.device
        #self.hf_device_map = model.hf_device_map
        
        self.init_cache(*args, **kwargs)
        
    def tie_weights(self):
        self.model.tie_weights()

    def eval(self):
        self.model = self.model.eval()
        return self
        
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.reset_cache:
            self.kv_cache.reset()
            past_key_values = self.kv_cache
        else:
            past_key_values = self.kv_cache_cls(**CacheKwargsFactory(*self.cache_args, **self.cache_kwargs))
        kwargs.update({'past_key_values': past_key_values})

        return self.model(*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        if self.reset_cache:
            self.kv_cache.reset()
            past_key_values = self.kv_cache
        else:
            past_key_values = self.kv_cache_cls(**CacheKwargsFactory(*self.cache_args, **self.cache_kwargs))
        kwargs.update({'past_key_values': past_key_values})

        return self.model.generate(*args, **kwargs)

    def init_cache(self, *args, **kwargs):
        self.reset_cache = False
        self.kv_cache = None
        self.kv_cache_cls = CacheClassFactory(*args, **kwargs)
        self.cache_args = args
        self.cache_kwargs = kwargs
        
        if hasattr(self.kv_cache_cls, 'reset'):
            self.reset_cache = True
            self.kv_cache = self.kv_cache_cls(**CacheKwargsFactory(*self.cache_args, **self.cache_kwargs))
            
        gc.collect()
        torch.cuda.empty_cache()
        

def get_kv_wraped_model(model, *args, kv_cache_key='base', **kwargs):
    kv_wraped_model = CausalModelWrapperForKVCache(model, *args, kv_cache_key=kv_cache_key, **kwargs)
    return kv_wraped_model


def get_base_path(save_path):
    hf_path = os.environ.get('HF_HOME','')
    if hf_path:
        hf_path = os.path.join(hf_path,'hub','test')
    base_path = os.path.join(hf_path, save_path)
    return base_path

    
def get_config(path):
    base_path = get_base_path(path)
    try:
        test_config_dict = yaml.safe_load(open(base_path+'/test_config.yaml'))
    except:
        test_config_dict = None
    
    return test_config_dict


def eval_ppl_with_gptq_evaluator(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: str,
    seq_length: int = 2048,
    max_num_samples: int = -1,
) -> float:
    """Evaluate the perplexity of a model on a task using GPTQ style evaluation.

    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        task (str): The task name.
        seq_length (int, optional): The sequence length. Defaults to ``2048``.
        max_num_samples (int, optional): The maximum number of samples to evaluate. Defaults to ``-1``.

    Returns:
        float: The perplexity.
    """
    assert seq_length > 0, "seq_length must be positive"
    if task.startswith("wikitext"):
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    elif task.startswith("pile"):
        test_dataset = load_dataset("pile", task, split="test")
        test_dataset = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    else:
        raise ValueError(f"Invalid task: {task}")

    test_dataset = test_dataset.input_ids.to(model.device)
    num_samples = test_dataset.numel() // seq_length
    if max_num_samples > 0:
        num_samples = min(num_samples, max_num_samples)
    model = model.eval()

    nlls = []
    for i in tqdm(range(num_samples), desc=f"evaluating on {task} with seq_length {seq_length}"):
        batch = test_dataset[:, (i * seq_length) : ((i + 1) * seq_length)]
        with torch.inference_mode():
            shift_logits = model(batch.to(model.device)).logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seq_length
        nlls.append(neg_log_likelihood)
    return math.exp(sum(nlls) / (num_samples * seq_length))



def quantize(test_config, device='cuda'):
    
    model_path = test_config.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype=torch.bfloat16,  ##
                                                 local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.config._attn_implementation = 'sdpa'
    from mcomp.config.configs import Config
    # test_config selector
    
    if test_config.i4fp8_linear and test_config.kv_quant and test_config.qk_pre_scale and test_config.qk_post_scale:
        additional_path = "../quant_configs/llama3_quant_linear_kv_pre_post.yaml"
    elif test_config.i4fp8_linear and test_config.kv_quant and test_config.qk_pre_scale:
        additional_path = "../quant_configs/llama3_quant_linear_kv_pre.yaml"
    elif test_config.i4fp8_linear and test_config.kv_quant and test_config.qk_post_scale:
        additional_path = "../quant_configs/llama3_quant_linear_kv_post.yaml"
    elif test_config.i4fp8_linear and test_config.kv_quant:
        additional_path = "../quant_configs/llama3_quant_linear_kv.yaml"
    elif test_config.i4fp8_linear:
        additional_path = "../quant_configs/llama3_quant_linear_only.yaml"
    elif test_config.kv_quant:
        additional_path = "../quant_configs/llama3_quant_kv_only.yaml"
    else:
        additional_path = "../quant_configs/llama3_quant_nothing.yaml"
        
    if test_config.quant_yaml:
        additional_path = os.path.join("../quant_configs",test_config.quant_yaml)
    
    print(f"additional_path: {additional_path}")
    
    yaml_paths = [
        '../quant_configs/llama/llama3.yaml',
        #'quant_configs/llama3_quant_nothing.yaml',
        #'quant_configs/llama3_quant_linear_only.yaml',
    ]
    yaml_paths.append(additional_path)
    config, invalid_data, remain_data = Config.get_config_from_yamls(yaml_paths)
    
    # test_config selector
    config.calib_data.name = test_config.calib_data
    config.calib_data.batch = test_config.batch
    config.calib_data.seq_len = test_config.seq_len
    
    config.quant.init_orthogonal_transform = test_config.orthogonal
    config.quant.awq_smoothing = test_config.awq_smoothing
    config.quant.chwise_scale = test_config.chwise_scale
    config.quant.scale_type = test_config.scale_type
    config.quant.qkv_scale = test_config.qkv_scale
    config.quant.o_scale = test_config.o_scale
    config.quant.ug_scale = test_config.ug_scale
    config.quant.d_scale = test_config.d_scale
    config.quant.v_mul = test_config.v_mul
    
    config.layer_quant_configs[0].rope.pre_rope_smooth = test_config.qk_pre_scale
    config.layer_quant_configs[0].rope.post_rope_smooth = test_config.qk_post_scale
    config.layer_quant_configs[0].rope.top_k = test_config.top_k
    config.layer_quant_configs[0].rope.alpha = test_config.alpha
    config.layer_quant_configs[0].rope.beta = test_config.beta

    if test_config.no_scaled_linear:
        if hasattr(config.layer_quant_configs[0], 'qdtype'):
            config.layer_quant_configs[0].qdtype[0].types = 'I4F8_F8'
        if hasattr(config.layer_quant_configs[1], 'qdtype'):
            config.layer_quant_configs[1].qdtype[0].types = 'I4F8_F8'
    
    from mcomp.quant.fireQLlamaQuantizer import FireQLlamaQuantizer

    print("Quantization Configuration")
    print("quant")
    print(config.quant)
    print("layer_quant_config")
    print(config.layer_quant_configs)
    quantizer = FireQLlamaQuantizer(model, tokenizer, config, device)
    
    quantizer.quantize_model()
    
    return quantizer, model, tokenizer

def load_model(save_path):
    from mcomp.quant.fireQLlamaQuantizer import FireQLlamaQuantizer

    quantizer, model, tokenizer = FireQLlamaQuantizer.load_quantized(save_path)
    
    return quantizer, model, tokenizer

def save_test_config(save_path, test_config):
    base_path = get_base_path(save_path)

    os.makedirs(base_path, exist_ok=True)

    dump_config_to_yaml(f'{base_path}/test_config', test_config)
    pass

def add_hooks(decoder_layer, no_kwargs_name_list, kwargs_name_list, store_dict=None):
    if store_dict is None:
        raise Exception("store_dict must be provided with a {}")

    def update_dict_tensor(sak:dict, key, tensor:torch.Tensor):
        if sak.get(key, None) is None:
            # create
            sak[key] = tensor.clone().detach().cpu()
        else:
            sak[key] = torch.vstack((sak[key], tensor.clone().detach().cpu()))
     
    def _hook_fn_in(module_name):
        def wrap_hook(module, x):
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) > 0:
                x = x[0]
                # assumed first input is the hidden_state
                
            if isinstance(x, torch.Tensor):
                update_dict_tensor(store_dict, f"{module_name}", x)
    
        return wrap_hook
        
    def _hook_fn_out(module_name):
        def wrap_hook(module, x, y):
            if type(y) in [list, tuple] and len(y) > 0:
                y = y[0]
                # assumed first output is the hidden_state
                
            if isinstance(y, torch.Tensor):
                update_dict_tensor(store_dict, f"{module_name}", y)

        return wrap_hook

    def mname_names_suffix_check(name, names):
        for n in names:
            if name.endswith(n):
                return n
        return ""

    handle_list = []
    for n, m in decoder_layer.named_modules():
        tname = mname_names_suffix_check(n, no_kwargs_name_list)
        if tname:
            handle = m.register_forward_pre_hook(_hook_fn_in(tname))
            handle_list.append(handle)
        for name, tname in kwargs_name_list:
            if n.endswith(name):
                handle = m.register_forward_hook(_hook_fn_out(tname))
                handle_list.append(handle)
    return handle_list

def remove_handles(remove_handles:list):
    for handle in remove_handles:
        handle.remove()
    

