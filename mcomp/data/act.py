from datasets import load_dataset
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List, Dict
import gc
import urllib3
import copy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from transformers import DynamicCache

@dataclass
class ActivationState:

    cur_index:int
    calculated:bool
    activations: List[torch.Tensor]=None 
    layer_data: List[Dict[str, torch.Tensor]]=None
    block_step_list: List[int]=None
    attn_kwargs: Dict = None
    decoder_kwargs: Dict = None
    
    
class ActivationGetter:

    def __init__(self, model, on_device='cpu'):

        self._inp = ActivationState(-2, False, None, None, None)
        self.model = model
        self.data_retrieved = False
        self.data_sign = ""
        self.on_device = on_device
        self.sample_data = None
        self.kv_data = False

    def get_calib_dataset(self, tokenizer, _data, split, text_col, batch, seq_len):

        if self.data_retrieved and self.data_sign == f"{_data}_{split}_{text_col}_{batch}_{seq_len}":
            return self.sample_data

        if type(_data) == str:
            dataset = load_dataset(_data,split=split, download_mode="reuse_dataset_if_exists")
        elif type(_data) in [list, tuple]:
            dataset = load_dataset(*_data, split=split, download_mode="reuse_dataset_if_exists")

        token_list = []
        num_tokens = 0
        for idx, data in enumerate(dataset):
            line = data[text_col].strip()
            encoded = tokenizer.encode(line)
            num_tokens += len(encoded)
            token_list.append(torch.Tensor([encoded]))
            if num_tokens >= batch*seq_len:
                break
        tokens = torch.cat(token_list, dim=1)
        
        stride = seq_len
        window = seq_len
        batch_list = []
        for idx, _b in enumerate(range(batch)):
            starting_idx = (idx*stride) % tokens.shape[1]
                
            batch_list.append(tokens[:,starting_idx:starting_idx+window].to(torch.int32))
        
        sample_data = torch.cat(batch_list, dim=0)
        self.sample_data = sample_data
        
        self.data_retrieved = True
        self.data_sign = f"{_data}_{split}_{text_col}_{batch}_{seq_len}"
        return sample_data

    def init_activation(self, model, model_config, config, remove_sample_data=True):

        if self._inp.activations is None:
            self._inp.activations = []

        # layer_kwargs extract
        # from AWQ implementation
        layer_kwargs = {}
        inps = []

        class KwargsCatcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference
        
        ## TODO get_model_layers like autoAWQ
        ## TODO embedding layer .to

        modules = self.model.model.layers
        
        orig_module_device = next(modules[0].parameters()).device
        if orig_module_device != self.on_device:
            modules[0] = modules[0].to(self.on_device)
            
        modules[0] = KwargsCatcher(modules[0])

        with torch.no_grad():
            try:
                self.model(self.sample_data.to(next(self.model.parameters()).device))
            except ValueError:  # work with early exit
                pass
        modules[0] = modules[0].module
        if orig_module_device != self.on_device:
            modules[0] = modules[0].to(orig_module_device)
        
        layer_kwargs = self.model.prepare_inputs_for_generation(self.sample_data, **layer_kwargs)
        layer_kwargs.pop("input_ids")
        layer_kwargs.pop('past_key_value') # no use of KVCache
        self._inp.decoder_kwargs = copy.copy(layer_kwargs)
        
        layer_kwargs.pop('use_cache')
        
        inps = inps[0]
        if remove_sample_data:
            del self.sample_data

        ## TODO module positioning
        ## TODO embedding positioning
        
        self._get_orig_kwargs_devices_dict(layer_kwargs, self.on_device)

        self._inp.activations.append(inps.to(self.on_device))
        self._inp.attn_kwargs = layer_kwargs

    def _step(self, model, model_config, config, cur_index, coverage, no_kwargs_name_list=None, kwargs_name_list=None):
        if not self.data_retrieved: 
            raise Exception("Data has not been retrieved")
        assert coverage <= 2, f"only Coverage, equal or less than 2, is supported, given {coverage}"

        if self._inp.block_step_list is None:
            self._inp.block_step_list = []
        no_kwargs_name_list = [] if no_kwargs_name_list is None else no_kwargs_name_list
        kwargs_name_list = [] if kwargs_name_list is None else kwargs_name_list

        block_idx = cur_index // 2
        assert block_idx < model_config.num_hidden_layers
        is_mlp = cur_index % 2
        saved_block_idx = self._inp.cur_index // 2
        
        if saved_block_idx < block_idx and block_idx not in self._inp.block_step_list:
            if abs(block_idx - saved_block_idx) > 1:
                raise Exception("Invalid Block Index, Someting went wrong during processing activaitons")
            
            # TODO get model layers
            decoder_layer = model.model.layers[block_idx]

            assert len(self._inp.activations) > 0, "init_activation must be called first before going steps"
            self._step_decoder_block_with_hooks(decoder_layer, no_kwargs_name_list, kwargs_name_list)
            self._inp.block_step_list.append(block_idx)

        next_block_idx = (cur_index + coverage - 1) // 2
        if next_block_idx > block_idx and next_block_idx not in self._inp.block_step_list  and next_block_idx < model_config.num_hidden_layers:
            decoder_layer = model.model.layers[next_block_idx]
            self._step_decoder_block_with_hooks(decoder_layer, no_kwargs_name_list, kwargs_name_list)
            self._inp.block_step_list.append(next_block_idx)
        
        self._inp.cur_index = cur_index
        
        if self._inp.cur_index % 2 == 0 and len(self._inp.layer_data) > 1:
            self._inp.activations.pop(0)
            self._inp.layer_data.pop(0)
            self._inp.block_step_list.pop(0)
            
        gc.collect()
        torch.cuda.empty_cache()
            
    def _step_decoder_block_with_hooks(self, decoder_layer, no_kwargs_name_list, kwargs_name_list):
        # if self._inp.layer_data is None:
        #     _act = self._inp.activations.pop(0)
        # else:
        _act = self._inp.activations[-1]
        
        _ACTIVATION_STORE = {}
        handle_list = self.add_hooks(decoder_layer, no_kwargs_name_list, kwargs_name_list, _ACTIVATION_STORE)
        if self.kv_data:
            past_key_value = DynamicCache()
            self._inp.decoder_kwargs.update({'past_key_value': past_key_value})
            
        y = self._step_decoder_block(decoder_layer, _act, self._inp.decoder_kwargs, self.on_device)
        if self.kv_data:
            past_key_value = self._inp.decoder_kwargs.pop('past_key_value')
            _ACTIVATION_STORE.update({'past_key_value': past_key_value})

        self.remove_handles(handle_list)
        self._inp.activations.append(y)
        if self._inp.layer_data is None:
            self._inp.layer_data = []
        self._inp.layer_data.append(_ACTIVATION_STORE)

    def update_dict_tensor(self, sak:dict, key, tensor:torch.Tensor):
        if sak.get(key, None) is None:
            # create
            sak[key] = tensor.clone().detach().cpu()
        else:
            sak[key] = torch.vstack((sak[key], tensor.clone().detach().cpu()))
        
    def add_hooks(self, decoder_layer, no_kwargs_name_list, kwargs_name_list, store_dict=None):
        if store_dict is None:
            raise Exception("store_dict must be provided with a {}")

        def _hook_fn_in(module_name):
            def wrap_hook(module, x):
                if (isinstance(x, list) or isinstance(x, tuple)) and len(x) > 0:
                    x = x[0]
                    # assumed first input is the hidden_state
                    
                if isinstance(x, torch.Tensor):
                    self.update_dict_tensor(store_dict, f"{module_name}", x)
        
            return wrap_hook
            
        def _hook_fn_out(module_name):
            def wrap_hook(module, x, y):
                if type(y) in [list, tuple] and len(y) > 0:
                    y = y[0]
                    # assumed first output is the hidden_state
                    
                if isinstance(y, torch.Tensor):
                    self.update_dict_tensor(store_dict, f"{module_name}", y)

            return wrap_hook

        handle_list = []
        for n, m in decoder_layer.named_modules():
            tname = self.mname_names_suffix_check(n, no_kwargs_name_list)
            if tname:
                handle = m.register_forward_pre_hook(_hook_fn_in(tname))
                handle_list.append(handle)
            for name, tname in kwargs_name_list:
                if n.endswith(name):
                    handle = m.register_forward_hook(_hook_fn_out(tname))
                    handle_list.append(handle)
        return handle_list

    def mname_names_suffix_check(self, name, names):
        for n in names:
            if name.endswith(n):
                return n
        return ""

    def remove_handles(self, remove_handles:list):
        for handle in remove_handles:
            handle.remove()

    @torch.no_grad()
    def _step_decoder_block(self, decoder_layer, activation, decoder_kwargs, device='cpu'):

        orig_kwarg_devices = self._get_orig_kwargs_devices_dict(decoder_kwargs, device)
        
        # orig_act_device = activation.device
        # orig_device = next(decoder_layer.parameters()).device
        
        decoder_layer = decoder_layer.to(device)
        activation = activation.to(device)
        
        y = decoder_layer(activation, **decoder_kwargs)

        #decoder_layer = decoder_layer.to(orig_device)
        
        #self._set_orig_kwargs_devices(decoder_kwargs, orig_kwarg_devices)
                
        ## hidden_state only
        #y = y[0].to(orig_act_device)
        if type(y) in [tuple, list]:
            y = y[0]
        
        return y
        
    def _get_orig_kwargs_devices_dict(self, kwargs, device):
        orig_kwarg_devices = {}
        for k,v in kwargs.items(): 
            if isinstance(v, torch.Tensor):
                orig_kwarg_devices.update({k:v.device})
                kwargs[k] = v.to(device)
            elif type(v) in (tuple, list):
                tmp_list = []
                for i in range(len(v)):
                    if isinstance(v[i], torch.Tensor):
                        tmp_list.append(v[i].device)
                    else:
                        tmp_list.append(None)
                kwargs[k] = tuple([_v.to(device) if isinstance(_v, torch.Tensor) else _v for _v in v ])
                orig_kwarg_devices.update({k:tmp_list})
        return orig_kwarg_devices

    def _set_orig_kwargs_devices(self, kwargs, orig_kwargs):
        for k, _device in orig_kwargs.items():
            if type(_device) in [tuple, list]:
                kwargs[k] = tuple(
                    [ 
                        _v.to(orig_kwargs[k][i]) if isinstance(_v, torch.Tensor) else _v for i, _v in enumerate(kwargs[k]) 
                    ]
                )                
            else:
                kwargs[k] = kwargs[k].to(_device)
                
    def _restore_cur_step(self, cur_index):
        ## block-wise first
        if self._inp.layer_data is None: return
        if len(self._inp.layer_data) == 0: return
        
        self._inp.activations.pop(0)
        self._inp.layer_data.pop(0)
        self._inp.block_step_list.pop(0)
        self._inp.cur_index = cur_index - 2
        
        gc.collect()
        torch.cuda.empty_cache()
        
    def batched_forward(self, model, model_config, cur_index, coverage, batch_size=4, no_kwargs_name_list=None, kwargs_name_list=None):
        initial_activations = self._inp.activations[-1].to('cpu')
        next_activations = None
        total_batch = initial_activations.shape[0]
        initial_attn_mask = None
        if self._inp.decoder_kwargs is not None:
            attn_mask = self._inp.decoder_kwargs.get('attention_mask', None)
            if attn_mask is not None:
                initial_attn_mask = attn_mask.clone().to('cpu')
        
        
        for step in range(0, total_batch, batch_size):
            self._restore_cur_step(cur_index)
            start_idx = step
            end_idx = min(total_batch, start_idx + batch_size)
            cur_activation = initial_activations[start_idx:end_idx].to(self.on_device)
            _batch_size = end_idx - start_idx
            
            if self._inp.decoder_kwargs is not None:
                attn_mask = self._inp.decoder_kwargs.get('attention_mask', None)
                if attn_mask is not None and attn_mask.shape[0] != _batch_size:
                    assert attn_mask.shape[0] >= batch_size, "batch_size must be larger than the sample number"
                    self._inp.decoder_kwargs['attention_mask'] = attn_mask[:_batch_size]
            
            self._inp.activations[-1] = cur_activation
            self._step(model, model_config, None, cur_index, coverage, 
                            no_kwargs_name_list=no_kwargs_name_list,
                            kwargs_name_list=kwargs_name_list)

            if next_activations is None:
                next_activations = cur_activation.to('cpu')
            else:
                next_activations = torch.vstack([next_activations, cur_activation.to('cpu')])
            
            gc.collect()
            torch.cuda.empty_cache()
            yield self._inp.activations, self._inp.layer_data
            
        self._inp.activations[-1] = next_activations.to(self.on_device)
        del initial_activations
        del next_activations
        
        if initial_attn_mask is not None:
            self._inp.decoder_kwargs['attention_mask'] = initial_attn_mask.to(self.on_device)
        del initial_attn_mask
            
        gc.collect()
        torch.cuda.empty_cache()
        
    def batched_step(self, model, model_config, cur_index, coverage, batch_size=4, no_kwargs_name_list=None, kwargs_name_list=None):
        initial_activations = self._inp.activations[-1].to('cpu')
        next_activations = None
        total_batch = initial_activations.shape[0]
        initial_attn_mask = None
        if self._inp.decoder_kwargs is not None:
            attn_mask = self._inp.decoder_kwargs.get('attention_mask', None)
            if attn_mask is not None:
                initial_attn_mask = attn_mask.clone().to('cpu')
        
        
        for step in range(0, total_batch, batch_size):
            self._restore_cur_step(cur_index)
            start_idx = step
            end_idx = min(total_batch, start_idx + batch_size)
            cur_activation = initial_activations[start_idx:end_idx].to(self.on_device)
            _batch_size = end_idx - start_idx
            
            if self._inp.decoder_kwargs is not None:
                attn_mask = self._inp.decoder_kwargs.get('attention_mask', None)
                if attn_mask is not None and attn_mask.shape[0] != _batch_size:
                    assert attn_mask.shape[0] >= batch_size, "batch_size must be larger than the sample number"
                    self._inp.decoder_kwargs['attention_mask'] = attn_mask[:_batch_size]
            
            self._inp.activations[-1] = cur_activation
            self._step(model, model_config, None, cur_index, coverage, 
                            no_kwargs_name_list=no_kwargs_name_list,
                            kwargs_name_list=kwargs_name_list)

            if next_activations is None:
                next_activations = self._inp.activations[-1].to('cpu')
            else:
                next_activations = torch.vstack([next_activations, self._inp.activations[-1].to('cpu')])
            
            gc.collect()
            torch.cuda.empty_cache()
            yield self._inp.activations, self._inp.layer_data
            
        self._inp.activations[-1] = next_activations.to(self.on_device)
        del initial_activations
        del next_activations
        
        if initial_attn_mask is not None:
            self._inp.decoder_kwargs['attention_mask'] = initial_attn_mask.to(self.on_device)
        del initial_attn_mask
            
        gc.collect()
        torch.cuda.empty_cache()
        
