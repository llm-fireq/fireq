from abc import abstractmethod, ABC
import torch
import gc
from mcomp.config.const import _SCALE_DTYPE_MAP
from .modules.target import QuantTargetSequence
from mcomp.data.act import ActivationGetter
from .modules.calib import OnlineCalibrator
from .modules.target import QuantTargetSequence
from .modules.linears.base import BaseCalbiratedLinear

from mcomp.utils.config import (
  get_layer_type_map, 
  dump_config_to_yaml,
  update_lqc_qk_calib,
  update_lqc_rope_k_calib,
  update_lqc_adds,
  update_lqc_qdtype,
  get_torch_dtype,
)

from mcomp.utils import (
  find_module_by_exactmatching,
  get_parent_by_exactmatching,
  shard_state_dict, 
  load_shards, 
  save_state_dict_with_shard, 
  get_base_path
)

from .modules.pair import CalibrationPair
from mcomp.transform import (
  default_offline_rotation_through_model,
  transform_norm_and_linear_through_model,
)
from mcomp.utils.calib import hadamard
from .modules.cache import BasePseudoQuantCache, PseudoIFPQuantDynamicCacheConfig, PseudoIFPQuantDynamicCache
import numpy as np
import random
from typing import Tuple, List

from mcomp.utils.quant import pseudo_act_quantize_fp8
from mcomp.kernels.triton import triton_fp8_f16_quant

from mcomp.config import QuantBlockConfig, ModelQuantMapping
from copy import deepcopy
import os
import hashlib
import accelerate
import tqdm
from accelerate import (
    init_empty_weights, 
    load_checkpoint_and_dispatch,
    infer_auto_device_map
                        )
from mcomp.utils.module import (
    system_memory,
    gpu_memory,
)


_QUANT_STRATEGY = {
    'blockwise': (2, 2), ## (stride, coverage)
    'layerwise': (1, 1),
    'alternated': (1, 2),
}


class BaseQuantizer(ABC):

    @abstractmethod
    def quantize(self):
        pass
    
    @abstractmethod
    def provide_interested_name_list(self):
        pass
    
    def __pre_quantize__(self):
        pass
    
    def __post_quantize__(self):
        pass
        
    def __init__(self, model, tokenizer, config, device):
        self.model = model
        
        # if it is not eager. change to eager
        # use default
        # if hasattr(model.config, '_attn_implementation'):
        #     if getattr(model.config, '_attn_implementation') == 'sdpa':
        #         setattr(model.config, '_attn_implementation', 'eager')
        # else:
        #     setattr(model.config, '_attn_implementation', 'eager')
        # _attn_implementation_internal
        
        _SCALE_DTYPE_MAP.update(PRISTINE=model.dtype)
            
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.config = config
        self.on_device = device
        self.activation_getter = ActivationGetter(model, device)
        self.group_size = config.quant.w_group_size
        if hasattr(model, 'dtype'):
            self.model_dtype = model.dtype
        else:
            self.model_dtype = next(model.parameters()).dtype

        ### TODO Change to Dynamic Assigning or Quantizer Segregation
        self.init_orthogonal_transform = config.quant.init_orthogonal_transform
        
        self.mlp_cls = None
        self.attn_cls = None
        self.mlp_inf_cls = None
        self.attn_inf_cls = None
        
        self.init_rotation = None
        self.init_rotation_block_size = None
        self.init_rotation_scale = None
        
        self.calib_rotation = None
        self.calib_rotation_block_size = None
        self.calib_rotation_scale = None

        self.layer_type_map = get_layer_type_map(config)
        self.qlayer_map = self.__get_quant_layer_config_dict(config)
        self.qblock_map = self.__get_quant_block_config_dict(config)
        self.model_layer_qblock_map = self.__get_layer_quant_block_mapping(self.model_config, config)

        self.applied_layer_qblock_map = [None] * self.model_config.num_hidden_layers

        assert config.quant.strategy in _QUANT_STRATEGY.keys()
        self.strategy =  config.quant.strategy
        self.stride, self.coverage =  _QUANT_STRATEGY[config.quant.strategy]
        self.cur_index = 0
        
        self.use_normalization_unified_layer = False ## TODO associate with config
        if self.use_normalization_unified_layer:
            self.qts_residual_group_size = 1
        else:
            self.qts_residual_group_size = 2
            
        self.kvcache_config = None
        if self.config.quant.kv_quant is not None and self.config.quant.kv_quant:
            # TODO cache factory using self.config.quant.k_qdtype, v_qdtype
            n_bits = self.config.quant.kv_n_bits
            group_size = self.config.quant.kv_group_size
            scale_dtype = self.config.quant.kv_scale_dtype
            
            self.kvcache_config = PseudoIFPQuantDynamicCacheConfig(
                                                    n_bits, 
                                                    group_size, # TODO scale_dtype, out_dtype
                                                    scale_dtype=get_torch_dtype(scale_dtype),
                                                    device=device
                                                         ) ## KV Cache, device treatment?!

    def init_inp(self):
        dataconfig = self.config.calib_data
        self.activation_getter.get_calib_dataset(self.tokenizer,
                                                 (dataconfig.name, dataconfig.subset_name), 
                                                 split=dataconfig.split, 
                                                 text_col=dataconfig.text_column, 
                                                 batch=dataconfig.batch,
                                                 seq_len=dataconfig.seq_len
                                                )
        self.activation_getter.init_activation(self.model, 
                                               self.model_config, 
                                               self.config, 
                                               remove_sample_data=True)
        

    @classmethod
    def construct_module(cls, module, consturct_cls, *args, **kwargs):
        for n,m in module.named_modules():
            if isinstance(m, consturct_cls):
                m.construct(module, *args, **kwargs)
                
    @classmethod
    def construct_linear(cls, module, *args, **kwargs):
        cls.construct_module(module, BaseCalbiratedLinear, *args, **kwargs)

    def _pre_quantize(self,  
                  rotation=None, 
                  block_size=None, 
                  rotation_scale=None, 
                  transposed=False):

        if self.init_orthogonal_transform:
            self._initial_transform_model_with_orthonormal(rotation, block_size, rotation_scale, transposed)
            
        self.init_inp()
    
    def quantize_model(self):
        # init_orthogonal_transform
        self._pre_quantize()
        
        self.__pre_quantize__()

        for layer_idx in tqdm.tqdm(range(self.model_config.num_hidden_layers), total=self.model_config.num_hidden_layers):

            qts, qlconfig_list = self._get_target_sequence(self.cur_index, self.coverage, residual_group_size=self.qts_residual_group_size)
            pairs = self._get_pairs(qts, qlconfig_list, self.cur_index, coverage=self.coverage)

            name_list, pre_module_name_list_map = self.provide_interested_name_list()
            name_list = [] if name_list is None else name_list
            pre_module_name_list_map = [] if pre_module_name_list_map is None else pre_module_name_list_map
                
            assert type(name_list) == list
            assert type(pre_module_name_list_map) == list
            
            # self.activation_getter._step(self.model,
            #                             self.model_config,
            #                             self.config,
            #                             self.cur_index,
            #                             self.coverage,
            #                             no_kwargs_name_list = name_list,
            #                             kwargs_name_list = pre_module_name_list_map,
            #                             )
            # ## quantize after big step
            # self.quantize(
            #             qts, 
            #             pairs,
            #             self.activation_getter._inp.activations,
            #             self.activation_getter._inp.layer_data
            #                     )
            # else:

            ## quantize prior to small batched step
            self.quantize(
                        qts, 
                        pairs,
                        self.activation_getter._inp.activations,
                        self.activation_getter._inp.layer_data
                                )
            
            kwargs4apply_into_model = {}
            for _act, _ldatas in self._get_gen_for_batched_step(batch_size=self.config.quant.batch_size):
                ## weight_clipping, then provides layer_datas

                if self.config.quant.weight_clipping:
                    if _ldatas is not None:
                        for k, v in _ldatas[0].items():
                            
                            value = kwargs4apply_into_model.get(k, None)
                            if value is None:
                                kwargs4apply_into_model[k] = v
                            else:
                                kwargs4apply_into_model[k] = torch.vstack([value, v])
                 
                
            if self.config.quant.weight_clipping:
                kwargs4apply_into_model.update({
                    'num_steps': self.config.quant.weight_clipping_config.num_steps,
                    'max_sample_num': self.config.quant.weight_clipping_config.max_sample_num,
                    'no_wc_layers': self.config.quant.weight_clipping_config.no_wc_layers,
                    'w_batch_size': self.config.quant.weight_clipping_config.w_batch_size,
                    'x_batch_size': self.config.quant.weight_clipping_config.x_batch_size,
                    'weight_clipping': self.config.quant.weight_clipping,
                    'device': self.on_device,
                                                })
                
            self.apply_into_model(qts, pairs, **kwargs4apply_into_model)
            
            del kwargs4apply_into_model
            gc.collect()
            torch.cuda.empty_cache()
            
            prev_block_index = self.cur_index // 2
            
            self.cur_index += self.stride
            
            next_block_index = self.cur_index // 2
            if next_block_index > prev_block_index:
                self.model.model.layers[prev_block_index] = self.model.model.layers[prev_block_index].to('cpu')
        
        self.__post_quantize__()    
        
    def _apply_into_model(self, qts, pairs, cur_index, **kwargs):

        block_index = cur_index // 2
        is_mlp = cur_index % 2

        lt = self._get_layer_type(cur_index, is_mlp, self.layer_type_map)

        ## model_block_pos 
        model_block = self.model.model.layers[block_index]

        ## attn or mlp
        pseudolayer = find_module_by_exactmatching(qts, lt['name'])
        #corresponding_ln = find_module_by_exactmatching(qts, lt['ln_name'])
        parent = get_parent_by_exactmatching(model_block,lt['name'])

        lq_config = pseudolayer.layer_quant_config
        copied_lqc = deepcopy(lq_config)
        
        update_lqc_qk_calib(qts, copied_lqc)
        update_lqc_rope_k_calib(qts, copied_lqc)
        update_lqc_adds(pairs, copied_lqc)
        update_lqc_qdtype(qts, copied_lqc)
        # TODO update_lqc_rope_post_rope_smooth
        

        if self.applied_layer_qblock_map[block_index] is None:
            if is_mlp:
                self.applied_layer_qblock_map[block_index] = [None, copied_lqc]
            else:
                self.applied_layer_qblock_map[block_index] = [copied_lqc, None]
        else:
             self.applied_layer_qblock_map[block_index][is_mlp] = copied_lqc
            
        pseudolayer.layer_quant_config = copied_lqc
        ## block changing setattr
        if is_mlp:                                                  ### TODO quantization selector
            inf_module = self.mlp_inf_cls.from_pseudo(pseudolayer, triton_fp8_f16_quant)
        else:
            inf_module = self.attn_inf_cls.from_pseudo(pseudolayer, triton_fp8_f16_quant)
        
        delattr(parent, lt['name'])
        setattr(parent, lt['name'], inf_module.to('cpu'))

    def apply_into_model(self, qts, pairs, **kwargs):

        self.construct_module(qts, BaseCalbiratedLinear, **kwargs)

        for i in range(self.coverage):
            self._apply_into_model(qts, pairs, self.cur_index + i, **kwargs)
               
    
    def _initial_transform_model_with_orthonormal(self, 
                                                  rotation=None, 
                                                  block_size=None,
                                                  rotation_scale=None,
                                                  transposed=False
                                                  ):
        transform_norm_and_linear_through_model(self.model, self.model_config, self.config)

        
        block_size = None
        rotation = hadamard(self.model_config.hidden_size)
        rotation_scale = np.sqrt(self.model_config.hidden_size).item()
        transposed = False
        
        self.init_rotation = rotation.clone()
        self.init_rotation_block_size = block_size
        self.init_rotation_scale = rotation_scale
        
        default_offline_rotation_through_model(
            self.model,
            self.model_config,
            self.config,
            rotation,
            block_size=block_size,
            rotation_scale=rotation_scale,
            transposed=transposed
        )

    def __get_quant_layer_config_dict(self, config):
        qlayer_config_dict = {}
        for ql_config in config.layer_quant_configs:
            qlayer_config_dict[ql_config.index] = ql_config
        return qlayer_config_dict
    
    def __get_quant_block_config_dict(self, config):
        qblock_config_dict = {}
        for qb_config in config.quant.quant_block_list:
            qblock_config_dict[qb_config.quant_block_index] = qb_config
        return qblock_config_dict
    
    def __get_layer_quant_block_mapping(self, model_config, config):
        quant_block_mapping = [None] *model_config.num_hidden_layers
        
        for quant_mapping in config.model_quant_mapping:
            qblock_idx = quant_mapping.quant_block_index
            for idx in quant_mapping.mapping_layers:
                if idx == 'all':
                    for i in range(model_config.num_hidden_layers):
                        quant_block_mapping[i] = qblock_idx
                elif type(idx) == int:
                    assert 0 <= idx < model_config.num_hidden_layers
                    quant_block_mapping[idx] = qblock_idx
    
        for layer_idx, qblock_idx in enumerate(quant_block_mapping):
            assert qblock_idx is not None, f"Check Quant Block Mapping as to layer: {layer_idx}"
        return quant_block_mapping
    
    def _get_layer_config(self, cur_index):
        block_index = cur_index // 2
        is_mlp = cur_index % 2
        qblock_config_index = self.model_layer_qblock_map[block_index]
        if is_mlp:
            layer_config_index = self.qblock_map[qblock_config_index].mlp_index
            if layer_config_index is None:
                layer_config_index = self.qblock_map[qblock_config_index].moe_index
            qlayer_config = self.qlayer_map[layer_config_index]
        else:
            layer_config_index = self.qblock_map[qblock_config_index].attn_index
            qlayer_config = self.qlayer_map[layer_config_index]
        return qlayer_config
        
    def _get_layer_type(self, cur_index, is_mlp, layer_type_map):
        if is_mlp: 
            return layer_type_map['mlp']
        else:
            return layer_type_map['attn']

    def _get_pairs(self, qts, quant_layer_config_list, cur_index, coverage=2):
        pairs = []
        is_mlp = cur_index % 2
        adds_list = []
        for ql_config in quant_layer_config_list:
            if ql_config.adds is None: continue
            for add in ql_config.adds:
                adds_list.append(add.mapping_index)
        
        for map_elem in self.config.adds_mappings:
            if coverage == 1:
                if is_mlp and map_elem.block == 'attn': continue
                if not is_mlp and map_elem.block in ['mlp', 'moe']: continue
            if map_elem.index not in adds_list: continue
            _pair = CalibrationPair(qts, map_elem)
            if not _pair.invalid_pair:
                pairs.append(_pair)

        ## add calib pair
        for n,m in qts.named_modules():
            if hasattr(m,'get_calib_pair'):
                ret = m.get_calib_pair()

                if ret:
                    for elem_pair in ret:
                        if not elem_pair.invalid_pair:
                            pairs.append(elem_pair)
            
        return pairs
    
    def get_gen_for_batched_data(self, batch_size):
            
        no_kwargs_name_list, kwargs_name_list = self.provide_interested_name_list()
        return self.activation_getter.batched_forward(self.model, 
                                                   self.model_config, 
                                                   self.cur_index,
                                                   self.coverage,
                                                   batch_size=batch_size,
                                                   no_kwargs_name_list=no_kwargs_name_list,
                                                   kwargs_name_list=kwargs_name_list,
                                                   )
        
    def _get_gen_for_batched_step(self, batch_size):
            
        no_kwargs_name_list, kwargs_name_list = self.provide_interested_name_list()
        return self.activation_getter.batched_step(self.model, 
                                                   self.model_config, 
                                                   self.cur_index,
                                                   self.coverage,
                                                   batch_size=batch_size,
                                                   no_kwargs_name_list=no_kwargs_name_list,
                                                   kwargs_name_list=kwargs_name_list,
                                                   )
    
    def _get_target_sequence(self, cur_index, coverage, residual_group_size=2):
        
        layer_list = []
        name_list = []
        quant_layer_config_list = []
        for i in range(coverage):
            #target_block = model.model.layers[block_index]
            
            block_index = cur_index // 2
            self.model.model.layers[block_index] = self.model.model.layers[block_index].to(self.on_device)
            is_mlp = cur_index % 2

            layer_config = self._get_layer_config(cur_index)
            quant_layer_config_list.append(layer_config)
            lt = self._get_layer_type(cur_index, is_mlp, self.layer_type_map)
            if is_mlp:
                layer_cls = self.mlp_cls
                assert layer_config.type in ['mlp', 'moe']
            else:
                layer_cls = self.attn_cls
                assert layer_config.type == 'attn'
                if self.config.quant.kv_quant is not None and self.config.quant.kv_quant:
                    layer_config.kv_quant = self.config.quant.kv_quant
                # TODO attn factory
            
            ln = getattr(self.model.model.layers[block_index], lt["ln_name"])
            layer = getattr(self.model.model.layers[block_index], lt["name"])
            
            if lt['pre_ln']:
                layer_list.append(ln)
                name_list.append(lt["ln_name"])
            
            layer_list.append(
                layer_cls(layer, self.model_config, layer_config, self.model_dtype, torch.float64) ## TODO on_dtype and calib_dtype into config!!
                             )
            name_list.append(lt["name"])

            if not lt["no_ln"] and not lt["pre_ln"]:
                layer_list.append(ln)
                name_list.append(lt["ln_name"])

            cur_index += 1
    
        qts = QuantTargetSequence.build_target_sequential(layer_list, names=name_list, residual_group_size=residual_group_size)
        qts = qts.to(self.on_device)
        return qts, quant_layer_config_list
    
    def save_quantized(self, save_path):
        
        def tmp_sha_hash_with_poping_id(lqc, pop_key='index'):
            lqc_dict = lqc.to_dict()
            lqc_dict.pop(pop_key)
            h = hashlib.sha256()
            h.update(repr(sorted(lqc_dict.items())).encode())
            return hash(h.hexdigest())
        
        
        ## Construct all the applied layer quant configs
        tmp_lqconfigs = [None]*2*self.model_config.num_hidden_layers
        tmp_lqconfig_index = []
        layer_idx = 0
        lqc_dict = {}
        print(self.applied_layer_qblock_map)
        for lqcs in self.applied_layer_qblock_map:
            for lqc in lqcs:
                ret = lqc_dict.get(tmp_sha_hash_with_poping_id(lqc), None)
                if ret is None:
                    lqc.index = layer_idx
                    tmp_lqconfigs[layer_idx] = lqc
                    lqc_dict.update({tmp_sha_hash_with_poping_id(lqc): layer_idx})
                    tmp_lqconfig_index.append(layer_idx)
                    layer_idx += 1
                else:
                    tmp_lqconfig_index.append(ret)
                   
        ## Construct all the quant block config      
        tmp_qblocks = [None]*self.model_config.num_hidden_layers
        tmp_qblock_index = []
        block_idx = 0
        block_dict = {}
        is_moe = False
        for idx, lqcs in enumerate(self.applied_layer_qblock_map):
            attn_index = tmp_lqconfig_index[2*idx]
            mlp_index = tmp_lqconfig_index[2*idx+1]

            if is_moe:
                moe_index = mlp_index
                mlp_index = None
            else:
                moe_index = None
            
            qblock = QuantBlockConfig(0, attn_index, mlp_index, moe_index)
            ret = block_dict.get(tmp_sha_hash_with_poping_id(qblock, 'quant_block_index'), None)

            if ret is None:
                qblock.quant_block_index = block_idx
                tmp_qblocks[block_idx] = qblock

                block_dict.update(
                    {tmp_sha_hash_with_poping_id(qblock, 'quant_block_index'): block_idx}
                )
                tmp_qblock_index.append(block_idx)
                block_idx += 1
            else:
                tmp_qblock_index.append(ret)
        
        # Construct ModelQuantMapping
        cnt_list = [0] * self.model_config.num_hidden_layers
        
        for qblock_idx in tmp_qblock_index:
            cnt_list[qblock_idx] += 1
            
        max_idx = -1
        max_val = -1
        for qblock_idx, block_cnt in enumerate(cnt_list):
            if block_cnt > max_val:
                max_val = block_cnt
                max_idx = qblock_idx
                
        checked_set = {max_idx}
        model_quant_mappings = []

        model_quant_mappings.append(ModelQuantMapping(max_idx, mapping_layers=['all']))
        checked_set = {max_idx}

        for qblock_idx in tmp_qblock_index:
            if qblock_idx in checked_set:
                continue
            else:
                map_layers = []
                for idx, j in enumerate(tmp_qblock_index):
                    if qblock_idx == j:
                        map_layers.append(j)
                model_quant_mappings.append(ModelQuantMapping(qblock_idx, mapping_layers=map_layers))
                checked_set.add(qblock_idx)
                
        save_quant_config = deepcopy(self.config)
        save_quant_config.model_quant_mapping = model_quant_mappings
        save_quant_config.layer_quant_configs = tmp_lqconfigs[:layer_idx]
        save_quant_config.quant.quant_block_list = tmp_qblocks[:block_idx]
        
        # save saved_quant
        hf_path = os.environ.get('HF_HOME', "")
        base_path = os.path.join(hf_path, 'hub', save_path)
        
        os.makedirs(base_path, exist_ok=True)
        
        dump_config_to_yaml(os.path.join(base_path,'model_quant.config'), save_quant_config)
        
        self.model_config.save_pretrained(base_path)
        self.tokenizer.save_pretrained(base_path)
        
        ## save_model
        accelerator = accelerate.Accelerator()
        accelerator.save_model(self.model, base_path)
        
        print("Saving this Quantized Model has done.") ## TODO logging


    def _load_quantized(self, save_path):
        ## TODO
        # model_config, 및 model load 연결?!
        # with init_empty_weights():

        act_quant_fn = triton_fp8_f16_quant ## TODO act_quant selector

        for layer_index, decoder_layer in tqdm.tqdm(enumerate(self.model.model.layers), 
                                                    total=self.model_config.num_hidden_layers):

            attn_lqc = self._get_layer_config(2*layer_index)
            mlp_lqc = self._get_layer_config(2*layer_index+1)

            attn_lt = self._get_layer_type(2*layer_index, 0, self.layer_type_map)
            mlp_lt = self._get_layer_type(2*layer_index+1, 1, self.layer_type_map)

            attn_parent = get_parent_by_exactmatching(decoder_layer, attn_lt['name'])
            mlp_parent = get_parent_by_exactmatching(decoder_layer, mlp_lt['name'])
            
            attn_module = self.attn_inf_cls(
                                            self.model_config, 
                                            layer_index, 
                                            attn_lqc, 
                                            self.group_size, 
                                            act_quant_fn
                                            )
            mlp_module = self.mlp_inf_cls(
                                        self.model_config, 
                                        mlp_lqc, 
                                        self.group_size, 
                                        act_quant_fn
                                        )

            delattr(attn_parent, attn_lt['name'])
            setattr(attn_parent, attn_lt['name'], attn_module)

            delattr(mlp_parent, mlp_lt['name'])
            setattr(mlp_parent, mlp_lt['name'], mlp_module)
            

        hf_path = os.environ.get('HF_HOME','')
        weight_path = os.path.join(hf_path, 'hub', save_path)
        
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=weight_path,
            device_map='auto'
        )
        
        self.model.eval()
        
    @classmethod
    def load_quantized(cls, save_path, on_device='cpu', no_split_module_classes=["LlamaDecoderLayer"], ): # TODO hardcoding for urgent situation.
        
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from mcomp.config.configs import Config
        
        base_path = get_base_path(save_path)
        yaml_paths = os.path.join(base_path, "model_quant.config.yaml")
        
        model_config = AutoConfig.from_pretrained(base_path)
        config, _, _ = Config.get_config_from_yamls([yaml_paths])
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_config, 
                                                     torch_dtype=model_config.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        act_quant_fn = triton_fp8_f16_quant ## TODO
        
        ## quantizer Selector?!
        self = cls(model, tokenizer, config, device=on_device)
        ## todo layer_selector
        for layer_index, decoder_layer in tqdm.tqdm(enumerate(self.model.model.layers), 
                                                    total=self.model_config.num_hidden_layers):

            attn_lqc = self._get_layer_config(2*layer_index)
            mlp_lqc = self._get_layer_config(2*layer_index+1)

            attn_lt = self._get_layer_type(2*layer_index, 0, self.layer_type_map)
            mlp_lt = self._get_layer_type(2*layer_index+1, 1, self.layer_type_map)

            attn_parent = get_parent_by_exactmatching(decoder_layer, attn_lt['name'])
            mlp_parent = get_parent_by_exactmatching(decoder_layer, mlp_lt['name'])
            
            attn_module = self.attn_inf_cls(
                                            self.model_config, 
                                            layer_index, 
                                            attn_lqc, 
                                            self.group_size, 
                                            act_quant_fn
                                            )
            mlp_module = self.mlp_inf_cls(
                                        self.model_config, 
                                        mlp_lqc, 
                                        self.group_size, 
                                        act_quant_fn
                                        )

            delattr(attn_parent, attn_lt['name'])
            setattr(attn_parent, attn_lt['name'], attn_module)

            delattr(mlp_parent, mlp_lt['name'])
            setattr(mlp_parent, mlp_lt['name'], mlp_module)
            
        hf_path = os.environ.get('HF_HOME','')
        weight_path = os.path.join(hf_path, 'hub', save_path)
        
        
        ## Hard Coding only for this.
        max_mem = {g["id"]: f'{int(g["free_gib"])}GiB' for g in gpu_memory()}
        max_mem["cpu"] = f'{int(system_memory()["available_gib"])}GiB'
        
        print(max_mem)
        # device_map = infer_auto_device_map(
        #     model,
        #     max_memory=max_mem,
        #     no_split_module_classes=no_split_module_classes,
        # )
        
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=weight_path,
            device_map= "auto"  #device_map,
        )
        
        self.model.eval()
        
        return self, model, tokenizer




        
        