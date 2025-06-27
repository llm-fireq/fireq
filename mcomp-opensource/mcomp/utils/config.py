from mcomp.config import CalibratorConfig
from mcomp.quant.modules.calib import OnlineCalibrator
from mcomp.quant.modules.linears.base import BaseCalbiratedLinear
from mcomp.config import Smoothed, Rotated, Reordered, QDtypeConfig
from mcomp.utils import pair as pair_utils

from .common import find_module_by_suffix, find_module_by_exactmatching, get_parent_by_exactmatching, get_parent_by_suffix
from mcomp.config.const import _SCALE_DTYPE_MAP
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from copy import deepcopy
import torch
import torch.nn as nn
import yaml

def get_layer_type_map(config):
    extract_key = ['name', 'pre_ln', 'ln_name', 'no_ln', 'proj_in', 'proj_out']
    layer_type_map = {}
    for layer_type in config.model_type.layer_types:
        tmp_list = []
        module_type = layer_type.type
        for key in extract_key:
            tmp_list.append((key, getattr(layer_type, key)))
    
        layer_type_map.update({module_type: {k:v for k,v in tmp_list}})
    return layer_type_map


class DummyPristineDict(dict):

    def __getitem__(self, key):
        return 'PRISTINE'

def get_type_matching_dict(layer_quant_config):
    
    if layer_quant_config.pristine:
        return DummyPristineDict()
    
    type_matching_dict = {}
    if layer_quant_config.qdtype is not None:
        for qdtype in layer_quant_config.qdtype:
            for name in qdtype.names:
                type_matching_dict[name] = qdtype.types

    return type_matching_dict

def set_definitive_calibrator_config(calibrator: OnlineCalibrator, calibrator_config: CalibratorConfig) -> bool:

    if calibrator.reordering is None:
        calibrator_config.reordering=False
        calibrator_config.reordering_block_size=None
    else:
        calibrator_config.reordering = True
        calibrator_config.reordering_block_size = calibrator._reordering_block_size
        
    if calibrator.rotation is None:
        calibrator_config.rotation = False
        calibrator_config.rotation_block_size = None
    else:
        calibrator_config.rotation = True
        calibrator_config.rotation_block_size = calibrator._rotation_block_size

    if calibrator.smooth is None:
        calibrator_config.smooth = False
    else:
        calibrator_config.smooth = True

    calibrator_config.rotation_priority = calibrator.rotation_priority
    calibrator_config.smooth_priority = calibrator.smooth_priority
    calibrator_config.reordering_priority = calibrator.reordering_priority
    calibrator_config.dimension = calibrator.hidden_dimension
    
    if calibrator_config.reordering or calibrator_config.rotation or calibrator_config.smooth:
        return True
    else:
        return False    

#update qk_calib
def update_lqc_qk_calib(qts, copied_layer_quant_config):
    if copied_layer_quant_config.qk_calib is None:
        return

    from_name = 'post_rope_query_side_onlinecalib' if copied_layer_quant_config.qk_calib.from_name is None else copied_layer_quant_config.qk_calib.from_name 
    to_name = 'post_rope_key_side_onlinecalib' if copied_layer_quant_config.qk_calib.to_name is None else copied_layer_quant_config.qk_calib.to_name

    copied_layer_quant_config.qk_calib.from_name = from_name
    copied_layer_quant_config.qk_calib.to_name = to_name
    
    qk_from_calibrator = find_module_by_suffix(qts, from_name)
    meaning = set_definitive_calibrator_config(qk_from_calibrator, copied_layer_quant_config.qk_calib)
    if not meaning:
        copied_layer_quant_config.qk_calib = None 
        parent_module = get_parent_by_suffix(qts, from_name)
        parent_module.qk_calib = False
        delattr(parent_module, from_name)
        delattr(parent_module, to_name)

#update rope.k_calib
def update_lqc_rope_k_calib(qts, copied_layer_quant_config):
    if copied_layer_quant_config.rope is None:
        return
    if copied_layer_quant_config.rope.k_calib is None:
        return
    if copied_layer_quant_config.rope.type == 'post':
        copied_layer_quant_config.rope.k_calib = None 
        return
        
    lqc_r_kc_from_name = copied_layer_quant_config.rope.k_calib.from_name
    lqc_r_kc_to_name = copied_layer_quant_config.rope.k_calib.to_name

    from_name = 'pre_rope_key_onlinecalib' if lqc_r_kc_from_name is None else lqc_r_kc_from_name 
    to_name = 'pre_rope_key_counter_onlinecalib' if copied_layer_quant_config.qk_calib.to_name is None else lqc_r_kc_to_name

    copied_layer_quant_config.rope.k_calib.from_name = from_name
    copied_layer_quant_config.rope.k_calib.to_name = to_name
    
    k_from_calibrator = find_module_by_suffix(qts, from_name)
    meaning = set_definitive_calibrator_config(k_from_calibrator, copied_layer_quant_config.rope.k_calib)
    if not meaning:
        copied_layer_quant_config.rope.k_calib = None
        parent_module = get_parent_by_suffix(qts, from_name)
        parent_module.k_calib = False
        delattr(parent_module, from_name)
        delattr(parent_module, to_name)


# get priority using pair's from-to
def get_priority(pair, calib_method):
    # calib_method [smooth, rotation, reordering]
    attr_names = ["{method}_priority", "{method}_out_priority", "{method}_in_priority"]
    ret = None
    for module in pair._from_modules:
        if type(module) in ALL_LAYERNORM_LAYERS: continue
        orient = pair._from_orient if isinstance(module, BaseCalbiratedLinear) else 0
        #orient = self._from_orient if not isinstance(module, OnlineCalibrator) else 0
        
        ret = getattr(module, attr_names[orient].format(method=calib_method), None)
        if ret is not None:
            return ret
    for module in pair._to_modules:
        if type(module) in ALL_LAYERNORM_LAYERS: continue
        orient = pair._to_orient if isinstance(module, BaseCalbiratedLinear) else 0
        #orient = self._to_orient if not isinstance(module, OnlineCalibrator) else 0
        
        ret = getattr(module, attr_names[orient].format(method=calib_method), None)
        if ret is not None:
            return ret
    return ret

def update_calib_config(pairs, calib_config):
    tg_pair_list = pair_utils.find_by_index(pairs, calib_config.mapping_index)
    if tg_pair_list:
        tg_pair = tg_pair_list[0]
        
        if tg_pair.smoothed:
            priority = get_priority(tg_pair, 'smooth')
            calib_config.smoothed = Smoothed(priority=priority)

        else:
            calib_config.smoothed = None
            
        if tg_pair.rotated:
            priority = get_priority(tg_pair, 'rotation')
            calib_config.rotated = Rotated(
                                           priority=priority, 
                                           block_size=tg_pair.rotation_block_size
                                          )
        else:
            calib_config.rotated = None
            
        if tg_pair.reordered:
            priority = get_priority(tg_pair, 'reordering')
            calib_config.reordered = Reordered(
                                           priority=priority, 
                                           block_size=tg_pair.reordering_block_size
                                          )
        else:
            calib_config.reordered = None
            
    if calib_config.smoothed or calib_config.rotated or calib_config.reordered:
        return True
    else:
        return False

def update_lqc_adds(pairs, copied_layer_quant_config) -> None:
    if copied_layer_quant_config.adds is None: 
        return
    
    new_adds = []
    for idx in range(len(copied_layer_quant_config.adds)):
        meaning = update_calib_config(pairs, copied_layer_quant_config.adds[idx])
        if not meaning:
            copied_layer_quant_config.adds[idx] = None
        else:
            new_adds.append(copied_layer_quant_config.adds[idx])
    copied_layer_quant_config.adds = new_adds
    
def update_lqc_qdtype(qts, copied_layer_quant_config) -> None:
    
    if copied_layer_quant_config.qdtype is None:
        return
    
    type_map = {
        torch.float8_e4m3fn: 'FP8',
        torch.bfloat16: 'BF16',
        torch.half: 'FP16',
        torch.float16: 'FP16',
        torch.float: 'FP32',
        torch.float32: 'FP32',
        torch.float64: 'FP64',
    }

    qdtype_map_dict = {}
    for qdtype in copied_layer_quant_config.qdtype:
        for mname in qdtype.names:
        
            m = find_module_by_exactmatching(qts, mname)
            
            if isinstance(m, nn.Linear):
                types = type_map[m.weight.dtype]
                on_dtype = None
                calib_dtype = None
                dynamic_act_quant = None
            elif isinstance(m, BaseCalbiratedLinear):
                types = m.linear_dtype
                on_dtype = type_map.get(m.on_dtype, None)
                calib_dtype = type_map.get(m.calib_dtype, None)
                dynamic_act_quant = m.dynamic_act_quant
            else:
                continue
        
            quant_config = QDtypeConfig([], types, on_dtype, calib_dtype, dynamic_act_quant)
        
            dict_ret = qdtype_map_dict.get(quant_config, None)
            if dict_ret is None:
                qdtype_map_dict.update({quant_config: [mname]})
            else:
                qdtype_map_dict[quant_config].append(mname)
    
    qdtype_list = []
    for k, v in qdtype_map_dict.items():
        k.names = v
        qdtype_list.append(k)

    copied_layer_quant_config.qdtype = qdtype_list
    
#update post_rope_smooth
def update_lqc_post_rope_smooth(qts, copied_layer_quant_config):
    if copied_layer_quant_config.rope is None:
        return
    if not copied_layer_quant_config.rope.post_rope_smooth:
        return
    if copied_layer_quant_config.rope.type == 'pre':
        copied_layer_quant_config.rope.k_calib = None 
        copied_layer_quant_config.rope.post_rope_smooth = False
        return

    from_name = 'post_q_smooth'
    to_name = 'post_k_smooth'
    
    k_smooth_layer = find_module_by_suffix(qts, from_name)
    
    meaning = True
    if k_smooth_layer.smooth.sum() == k_smooth_layer.smooth.numel():
        meaning = False
    
    if not meaning:
        copied_layer_quant_config.rope.post_rope_smooth = False
        parent_module = get_parent_by_suffix(qts, from_name)
        parent_module.post_rope_smooth = False
        delattr(parent_module, from_name)
        delattr(parent_module, to_name)

def dump_config_to_yaml(yaml_path, config):
    if yaml_path.endswith('.yaml'):
        pass
    else:
        yaml_path = yaml_path + '.yaml'

    with open(yaml_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)

def get_torch_dtype(key):
    key = key.upper()

    assert key in _SCALE_DTYPE_MAP
    return _SCALE_DTYPE_MAP[key]

def get_qdtype_from_layer_quant_config(lq_config, name):
            
    if lq_config.pristine:
        return None
    
    qdtypeConfigs = lq_config.qdtype
    if qdtypeConfigs is None:
        return None
    
    qdtype = None
    for _qdt in qdtypeConfigs:
        if name in _qdt.names:
            qdtype = _qdt
    return qdtype
        
