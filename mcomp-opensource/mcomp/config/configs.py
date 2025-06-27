from .build_config import config_section
from typing import Any, Dict, List, Type, Union
from dataclasses import field

# TODO tuple, union으로 확장하기

@config_section('mapping_element')
class MappingElementConfig:
    index:int
    block: str
    names: List[str]
    orientation: int
    matching_names: List[str]
    matching_orientation: 1
    alias: str = None
    not_allowed: List[str] = None


@config_section("smoothed")
class Smoothed:
    method: str = 'auto'
    priority: int = None


@config_section("rotated")
class Rotated:
    method: str = 'auto'
    priority: int = None
    block_size: int = None


@config_section("reordered")
class Reordered:
    method: str = 'auto'
    priority: int = None
    block_size: int = None


@config_section("adds")
class CalibConfig:
    mapping_index: int
    smoothed: Smoothed = None
    rotated: Rotated = None
    reordered: Reordered = None


@config_section("layer_types")
class LayerType:
    type: str
    name: str
    pre_ln:bool
    ln_name: str
    no_ln:bool = False
    proj_in: List[str] = None
    proj_out: List[str] = None
    
@config_section("model_type")
class ModelTypeConfig:
    layer_types: List[LayerType]
    model_norm_name: str = None
    model_norm_pos: str = "last"
    embed_name: str = None
    head_name: str = None
    
@config_section("qdtype")
class QDtypeConfig:

    names: List[str]
    types: str = 'pristine'
    on_dtype: str = None
    calib_dtype: str = None
    dynamic_act_quant: bool = True
    #weight_quant: str = None


@config_section("calibrator_config")
class CalibratorConfig:

    dimension:int = None
    rotation: bool = False
    rotation_block_size:int = None
    smooth: bool = False
    reordering:bool = False
    reordering_block_size:int = None
    rotation_priority:int = 1
    smooth_priority:int = -1
    reordering_priority:int = 0
    from_name: str = None
    to_name: str = None


@config_section("rope")
class ROPEConfig:
    type:str = 'post'
    scaled:bool = False
    k_calib: CalibratorConfig = None
    pre_rope_smooth: bool = False
    alpha: float = 1.0
    post_rope_smooth: bool = False
    beta: float = 1.0
    top_k:int = 3

    # def to_dict(self):
    #     return {
    #         "type": self.type,
    #         "scaled": self.scaled,
    #         "k_calib": self.k_calib.to_dict(),
    #     }
    
@config_section("attn_ops_config")
class AttentionOpsConfig:
    
    in_k_dtype: str = 'bf16'
    in_q_dtype: str = 'bf16'
    in_v_dtype: str = 'bf16'
    attn_implementation: str = 'eager'

@config_section("layer_quant_config")
class LayerQuantConfig:
    
    index: int
    type: str = None
    pristine: bool = False
    qdtype: List[QDtypeConfig] = None
    adds: List[CalibConfig] = None
    rope: ROPEConfig = field(default_factory=ROPEConfig)
    attn_ops: AttentionOpsConfig = None
    kv_quant: bool = False
    qk_calib: CalibratorConfig = None


@config_section("quant_block")
class QuantBlockConfig:

    quant_block_index: int
    attn_index: int = None
    mlp_index: int = None
    moe_index: int = None
   
@config_section("weight_clipping")
class WeightClippingConfig:
    
    num_steps: int = 10
    max_sample_num: int = 4096
    w_batch_size: int = 64
    x_batch_size: int = 1024
    no_wc_layers: List[str] = None
    
@config_section("kv_cache")
class KVCacheConfig:
    
    batch_size:int = 4
    kv_quant:bool = False
    kv_n_bits:int = 4
    kv_group_size:int = 128
    kv_scale_dtype:str = 'fp8'

@config_section("quant")
class QuantConfig:

    quant_block_list: List[QuantBlockConfig]
    strategy:str = "blockwise"
    init_orthogonal_transform:bool = False
    awq_smoothing:bool = False
    batch_size:int = 4
    kv_quant:bool = False
    kv_n_bits:int = 4
    kv_group_size:int = 128
    kv_scale_dtype:str = 'fp8'
    weight_clipping:bool = False
    weight_clipping_config: WeightClippingConfig = field(default_factory=WeightClippingConfig)
    
    w_group_size:int = 128
    a_group_size:int = 0
    
    chwise_scale:bool = False
    scale_type:int = 0
    qkv_scale:float = 0.0
    o_scale:float = 0.0
    ug_scale:float = 0.0
    d_scale:float = 0.0
    v_mul:float = 1.0
    default_multiplier:int = 0
    
    vo_rotate:bool = False
    ffn_o_rotate:bool = False
    mlp_intermediate_rotate:bool = False
    qk_calib_rotate:bool = False
    
    
    
@config_section("model_quant_mapping")
class ModelQuantMapping:
    quant_block_index: int
    mapping_layers: List[int]

@config_section("calib_data")
class CalibrationDataConfig:

    name:str = 'wikitext'
    subset_name:str = None
    split: str = 'validation'
    text_column: str = 'text'
    batch: int = 10
    seq_len: int = 2048

@config_section("root")
class Config:
    adds_mappings: List[MappingElementConfig]
    quant: QuantConfig
    model_quant_mapping: List[ModelQuantMapping]
    calib_data: CalibrationDataConfig
    model_type: ModelTypeConfig
    layer_quant_configs: List[LayerQuantConfig]