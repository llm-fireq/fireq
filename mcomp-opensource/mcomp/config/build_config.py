from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Type, Tuple
import yaml
from typing import get_origin, get_args
import hashlib

def _load_yaml_files(yaml_paths: List[str]) -> List[Dict]:
    """
    open the give yaml_paths in list.

    Args: List[yaml_path]
    Return: List[yaml_dict]
    """
    return [yaml.safe_load(open(path)) for path in yaml_paths]

def load_from_yamls(yaml_paths:List[str]) -> Dict:
    """
    Given yaml_paths in list, merge all the non-root object in the type of list.

    Args: List[yaml_path]
    Return: Dict
    """
    yaml_dict_list = _load_yaml_files(yaml_paths)

    merged_dict = {}
    
    for ydict in yaml_dict_list:
        for key, value in ydict.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                raise ValueError(f"Repeated Key: {key}")
    return merged_dict


CONFIG_REGISTRY: Dict[str, Type] = {}

def _to_dict(data_cls):
    ret_dict = {}
    keys = data_cls.__dataclass_fields__.keys()
    for key in keys:
        val = getattr(data_cls, key)
        
        if val is None: continue

        if is_dataclass(val):
            ret_dict.update({str(key): val.to_dict()})
            continue

        if type(val) in [tuple, list]:
            if len(val) > 0:
                if is_dataclass(val[0]):
                    ret_dict.update({str(key): [v.to_dict() for v in val]})
                else:
                    ret_dict.update({str(key): val})
                continue
            else:
                continue
        
        ret_dict.update({str(key): val})
    return ret_dict

def sha_hash(data_cls):

    h = hashlib.sha256()
    h.update(repr(sorted(data_cls.to_dict().items())).encode())
    return hash(h.hexdigest())
    

def config_section(name: str = None):
    """
    Decorator to declare Configs

    Args: name (str, default: None) 
    name is used to construct config files.

    @config_section("quant")
    class QuantConfig:
        pass
    """

    def wrapper(cls):
        section_name = name or cls.__name__.lower()
        setattr(cls, '__config_section__', section_name)
        CONFIG_REGISTRY[section_name] = cls

        setattr(cls, "field_tree", {})
        setattr(cls, "build_field_tree", build_field_tree)
        setattr(cls, "get_config_from_yamls", get_config_from_yamls)
        setattr(cls, "to_dict", _to_dict)
        setattr(cls, "__hash__", sha_hash)
        
        return dataclass(cls)
    return wrapper
  
def __dfs_build_field_tree(config, config_dict: Dict) -> None:
    
    for f in fields(config):
        list_type = False
        if f.name not in config_dict:
            origin = get_origin(f.type)

            origin_type = f.type
            if origin in (list, List):
                origin_type = get_args(f.type)[0]
                list_type = True
                
            if is_dataclass(origin_type):
                config_dict[f.name] = origin_type,{}, list_type
                __dfs_build_field_tree(origin_type, config_dict[f.name][1])
            else:
                config_dict[f.name] = origin_type, None, None


def _build_field_tree(config, config_dict: Dict) -> None:
    __dfs_build_field_tree(config, config_dict)


def safe_instantiate(cls, input_dict: Dict, cls_member_dict: Dict):
    valid_dict = {}
    invalid_dict = {}
    for k,v in input_dict.items():
        if k in cls_member_dict:
            valid_dict[k] = v
        else:
            invalid_dict[k] = v
    return cls(**valid_dict), invalid_dict

def extract_dataclass_from_field_tree(field_tree:Dict) -> Dict:
    ret_dict = {}
    for k, v in field_tree.items():
        if is_dataclass(v[0]):
            ret_dict[k] = v
            continue
        origin = get_origin(v[0])

        origin_type = v[0]
        if origin in (list, List):
            origin_type = get_args(v[0])[0]
        if is_dataclass(origin_type):
            ret_dict[k] = origin_type
    return ret_dict

def is_config(config) -> bool:
    if is_dataclass(config):
        return True

    origin = get_origin(config)
    if origin in (list, List):
        origin_type = get_args(config)
        if is_dataclass(origin_type):
            return True

    return False

def _dfs_get_config(config_cls: Type, config_node: Tuple[Type,Dict,bool], merged_node: Dict, remain_node: Dict, invalid_data: Dict):

    # intersection check
    config_set = set(config_node[1].keys())
    yaml_set = set(merged_node.keys())
    intersect = yaml_set.intersection(config_set)
    difference = yaml_set - config_set

    for key in difference:
        remain_node[key] = merged_node[key]

    has_children = set(extract_dataclass_from_field_tree(config_node[1]).keys())
    param_dict = {}
    
    for key in intersect:
        try:
            if key in has_children:
                is_list = config_node[1][key][2]
                if is_list:
                    children_list = []
                    invalid_list = []
                    remain_node[key] = []
                    invalid_data[key] = []
    
                    for data_dict in merged_node[key]:
                        remain_node[key].append({})
                        invalid_data[key].append({})
                        children_config, _invalid_data = _dfs_get_config(config_node[1][key][0], 
                                                      config_node[1][key], 
                                                      data_dict, 
                                                      remain_node[key][-1], 
                                                      invalid_data[key][-1],
                                                      )
                        children_list.append(children_config)
                        invalid_list.append(_invalid_data)
                    invalid_data[key] = invalid_list
                    param_dict[key] = children_list
                else:
                    remain_node[key] = {}
                    invalid_data[key] = {}
                    children_config, _invalid_data = _dfs_get_config(config_node[1][key][0], 
                                                      config_node[1][key], 
                                                      merged_node[key], 
                                                      remain_node[key], 
                                                      invalid_data[key],
                                                      )
                    param_dict[key] = children_config
                    invalid_data[key] = _invalid_data
            else:
                param_dict[key] = merged_node[key]
        except:
            raise KeyError(f"{key} config should be revised")

    return safe_instantiate(config_cls, param_dict, config_node[1])

@classmethod
def build_field_tree(config_cls: Type):
    _build_field_tree(config_cls, getattr(config_cls, "field_tree"))


@classmethod
def get_config_from_yamls(config_cls: Type, yaml_paths:List[str]):
    """
    ex) config, invalid_data, remain_data = Config.get_config_from_yamls(yaml_paths)
    """
    remain_node = {}
    invalid_data = {}

    merged_dict = load_from_yamls(yaml_paths)
    config_cls.build_field_tree()
    
    config_instant, _ = _dfs_get_config(config_cls, (config_cls, config_cls.field_tree, False), merged_dict, remain_node, invalid_data)

    return config_instant, invalid_data, remain_node, 