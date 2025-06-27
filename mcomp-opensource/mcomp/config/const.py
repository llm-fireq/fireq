import torch
from enum import Enum

class ScaleDtype(Enum):
    BF16 = torch.bfloat16
    FP16 = torch.float16
    FP8 = torch.float8_e4m3fn
    HALF = torch.half
    FP32 = torch.float32
    FP64 = torch.float64

_SCALE_DTYPE_MAP = {}
def init_scale_dtype_map():
    for member in ScaleDtype:
        _SCALE_DTYPE_MAP[member.name] = member.value

init_scale_dtype_map()

class LinearDtype:
    I4F8_F8 = "I4F8_F8"
    SCALED_I4F8_F8 = "SCALED_I4F8_F8"
    I4_F8 = "I4_F8"
    I4 = "I4"
    I4_I8 = "I4_I8"
    
