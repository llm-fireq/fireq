import torch
import torch.nn as nn
from torch.autograd import Function

from mcomp.config.configs import CalibratorConfig


class OnlineCalibratorSmoothInferFunction(Function):

    @staticmethod
    def forward(ctx, x, smooth_factor):

        return smooth_factor*x

            
class OnlineCalibratorReorderingInferFunction(Function):

    @staticmethod
    def forward(ctx, x, permutation, block_size):

        if block_size is None:
            return x[...,permutation]
        else:
            return x.view(*x.shape[:-1],-1, block_size)[...,permutation].view(x.shape)
            
 
class OnlineCalibratorRotationInferFunction(Function):

    @staticmethod
    def forward(ctx, x, rotation, block_size):

        if block_size is None:
            return torch.matmul(x, rotation)
        else:
            return torch.matmul(x.view(*x.shape[:-1],-1,block_size),rotation).view(*x.shape)
                  

class OnlineCalibratorInfer(nn.Module):

    def __init__(self, calibrator_config, hidden_size, x_dtype):
        super().__init__()
        self.fn_list = []
        self.args_list = []
        self.calibrator_config = calibrator_config
        self.x_dtype = x_dtype

        unordered_list = []
        if calibrator_config.rotation:
            block_size = None
            if calibrator_config.rotation_block_size is not None:
                block_size = calibrator_config.rotation_block_size

            buffer_width = hidden_size if block_size is None else block_size
            self.register_buffer('rotation', 
                                 torch.randn(buffer_width,buffer_width).to(x_dtype)
                                )
            
            unordered_list.append((OnlineCalibratorRotationInferFunction, (self.rotation, block_size), calibrator_config.rotation_priority))
        if calibrator_config.smooth:
            self.register_buffer('smooth', 
                                 torch.ones(hidden_size).to(x_dtype)
                                )
            unordered_list.append((OnlineCalibratorSmoothInferFunction, (self.smooth,), calibrator_config.smooth_priority ))
        if calibrator_config.reordering:
            block_size = None
            if calibrator_config.reordering_block_size is not None:
                block_size = calibrator_config.reordering_block_size

            buffer_width = hidden_size if block_size is None else block_size
            self.register_buffer('reordering', 
                                 torch.arange(buffer_width).to(torch.int32)
                                )
            
            unordered_list.append((OnlineCalibratorRotationInferFunction, (self.reordering, block_size), calibrator_config.reordering_priority))

        unordered_list.sort(key = lambda x: x[2])
        
        for fcn, args, _ in unordered_list:
            self.fn_list.append(fcn)
            self.args_list.append(args)

    def forward(self, x):
        if self.fn_list:
            for i, fn in enumerate(self.fn_list):
                args = self.args_list[i]
                x = fn.apply(x, *args)

        return x

    @classmethod
    def from_pseudo(cls, pcalib, x_dtype, module_load_only=False):
        
        hidden_size = pcalib.hidden_dimension
        calibrator_config = CalibratorConfig()
        
        if hasattr(pcalib, "_smooth"):
            calibrator_config.smooth = True
            calibrator_config.smooth_priority = pcalib.smooth_priority
        if hasattr(pcalib, "_rotation"):
            calibrator_config.rotation = True
            calibrator_config.rotation_priority = pcalib.rotation_priority
            calibrator_config.rotation_block_size = pcalib._rotation_block_size
        if hasattr(pcalib, "_reordering"):
            calibrator_config.reordering = True
            calibrator_config.reordering_priority = pcalib.reordering_priority
            calibrator_config.reordering_block_size = pcalib._reordering_block_size
        
        new_module = cls(calibrator_config ,hidden_size, x_dtype)

        if module_load_only:
            return new_module
            
        if calibrator_config.smooth and pcalib._smooth is not None:
            new_module.smooth = pcalib._smooth.clone().to(x_dtype)
        if calibrator_config.rotation and pcalib._rotation is not None:
            new_module.rotation = pcalib._rotation.clone().to(x_dtype)
        if calibrator_config.reordering and pcalib._reordering is not None:
            new_module.reordering = pcalib._reordering.clone().to(torch.int32)

        return new_module

        