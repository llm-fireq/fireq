import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Union, List, Tuple, Dict
from .linears.base import BaseCalibratedModule

class OnlineCalibratorSmoothFunction(Function):

    @staticmethod
    def forward(ctx, x, smooth_factor):

        ctx.save_for_backward(x, smooth_factor)
        return smooth_factor*x

    @staticmethod
    def backward(ctx, grad_output):
        x, smooth_factor = ctx.saved_tensors

        gard_smooth_factor = (grad_output*x)                                    
        grad_smooth_factor = gard_smooth_factor.sum(dim=tuple(torch.arange(gard_smooth_factor.dim()-1)))
            
        grad_x = smooth_factor*grad_output

        return grad_x, grad_smooth_factor
            
class OnlineCalibratorReorderingFunction(Function):

    @staticmethod
    def forward(ctx, x, permutation, block_size):
        ctx.save_for_backward(x, permutation)
        ctx.block_size = block_size

        if block_size is None:
            return x[...,permutation]
        else:
            return x.view(*x.shape[:-1],-1, block_size)[...,permutation].view(x.shape)
            
    @staticmethod
    def backward(ctx, grad_output):
        x, permutation = ctx.saved_tensors
        block_size = ctx.block_size

        if block_size is None:
            grad_x = grad_output[...,permutation],
        else:
            grad_x = grad_output.view(*grad_output.shape[:-1],-1, block_size)[...,permutation].view(grad_output.shape)

        return grad_x, None, None    

class OnlineCalibratorRotationFunction(Function):

    @staticmethod
    def forward(ctx, x, rotation, block_size):
        ctx.save_for_backward(x, rotation)
        ctx.block_size = block_size

        if block_size is None:
            return torch.matmul(x, rotation)
        else:
            return torch.matmul(x.view(*x.shape[:-1],-1,block_size),rotation).view(*x.shape)
                  
    @staticmethod
    def backward(ctx, grad_output):
        x, rotation = ctx.saved_tensors
        block_size = ctx.block_size

        d, _ = rotation.shape
        if block_size is None:
            grad_x = grad_output@rotation
            grad_rotation = grad_output.view(-1, grad_output.shape[-1]).t() @ x.view(-1, d)   
        else:
            grad_x = (grad_output.view(*grad_output.shape[:-1],-1,block_size) @ rotation).view(*grad_output.shape)
            #grad_rotation = (grad_output.view(-1, grad_output.shape[-1]).t()).view(block_size, -1) @ x.reshape(-1, x.shape[-1]).view(-1, block_size)
            grad_rotation = (grad_output.view(-1, grad_output.shape[-1]).t()).view(block_size, -1) @ x.view(-1, block_size)

        return grad_x, grad_rotation, None 
    
        
class OnlineCalibrator(BaseCalibratedModule):

    def __init__(self, hidden_dimension, on_dtype:torch.dtype=torch.bfloat16, calib_dtype:torch.dtype=torch.float32):
        super().__init__()
        
        self.hidden_dimension = hidden_dimension
        self.calib_dtype = calib_dtype
        self.on_dtype = on_dtype
        
        self._smooth = None
        self._rotation = None
        self._reordering = None 

        self._rotation_block_size = None
        self._reordering_block_size = None
        
        self.smooth_priority = -1
        self.reordering_priority = 0
        self.rotation_priority = 1

        self.fcn_dict = {
            "smooth": OnlineCalibratorSmoothFunction.apply,
            "rotation": OnlineCalibratorRotationFunction.apply,
            "reordering": OnlineCalibratorReorderingFunction.apply,
        }

    def forward(self, x, *args, **kwargs):

        apply_list = list(zip( [self.smooth_priority,
                                     self.rotation_priority,
                                     self.reordering_priority], ["smooth", 
                                                                    "rotation", 
                                                                    "reordering"]  ))

        apply_list.sort(key=lambda x: x[0])
        args_dict = {
            "smooth": [None, self.smooth],
            "rotation": [None, self.rotation, self._rotation_block_size],
            "reordering": [None, self.reordering, self._reordering_block_size],
        }

        ## apply
        ret = x
        for _, rep in apply_list:
            args = args_dict.get(rep, None)
            if args is None: continue
            fcn = self.fcn_dict.get(rep, None)
            if fcn is None: continue
            if args[1] is None: continue
            args[0] = ret
            ret = fcn(*args)    

        ret = ret.to(self.on_dtype)
        return ret

    def set_priorities(self, smooth:int=-1, reordering:int=0, rotation:int=1):
        self.smooth_priority = smooth
        self.reordering_priority = reordering
        self.rotation_priority = rotation

    @property 
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, smoothing_factor:torch.Tensor):
        if smoothing_factor is None:
            if self._smooth is not None:
                del self._smooth
                torch.cuda.empty_cache()
            self._smooth = None
            return
            
        assert isinstance(smoothing_factor, torch.Tensor)
        assert smoothing_factor.dim() in [1, 2]
        dim = smoothing_factor.dim()

        if dim == 2:
            h,w = smoothing_factor.shape
            assert h*w == max(h,w), "For dim==2 smoothing factor, an axis should be one, vector only"
            smoothing_factor = smoothing_factor.view(-1)
        
        sf_size = smoothing_factor.numel()
        assert sf_size == self.hidden_dimension

        if self.smooth is None:
            del self._smooth
            
            self.register_buffer('_smooth', 
                                 smoothing_factor.detach().clone().to(
                                    dtype=self.on_dtype,
                                 ),
                                )
        else:
            self._smooth = smoothing_factor.detach().clone().to(dtype=self.on_dtype)
    
    @property 
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation:Tuple[torch.Tensor, int]):

        rotation_matrix, block_size = rotation
        if rotation_matrix is None:
            if self._rotation is not None:
                del self._rotation
                torch.cuda.empty_cache()
            self._rotation = None
            self._rotation_block_size = block_size
            return
        
        assert isinstance(rotation_matrix, torch.Tensor)
        assert rotation_matrix.dim() == 2

        h, w = rotation_matrix.shape
        assert h == w

        if block_size is None:
            assert self.hidden_dimension == h
        else:
            assert (self.hidden_dimension % block_size) == 0
            assert block_size == h
        
        if self.rotation is None:
            del self._rotation
            
            self.register_buffer('_rotation', 
                                 rotation_matrix.detach().clone().to(
                                    dtype=self.on_dtype,
                                 ),
                                )
        else:
            self._rotation = rotation_matrix.detach().clone().to(dtype=self.on_dtype)    
        self._rotation_block_size = block_size
               
    @property 
    def reordering(self):
        return self._reordering

    @reordering.setter
    def reordering(self, order:Tuple[Union[List, torch.Tensor], int]):

        ordering_index, block_size = order
        if ordering_index is None:
            if self._reordering is not None:
                del self._reordering
                torch.cuda.empty_cache()
            self._reordering = None
            self._reordering_block_size = block_size
            return

        sum_ordering_index:int = 0
        if isinstance(ordering_index, list):
            sum_ordering_index = sum(ordering_index)
            ordering_index = torch.Tensor(ordering_index).to(torch.int32)
            
        elif isinstance(ordering_index, torch.Tensor):
            assert ordering_index.dim() == 1
            ordering_index = ordering_index.to(torch.int32)
            sum_ordering_index = ordering_index.sum().item()
            
        else:
            raise TypeError("ordering_index should be provided in the type of list or torch.Tensor", type(ordering_index))

        if block_size is None:
            assert self.check_ordering_index(sum_ordering_index, self.hidden_dimension)
        else:
            assert (self.hidden_dimension % block_size) == 0
            assert self.check_ordering_index(sum_ordering_index, block_size)
            
        if self.reordering is None:
            del self._reordering
            
            self.register_buffer('_reordering', 
                                 ordering_index.detach().clone().to(
                                    dtype=torch.int32,
                                 ),
                                )
        else:
            self._reordering = ordering_index.detach().clone().to(dtype=torch.int32)

        self._reordering_block_size = block_size
        
    def check_ordering_index(self, sum_ordering_index:int, size:int):
        return sum_ordering_index == int((size*(size-1)) // 2)

    def construct(self):
        pass
        
# testcalib = OnlineCalibrator(8, on_dtype=torch.bfloat16, calib_dtype=torch.bfloat16)



