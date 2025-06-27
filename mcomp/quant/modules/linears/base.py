import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Union, List, Tuple, Dict
import gc
from abc import abstractmethod, ABC
from mcomp.quant.mixin import WeightClippingMixin
from mcomp.utils import (
    get_self_module_name,
)

class BaseCalibratedModule(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def construct(self):
        raise NotImplementedError

        
class BaseCalbiratedLinear(BaseCalibratedModule, WeightClippingMixin, ABC):

    weight_int_quantized = None
    
    def __init__(self, linear:nn.Linear, calib_dtype:torch.dtype=torch.float32):
        super().__init__()
        self.module = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.calib_dtype = calib_dtype
        
        self.scales = None
        self.quantized = False

        self.orig_w_dtype = linear.weight.dtype
        self.org_w_device = linear.weight.device
        self.on_dtype = linear.weight.dtype
        
        self.bit_width = None
        self.group_size = None
        self.weight_scale_dtype = None
        
        self.pqweight = nn.Parameter(
            linear.weight.detach().clone().to(
                dtype=self.orig_w_dtype,
                device=self.org_w_device
            ),
        )

        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().clone().to(
                    dtype=self.orig_w_dtype,
                    device=self.org_w_device
                ), 
            )

        self._smooth_in = None # activation_side:     S_in * W^T
        self._smooth_out = None # output_side:        W^T * S_out
        self._rotation_in = None # activation_side:   R_in * W^T   # buffer?!
        self._rotation_out = None # output_side:      W^T * R_out  # buffer?!
        self._reordering_in = None # activation_side: P_in * W^T   # buffer?!
        self._reordering_out = None # output_side:    W^T * P_out  # buffer?!

        self._rotation_in_block_size = None
        self._rotation_out_block_size = None
        self._reordering_in_block_size = None
        self._reordering_out_block_size = None
        
        self.smooth_in_priority = -1
        self.reordering_in_priority = 0
        self.rotation_in_priority = 1

        self.smooth_out_priority = -1
        self.reordering_out_priority = 0
        self.rotation_out_priority = 1

    def set_in_priorities(self, smooth:int=-1, reordering:int=0, rotation:int=1):
        self.smooth_in_priority = smooth
        self.reordering_in_priority = reordering
        self.rotation_in_priority = rotation

    def set_out_priorities(self, smooth:int=-1, reordering:int=0, rotation:int=1):
        self.smooth_out_priority = smooth
        self.reordering_out_priority = reordering
        self.rotation_out_priority = rotation
    
    @property 
    def smooth_in(self):
        return self._smooth_in

    @smooth_in.setter
    def smooth_in(self, smoothing_factor:torch.Tensor):
        if smoothing_factor is None:
            if self._smooth_in is not None:
                del self._smooth_in
                torch.cuda.empty_cache()
            self._smooth_in = None
            return
            
        assert isinstance(smoothing_factor, torch.Tensor)
        assert smoothing_factor.dim() in [1, 2]
        dim = smoothing_factor.dim()

        if dim == 2:
            h,w = smoothing_factor.shape
            assert h*w == max(h,w), "For dim==2 smoothing factor, an axis should be one, vector only"
            smoothing_factor = smoothing_factor.view(-1)
        
        sf_size = smoothing_factor.numel()
        assert sf_size == self.in_features

        if self.smooth_in is None:
            del self._smooth_in
            
            self.register_buffer('_smooth_in', 
                                 smoothing_factor.detach().clone().to(
                                    dtype=self.orig_w_dtype,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._smooth_in = smoothing_factor.detach().clone().to(dtype=self.orig_w_dtype, device=self.org_w_device)

    def apply_smooth_in(self):
        if self.smooth_in is None:
            return
        
        orig_weight_dtype = self.pqweight.data.dtype
        w_device = self.pqweight.data.device

        self.pqweight.data = self.pqweight.data.to(torch.float64) * self.smooth_in.to(torch.float64)
        
        self.pqweight.data = self.pqweight.data.to(orig_weight_dtype)
        
    
    @property 
    def smooth_out(self):
        return self._smooth_out

    @smooth_out.setter
    def smooth_out(self, smoothing_factor:torch.Tensor):
        if smoothing_factor is None:
            if self._smooth_out is not None:
                del self._smooth_out
                torch.cuda.empty_cache()
            self._smooth_out = None
            return
            
        assert isinstance(smoothing_factor, torch.Tensor)
        assert smoothing_factor.dim() in [1, 2]
        dim = smoothing_factor.dim()

        if dim == 2:
            h,w = smoothing_factor.shape
            assert h*w == max(h,w), "For dim==2 smoothing factor, an axis should be one, vector only"
            smoothing_factor = smoothing_factor.view(-1)
        
        sf_size = smoothing_factor.numel()
        assert sf_size == self.out_features

        if self.smooth_out is None:
            del self._smooth_out
            
            self.register_buffer('_smooth_out', 
                                 smoothing_factor.detach().clone().to(
                                    dtype=self.orig_w_dtype,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._smooth_out = smoothing_factor.detach().clone().to(dtype=self.orig_w_dtype, device=self.org_w_device)

    def apply_smooth_out(self):
        if self.smooth_out is None:
            return
        
        orig_weight_dtype = self.pqweight.data.dtype
        w_device = self.pqweight.data.device
        
        self.pqweight.data = self.pqweight.data.to(torch.float64) * (self.smooth_out.view(-1,1).to(torch.float64))
        
        self.pqweight.data = self.pqweight.data.to(orig_weight_dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.float64) * self.smooth_out.to(torch.float64)
            self.bias.data = self.bias.data.to(orig_weight_dtype)
            
    @property 
    def rotation_in(self):
        return self._rotation_in

    @rotation_in.setter
    def rotation_in(self, rotation:Tuple[torch.Tensor,int]):
        rotation_matrix, block_size = rotation
        if rotation_matrix is None:
            if self._rotation_in is not None:
                del self._rotation_in
                torch.cuda.empty_cache()
            self._rotation_in = None, block_size
            #self._rotation_in_block_size = block_size
            return

        assert isinstance(rotation_matrix, torch.Tensor)
        assert rotation_matrix.dim() == 2

        h, w = rotation_matrix.shape
        assert h == w

        if block_size is None:
            assert self.in_features == h
        else:
            assert (self.in_features % block_size) == 0
            #assert block_size == h
        
        if self.rotation_in is None:
            del self._rotation_in
            
            self.register_buffer('_rotation_in', 
                                 rotation_matrix.detach().clone().to(
                                    #dtype=self.orig_w_dtype,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._rotation_in = rotation_matrix.detach().clone().to(
                #dtype=self.orig_w_dtype, 
                device=self.org_w_device) 

        self._rotation_in_block_size = block_size

    def apply_rotation_in(self):
        if self.rotation_in is None:
            return

        orig_weight_dtype = self.pqweight.data.dtype
        w_device = self.pqweight.data.device
        orig_rotation_dtype = self.rotation_in
        
        self.pqweight.data = self.pqweight.data.to(torch.float64)
        self.rotation_in = self.rotation_in.to(torch.float64), self._rotation_in_block_size
        
        rotation_t = self.rotation_in.T.contiguous().to(w_device)
        
        if self._rotation_in_block_size is None:
            self.pqweight.data = torch.matmul(self.pqweight.data, rotation_t).contiguous()    
        else:
            out_dimension, in_dimension = self.pqweight.data.shape
            rot_kron = torch.kron(torch.eye(in_dimension//self._rotation_in_block_size).to(w_device), rotation_t).to(torch.float64).to(w_device)
            self.pqweight.data = (self.pqweight.data @ rot_kron).contiguous()
            # weight_shape = self.pqweight.data.shape
            # tmp_weight = self.pqweight.data.T.contiguous().view(-1,self._rotation_in_block_size,self.out_features).transpose(1,2)
            
            # tmp_weight = torch.matmul(tmp_weight, self.rotation_in)
            # self.pqweight.data = tmp_weight.transpose(0,1).reshape(weight_shape)

        self.pqweight.data = self.pqweight.data.to(orig_weight_dtype)
        self.rotation_in = self.rotation_in.to(orig_rotation_dtype), self._rotation_in_block_size
            
    @property 
    def rotation_out(self):
        return self._rotation_out

    @rotation_out.setter
    def rotation_out(self, rotation:Tuple[torch.Tensor,int]):
        
        rotation_matrix, block_size = rotation
        if rotation_matrix is None:
            if self._rotation_out is not None:
                del self._rotation_out
                torch.cuda.empty_cache()
            self._rotation_out = None
            self._rotation_out_block_size = block_size
            return
        
        assert isinstance(rotation_matrix, torch.Tensor)
        assert rotation_matrix.dim() == 2

        h, w = rotation_matrix.shape
        assert h == w

        if block_size is None:
            assert self.out_features == h
        else:
            assert (self.in_features % block_size) == 0
            #assert block_size == h
        
        if self.rotation_out is None:
            del self._rotation_out
            
            self.register_buffer('_rotation_out', 
                                 rotation_matrix.detach().clone().to(
                                    #dtype=self.orig_w_dtype,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._rotation_out = rotation_matrix.detach().clone().to(
                                                                    #dtype=self.orig_w_dtype, 
                                                                     device=self.org_w_device)

        self._rotation_out_block_size = block_size

    def apply_rotation_out(self):
        if self.rotation_out is None:
            return

        orig_weight_dtype = self.pqweight.data.dtype
        w_device = self.pqweight.data.device
        orig_rotation_dtype = self.rotation_out
        
        self.pqweight.data = self.pqweight.data.to(torch.float64)
        self.rotation_out = self.rotation_out.to(torch.float64), self._rotation_out_block_size
        rotation_t = self.rotation_out.T.contiguous().to(w_device)

        if self._rotation_out_block_size is None:
            self.pqweight.data = torch.matmul(rotation_t, self.pqweight.data).contiguous()  
            if self.bias is not None:
                self.bias.data = (self.bias.data @ self.rotation_out).contiguous()
        else:
            
            out_dimension, in_dimension = self.pqweight.data.shape
            
            rot_kron_t = torch.kron(torch.eye(out_dimension//self._rotation_out_block_size).to(w_device), rotation_t).to(torch.float64).to(w_device)
            rot_kron = torch.kron(torch.eye(out_dimension//self._rotation_out_block_size).to(w_device), self.rotation_out).to(torch.float64).to(w_device)
            self.pqweight.data = torch.matmul(rot_kron_t, self.pqweight.data).contiguous()  
            if self.bias is not None:
                self.bias.data = (self.bias.data @ rot_kron).contiguous()
            # weight_shape = self.pqweight.data.shape
            
            # tmp_weight = self.pqweight.data.reshape(-1, self._rotation_out_block_size, self.in_features)
            # self.pqweight.data = torch.matmul(self.rotation_out, tmp_weight).reshape(weight_shape)
            # if self.bias is not None:
            #     self.bias.data = (self.bias.data.view(-1, self._rotation_out_block_size) @ self.rotation_out).view(-1)

        self.pqweight.data = self.pqweight.data.to(orig_weight_dtype)
        self.rotation_out = self.rotation_out.to(orig_rotation_dtype), self._rotation_out_block_size
              
    @property 
    def reordering_in(self):
        return self._reordering_in

    @reordering_in.setter
    def reordering_in(self, ordering_index:Tuple[Union[List, torch.Tensor], int]):
        
        ordering_index, block_size = ordering_index
        if ordering_index is None:
            if self._reordering_in is not None:
                del self._reordering_in
                torch.cuda.empty_cache()
            self._reordering_in = None, None
            self._reordering_in_block_size = block_size
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
            assert self.check_ordering_index(sum_ordering_index, self.in_features)
        else:
            assert (self.in_features % block_size) == 0
            assert self.check_ordering_index(sum_ordering_index, block_size)
            
        if self.reordering_in is None:
            del self._reordering_in
            
            self.register_buffer('_reordering_in', 
                                 ordering_index.detach().clone().to(
                                    dtype=torch.int32,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._reordering_in = ordering_index.detach().clone().to(dtype=torch.int32, device=self.org_w_device)

        self._reordering_in_block_size = block_size

    def apply_reordering_in(self):
        if self.reordering_in is None:
            return

        if self._reordering_in_block_size is None:
            self.pqweight.data = self.pqweight.data[:,self.reordering_in].contiguous() 
        else:

            total_reordering_index = torch.arange(self.in_features)
            total_reordering_index = total_reordering_index.view(-1, self._reordering_in_block_size)[:,self.reordering_in].view(-1)

            self.pqweight.data = self.pqweight.data[:,total_reordering_index].contiguous() 

    @property 
    def reordering_out(self):
        return self._reordering_out

    @reordering_out.setter
    def reordering_out(self, ordering_index:Tuple[Union[List, torch.Tensor], int]):
        
        ordering_index, block_size = ordering_index
        if ordering_index is None:
            if self._reordering_out is not None:
                del self._reordering_out
                torch.cuda.empty_cache()
            self._reordering_out = None, None
            self._reordering_out_block_size = block_size
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
            assert self.check_ordering_index(sum_ordering_index, self.out_features)
        else:
            assert (self.out_features % block_size) == 0
            assert self.check_ordering_index(sum_ordering_index, block_size)
            
        if self.reordering_out is None:
            del self._reordering_out
            
            self.register_buffer('_reordering_out', 
                                 ordering_index.detach().clone().to(
                                    dtype=torch.int32,
                                    device=self.org_w_device
                                 ),
                                )
        else:
            self._reordering_out = ordering_index.detach().clone().to(dtype=torch.int32, device=self.org_w_device)

        self._reordering_out_block_size = block_size

    def apply_reordering_out(self):
        if self.reordering_out is None:
            return

        if self._reordering_out_block_size is None:
            self.pqweight.data = self.pqweight.data[self.reordering_out,:].contiguous() 
            if self.bias is not None:
                self.bias.data = self.bias.data[self.reordering_out].contiguous()
        else:
            total_reordering_index = torch.arange(self.out_features)
            total_reordering_index = total_reordering_index.view(-1, self._reordering_out_block_size)[:,self.reordering_out].view(-1)

            self.pqweight.data = self.pqweight.data[total_reordering_index,:].contiguous() 

            if self.bias is not None:
                self.bias.data = (self.bias.data.view(-1, self._reordering_out_block_size)[:,self.reordering_out]).view(-1).contiguous()
        
    def check_ordering_index(self, sum_ordering_index:int, size:int):
        return sum_ordering_index == int((size*(size-1)) // 2)
            
    def init_pqweight(self):

        # TODO CHECK Is it possible?! buffer to None?!
        # TODO CHECK Is it possible to change buffer size?!
        # self.smooth_in = None # activation_side:     W * S_in
        # self.smooth_out = None # output_side:        S_out * W
        # self.rotation_in = None # activation_side:   W * R_in   # buffer?!
        # self.rotation_out = None # output_side:      R_out * W  # buffer?!
        # self.reordering_in = None # activation_side: W  P_in   # buffer?!
        # self.reordering_out = None # output_side:    P_out W  # buffer?!

        self.pqweight.data = self.module.weight.data
        if self.bias is not None:
            self.bias.data = self.module.bias.data
        gc.collect()
        torch.cuda.empty_cache()
        
    
    def construct(self, module=None, **kwargs):

        self.init_pqweight()

        in_apply_list = list(zip( [self.apply_smooth_in, 
                                   self.apply_rotation_in, 
                                   self.apply_reordering_in], [self.smooth_in_priority,
                                                              self.rotation_in_priority,
                                                              self.reordering_in_priority] ))
        out_apply_list = list(zip( [self.apply_smooth_out, 
                                   self.apply_rotation_out, 
                                   self.apply_reordering_out], [self.smooth_out_priority,
                                                              self.rotation_out_priority,
                                                              self.reordering_out_priority] ))

        in_apply_list.sort(key=lambda x: x[1])
        out_apply_list.sort(key=lambda x: x[1])

        ## apply_in
        for fcn, _ in in_apply_list:
            fcn()

        ## apply_out
        for fcn, _ in out_apply_list:
            fcn()
            
        if module is not None:
            weight_clipping = kwargs.get('weight_clipping', None)
            if weight_clipping is not None and weight_clipping:
                if self.weight_int_quantized:
                    self.apply_weight_clipping(module, **kwargs)
        
        self.apply_pseudo_quantize()

    def apply_weight_clipping(self, module, **kwargs):
        
        # get self module name
        # 
        no_wc_layers = kwargs.get('no_wc_layers', None)
        if no_wc_layers is None:
            no_wc_layers = []
        
        self_name = get_self_module_name(module, self)
        wc_flag = False
        for key in kwargs.keys():
            if key.endswith(self_name) and self_name not in no_wc_layers:
                wc_flag = True
                self_name_key = key
                break
        
        if wc_flag:
            
            max_sample_num = kwargs.get("max_sample_num", None)
            num_steps = kwargs.get("num_steps", 10)
            w_batch_size = kwargs.get("w_batch_size", 64)
            x_batch_size = kwargs.get("x_batch_size", 1024)
            device = kwargs.get("device", None)
            scales_dt = self.weight_scale_dtype
            nbits = self.bit_width
            group_size = self.group_size
            
            assert scales_dt is not None
            assert nbits is not None
            assert group_size is not None
            assert device is not None
            x = kwargs.get(self_name_key, None)
            assert x is not None
            
            print(f"Calculating and Applying best_clip for {self_name}")
            best_clip = self.get_best_clip(
                self.pqweight.data,
                x,
                nbits,
                group_size,
                scales_dt,
                device,
                w_batch_size,
                x_batch_size,
                num_steps,
                max_sample_num, 
            )
            self.pqweight.data = self.apply_best_clip(
                self.pqweight.data,
                best_clip,
            )

    @abstractmethod
    def apply_pseudo_quantize(self):

        raise NotImplementedError
