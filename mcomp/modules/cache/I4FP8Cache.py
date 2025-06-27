from transformers.cache_utils import (
Cache, DynamicCache, StaticCache, CacheConfig
)
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any, Union
import torch

from ..utils import (
  pack_int4_to_int32,
  unpack_int32_to_int4
)

from .base_cache import BaseQuantizedCache

@dataclass
class I4FP8CacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        nbits (`Optional[int]`, *optional*, defaults to 4):
        group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 128.
        scale_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        max_batch_size,
        max_cache_len,
        nbits: Optional[int] = 4,
        group_size: Optional[int] = 128,
        scale_dtype: Optional[torch.dtype] = torch.float8_e4m3fn,
        device: Optional[str] = "cpu",
    ):
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.nbits = nbits
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [4]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.group_size,
                ),
            )


class I4FP8QuantizedCache(BaseQuantizedCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    is_compileable = True #?! (on StaticCache)
    
    def __init__(self, config, cache_config: I4FP8CacheConfig, layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,) -> None:
        super().__init__()

        self.max_batch_size = cache_config.max_batch_size
        self.max_cache_len = cache_config.max_cache_len
        self.scale_dtype = cache_config.scale_dtype
        self.group_size = cache_config.group_size
        self.nbits = cache_config.nbits

        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        assert self.head_dim % self.group_size == 0
        
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        self._key_scale_cache: List[torch.Tensor] = []
        self._value_scale_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.group_size = cache_config.group_size
        self.scale_dtype = cache_config.scale_dtype
        self.device = cache_config.device

        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim // 8)
        scahe_cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim // self.group_size)
        device = torch.device(self.device) if device is not None else None
        
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=torch.int32, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=torch.int32, device=layer_device)
            
            new_layer_key_scale_cache = torch.zeros(scahe_cache_shape, dtype=self.scale_dtype, device=layer_device)
            new_layer_value_scale_cache = torch.zeros(scahe_cache_shape, dtype=self.scale_dtype, device=layer_device)
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            torch._dynamo.mark_static_address(new_layer_key_scale_cache)
            torch._dynamo.mark_static_address(new_layer_value_scale_cache)
            
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            self._key_scale_cache.append(new_layer_key_scale_cache)
            self._value_scale_cache.append(new_layer_value_scale_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        k_scale_out = self._key_scale_cache[layer_idx]
        v_scale_out = self._value_scale_cache[layer_idx]
                
        packed_ikey, ikey_scale = self._group_quantize_for_kv(key_states, self.nbits, self.group_size, scales_dt=self.scale_dtype)
        packed_ivalue, ivalue_scale = self._group_quantize_for_kv(value_states, self.nbits, self.group_size, scales_dt=self.scale_dtype)

        # packing

        if cache_position is None:
            k_out.copy_(packed_ikey)
            v_out.copy_(packed_ivalue)
            k_scale_out.copy_(ikey_scale)
            v_scale_out.copy_(ivalue_scale)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, packed_ikey)
                v_out.index_copy_(2, cache_position, packed_ivalue)
                k_scale_out.index_copy_(2, cache_position, ikey_scale)
                v_scale_out.index_copy_(2, cache_position, ivalue_scale)

                
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = packed_ikey
                v_out[:, :, cache_position] = packed_ivalue

        return (k_out, k_scale_out), (v_out, v_scale_out)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
            self._key_scale_cache[layer_idx].zero_()
            self._value_scale_cache[layer_idx].zero_()


    # def update(
    #     self,
    #     key_states: torch.Tensor,
    #     value_states: torch.Tensor,
    #     layer_idx: int,
    #     cache_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Update the number of seen tokens
    #     if layer_idx == 0:
    #         self._seen_tokens += key_states.shape[-2]

    #     if key_states is not None:
    #         if len(self.key_cache) <= layer_idx:
    #             for _ in range(len(self.key_cache), layer_idx+1):
    #                 self.key_cache.append(torch.tensor([]))
    #                 self.value_cache.append(torch.tensor([]))
    #                 self._key_scale_cache.append(torch.tensor([]))
    #                 self._value_scale_cache.append(torch.tensor([]))
    
    #         ikey, ikey_scale = self._group_quantize_for_kv(key_states, self.nbits, self.group_size, scales_dt=self.scale_dtype)
    
    #         ivalue, ivalue_scale = self._group_quantize_for_kv(value_states, self.nbits, self.group_size, scales_dt=self.scale_dtype)
            
    #         if (
    #             not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
    #         ):  # fills previously skipped layers; checking for tensor causes errors
    #             self.key_cache[layer_idx] = ikey
    #             self.value_cache[layer_idx] = ivalue
    #             self._key_scale_cache[layer_idx] = ikey_scale
    #             self._value_scale_cache[layer_idx] = ivalue_scale
    #         else:
    #             self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], ikey], dim=-2)
    #             self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], ivalue], dim=-2)
    #             self._key_scale_cache[layer_idx] = torch.cat([self._key_scale_cache[layer_idx], ikey_scale], dim=-2)
    #             self._value_scale_cache[layer_idx] = torch.cat([self._value_scale_cache[layer_idx], ivalue_scale], dim=-2)

    #     return (self.key_cache[layer_idx], self._key_scale_cache[layer_idx]), (self.value_cache[layer_idx], self._value_scale_cache[layer_idx])


    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1


    def _group_dequantize(self, cache_input, scales, position, on_dtype=None, out_dtype=None):
    
        group_size = cache_input.shape[-1] // scales.shape[-1]
        tg_cache_input = cache_input[:,:,:position,:].contiguous()
        tg_scales = cache_input[:,:,:position,:].contiguous()

        # unpack
        tg_cache_input = unpack_int32_to_int4(tg_cache_input)
        leading_shape = tg_cache_input.shape[:-1] # [B Head :position Dim_head]
    
        tg_cache_input = tg_cache_input.reshape(*leading_shape, -1, group_size)
        scales = scales.view(*leading_shape,-1,1)
    
        if on_dtype is not None:
            cache_input = cache_input.to(on_dtype)
            scales = scales.to(on_dtype)

        if group_size == self.head_dim:
            deq_cache_input = cache_input*scales
        else:
            scales = scales.repeat_interleave(group_size, dim=-1)
            deq_cache_input = cache_input*scales
    
        if out_dtype is not None:
            deq_cache_input = deq_cache_input.to(out_dtype)
    
        return deq_cache_input.reshape(*leading_shape, -1)


    def _group_quantize_for_kv(self, cache_input, n_bit, group_size, scales_dt=None): # symmetric quant
        # cache_input: [batch, head_size, seq, head_dim]
        if scales_dt is None:
            scales_dt = self.scale_dtype
        leading_shape = cache_input.shape[:-1]
        
        cache_input = cache_input.reshape(*leading_shape, -1, group_size)
        max_val = cache_input.abs().amax(dim=-1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        
        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(cache_input).sum() == 0
        
        icache_input = torch.clamp(torch.round(cache_input/scales), min_int, max_int)
        scales = scales.to(scales_dt)
        
        icache_input = icache_input.reshape(*leading_shape, -1).to(torch.int8)
        scales = scales.view(*leading_shape, -1)

        packed_input = pack_int4_to_int32(icache_input)
    
        return packed_input, scales


