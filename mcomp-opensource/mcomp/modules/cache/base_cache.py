from abc import ABC, abstractmethod

from transformers.cache_utils import (
Cache, DynamicCache, StaticCache, CacheConfig
)


class BaseQuantizedCache(Cache, ABC):

    @abstractmethod
    def _group_dequantize(self, cache_input, scales):
        raise NotImplementedError


    def _group_quantize_for_kv(self, cache_input): # symmetric quant
        raise NotImplementedError
    
