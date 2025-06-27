import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attn import (
  FusedLlamaAttentionI4FP8withLN,
)

from .mlp import (
  FusedLlamaMLPI4FP8withLN,
)

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import Cache

from mcomp.utils import timing_utils

class FusedLlamaDecoderLayerLNImported(nn.Module):
    def __init__(self, config: LlamaConfig, attn_with_ln, mlp_with_ln):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = attn_with_ln
        self.mlp = mlp_with_ln
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

