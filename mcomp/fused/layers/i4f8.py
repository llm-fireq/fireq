import torch

from mcomp.fused import (
    _CUTLASS_INT4FP8Linear,
    _CUTLASS_INT4FP8LinearAdd,
    _CUTLASS_INT4FP8LinearMul,
    _CUTLASS_INT4FP8LinearSiLU,
    _CUTLASS_INT4FP8LinearSquareDot,
    _CUTLASS_INT4BF16Linear,
    _CUTLASS_INT4BF16LinearAdd,
    _CUTLASS_INT4BF16LinearMul,
    _CUTLASS_INT4BF16LinearSiLU,
    _CUTLASS_INT4BF16LinearSquareDot,
    _CUTLASS_FP8FP8FmhaFwd
)

class INT4FP8Linear:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4FP8Linear.INT4FP8Linear(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, outputs, 
            batch_size, seq_len)
        return outputs

class INT4FP8LinearAdd:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4FP8LinearAdd.INT4FP8LinearAdd(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, batch_size, seq_len)
        return outputs

class INT4FP8LinearMul:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4FP8LinearMul.INT4FP8LinearMul(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, batch_size, seq_len)
        return outputs
    

class INT4FP8LinearSiLU:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4FP8LinearSiLU.INT4FP8LinearSiLU(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, outputs, batch_size, seq_len)
        return outputs
    
class INT4FP8LinearSquareDot:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4FP8LinearSquareDot.INT4FP8LinearSquareDot(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                 square_sum: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, square_sum, batch_size, seq_len)
        return outputs
    
class INT4BF16Linear:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4BF16Linear.INT4BF16Linear(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, outputs, 
            batch_size, seq_len)
        return outputs

class INT4BF16LinearAdd:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4BF16LinearAdd.INT4BF16LinearAdd(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, batch_size, seq_len)
        return outputs

class INT4BF16LinearMul:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4BF16LinearMul.INT4BF16LinearMul(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, batch_size, seq_len)
        return outputs
    

class INT4BF16LinearSiLU:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4BF16LinearSiLU.INT4BF16LinearSiLU(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, outputs, batch_size, seq_len)
        return outputs
    
class INT4BF16LinearSquareDot:
    def __init__(self, weights: torch.Tensor, 
                 weight_scales: torch.Tensor, 
                 input_dim: int, 
                 output_dim: int, 
                 group_size: int):
        
        self.kernel = _CUTLASS_INT4BF16LinearSquareDot.INT4BF16LinearSquareDot(weights, weight_scales, input_dim, output_dim, group_size)
        self.output_dim = output_dim

    @torch.no_grad    
    def forward(self, inputs: torch.Tensor,
                 input_scales: torch.Tensor,
                 residuals: torch.Tensor,
                 square_sum: torch.Tensor,
                   batch_size, seq_len) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.output_dim, dtype=torch.bfloat16, device=inputs.device)
        self.kernel.forward(inputs, input_scales, residuals, outputs, square_sum, batch_size, seq_len)
        return outputs
    
class FP8FP8FmhaFwd:
    def __init__(self, head_group: int, 
                 num_heads: int, 
                 head_size: int):
        self.kernel = _CUTLASS_FP8FP8FmhaFwd.FP8FP8FmhaFwd(head_group, num_heads, head_size)
        self.hidden_size = head_group * num_heads * head_size
    
    @torch.no_grad
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                batch_size: int,
                seq_len: int) -> torch.Tensor:
        outputs = torch.empty(batch_size, seq_len, self.hidden_size, dtype=torch.bfloat16, device=query.device)
        self.kernel.forward(query, key, value, outputs, batch_size, seq_len)
        return outputs