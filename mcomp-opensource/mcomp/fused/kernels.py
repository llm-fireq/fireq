

class INT4FP8Linear:
    def __init__(self, weights, weight_scales, input_dim, output_dim, group_size):
        self.kernel = _CUDA_INT4FP8Linear.INT4FP8(weights, weight_scales, input_dim, output_dim, group_size)
        
    def forward(self, x):
        self.kernel.forward()