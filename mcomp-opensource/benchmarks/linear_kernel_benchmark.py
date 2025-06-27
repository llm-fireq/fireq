import torch
import time
import argparse
import numpy as np
import pprint
from mcomp.utils import timing_utils
from mcomp.fused.layers import (
    INT4FP8Linear, 
    INT4FP8LinearMul, 
    INT4FP8LinearSiLU, 
    INT4FP8LinearAdd,
    # INT4FP8LinearSquareDot,
    INT4BF16Linear, 
    INT4BF16LinearMul, 
    INT4BF16LinearSiLU, 
    INT4BF16LinearAdd,
    # INT4BF16LinearSquareDot,
)


mlp_sizes = [
    (4096, 14336), #llama3-8b
]

num_warmup_steps = 5
num_bench_steps = 100

def module_benchmark(module, *args):
    # warmup
    for i in range(num_warmup_steps):
        out = module(*args[1:])
    
    timing_utils.Timer.change_setup(1)
    timing_utils.Timer.tick(args[0])
    for i in range(num_bench_steps):
        out = module(*args[1:])
    timing_utils.Timer.tock(args[0])

def linearint4fp8_benchmark(args):
    bsz = args.bsz
    seq_len = args.seq_len
    group_size = 128
    
    
    layer_size = mlp_sizes

    for (feature_dim_in, feature_dim_out) in layer_size:
        x = torch.rand((bsz, seq_len, feature_dim_in)).cuda().to(torch.float8_e4m3fn).contiguous()
        x_bf16 = torch.rand((bsz, seq_len, feature_dim_in)).cuda().to(torch.bfloat16).contiguous()
        x_scale = torch.rand((bsz, seq_len)).cuda().to(torch.bfloat16).contiguous()
        residual = torch.rand((bsz, seq_len, feature_dim_out)).cuda().to(torch.bfloat16).contiguous()
        # square_sum = torch.rand((bsz, seq_len)).cuda().to(torch.bfloat16).contiguous()

        weight = torch.rand((feature_dim_out, feature_dim_in // 2)).cuda().to(torch.int8).contiguous()
        weight_scale = torch.rand((feature_dim_out, (feature_dim_in + 127) // group_size)).cuda().to(torch.float8_e4m3fn).contiguous()
        weight_scale_bf16 = torch.rand((feature_dim_out, (feature_dim_in + 127) // group_size)).cuda().to(torch.bfloat16).contiguous()

        int4fp8_linear = INT4FP8Linear(weight, weight_scale, feature_dim_in, feature_dim_out, group_size)
        int4fp8_linear_mul = INT4FP8LinearMul(weight, weight_scale, feature_dim_in, feature_dim_out, group_size)
        int4fp8_linear_add = INT4FP8LinearAdd(weight, weight_scale, feature_dim_in, feature_dim_out, group_size)
        int4fp8_linear_silu = INT4FP8LinearSiLU(weight, weight_scale, feature_dim_in, feature_dim_out, group_size)
        # int4fp8_linear_squaredot = INT4FP8LinearSquareDot(weight, weight_scale, feature_dim_in, feature_dim_out, group_size)

        int4bf16_linear = INT4BF16Linear(weight, weight_scale_bf16, feature_dim_in, feature_dim_out, group_size)
        int4bf16_linear_mul = INT4BF16LinearMul(weight, weight_scale_bf16, feature_dim_in, feature_dim_out, group_size)
        int4bf16_linear_add = INT4BF16LinearAdd(weight, weight_scale_bf16, feature_dim_in, feature_dim_out, group_size)
        int4bf16_linear_silu = INT4BF16LinearSiLU(weight, weight_scale_bf16, feature_dim_in, feature_dim_out, group_size)
        # int4bf16_linear_squaredot = INT4BF16LinearSquareDot(weight, weight_scale_bf16, feature_dim_in, feature_dim_out, group_size)

    
        print(f"Weight Sizes: {feature_dim_out, feature_dim_in}")

        module_benchmark(int4fp8_linear.forward, "linear", x, x_scale, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear")
        print(f"INT4FP8Linear time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4fp8_linear_mul.forward, "linear_mul", x, x_scale, residual, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_mul")
        print(f"INT4FP8LinearMul time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4fp8_linear_add.forward, "linear_add", x, x_scale, residual, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_add")
        print(f"INT4FP8LinearAdd time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4fp8_linear_silu.forward, "linear_silu", x, x_scale, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_silu")
        print(f"INT4FP8LinearSiLU time: {np.mean(avg_time/ num_bench_steps):.3f}ms")

        # module_benchmark(int4fp8_linear_squaredot.forward, "linear_squaredot", x, x_scale, x.clone(), square_sum, bsz, seq_len)
        # avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_squaredot")
        # print(f"INT4FP8LinearSquareDot time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4bf16_linear.forward, "linear_bf16", x_bf16, x_scale, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_bf16")
        print(f"INT4BF16Linear time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4bf16_linear_mul.forward, "linear_mul_bf16", x_bf16, x_scale, residual, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_mul_bf16")
        print(f"INT4BF16LinearMul time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4bf16_linear_add.forward, "linear_add_bf16", x_bf16, x_scale, residual, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_add_bf16")
        print(f"INT4BF16Linear time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        module_benchmark(int4bf16_linear_silu.forward, "linear_silu_bf16", x_bf16, x_scale, bsz, seq_len)
        avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_silu_bf16")
        print(f"INT4BF16LinearSiLU time: {np.mean(avg_time / num_bench_steps):.3f}ms")

        # module_benchmark(int4bf16_linear_squaredot.forward, "linear_squaredot_bf16", x_bf16, x_scale, residual, square_sum, bsz, seq_len)
        # avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("linear_squaredot_bf16")
        # print(f"INT4BF16LinearSquareDot time: {np.mean(avg_time / num_bench_steps):.3f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=16,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=1024,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linearint4fp8_benchmark(args)