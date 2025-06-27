import argparse
import pprint
import numpy as np
import torch

from mcomp.utils import timing_utils
from mcomp.fused.layers import FP8FP8FmhaFwd

num_warmup_steps = 5
num_bench_steps = 100

def module_benchmark(module, *args):
    # warmup
    for i in range(num_warmup_steps):
        out = module(*args[1:])
    
    print("\tmodule_benchmark: Warm-up finish!")
    timing_utils.Timer.change_setup(1)
    timing_utils.Timer.tick(args[0])
    for i in range(num_bench_steps):
        out = module(*args[1:])
    timing_utils.Timer.tock(args[0])



def qattention_benchmark(args):
    bsz = args.batch_size
    seq_len = args.seq_len
    head_group = args.head_group
    num_heads = args.num_heads
    head_dim = args.head_dim

    q = torch.rand((bsz, seq_len, head_group * num_heads, head_dim)).cuda().to(torch.float8_e4m3fn).contiguous()
    k = torch.rand((bsz, seq_len, num_heads, head_dim)).cuda().to(torch.float8_e4m3fn).contiguous()
    v = torch.rand((bsz, seq_len, num_heads, head_dim)).cuda().to(torch.float8_e4m3fn).contiguous()
    fp8fp8fmha = FP8FP8FmhaFwd(head_group, num_heads, head_dim)

    module_benchmark(fp8fp8fmha.forward, "attention", q, k, v, bsz, seq_len)
    avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("attention")
    avg_time = avg_time / num_bench_steps
    print(f"FP8FP8FmhaFwd time: {np.mean(avg_time):.3f}ms")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--head_group', type=int,
        help='Head group',
        default=4,
    )

    parser.add_argument(
        '--num_heads', type=int,
        help='Number of attention heads',
        default=8,
    )

    parser.add_argument(
        '--head_dim', type=int,
        help='Head size',
        default=128,
    )

    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    qattention_benchmark(args)