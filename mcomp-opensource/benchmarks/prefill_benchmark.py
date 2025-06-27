import os
import time
import torch
import pprint
import argparse
import tqdm
import numpy as np

from mcomp.quant.fireQLlamaQuantizer import FireQLlamaQuantizer
from mcomp.fused.modules.models.llama.fuser import LlamaFuser
from mcomp.utils import timing_utils

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
num_warmup_steps = 5
num_bench_steps = 10

def module_benchmark(module, *args):
    # Warm-up runs
    for i in tqdm.tqdm(range(num_warmup_steps)):
        out = module(*args)
    
    print("\tWarm-up finish!")
    # Set up timing utils for measurements
    timing_utils.Timer.change_setup(1)
    timing_utils.Timer.tick("prefill")
    for i in tqdm.tqdm(range(num_bench_steps)):
        out = module(*args)
    timing_utils.Timer.tock("prefill")
    

def benchmark(args):
    model_name = args.model
    bsz = args.bsz
    seq_len = args.seq_len
    print(f"------------------------------------Run benchmark {model_name} --------------------------------------")

    print(f"\tModel load start {model_name}")
    quantizer, model, tokenizer = FireQLlamaQuantizer.load_quantized(model_name)
    fuser = LlamaFuser(model, quantizer.config)
    fuser.fuse()
    print("\tModel load completed")
    
    model = model.cuda()
    test_input = torch.randint(100, 200, (bsz, seq_len), dtype=torch.int32).cuda()
    
    print("\tRun benchmark")
    module_benchmark(model, test_input)
    avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("prefill")
    avg_time = avg_time / num_bench_steps
    print(f"\tPrefill Average time: {avg_time:.3f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='Benchmark model',
        default='llama3-8B-i4f8-pre-post-chwise',
    )

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
    benchmark(args)
