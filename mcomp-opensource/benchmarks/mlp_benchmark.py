import os
import torch
import pprint
import argparse
import tqdm

from mcomp.quant.fireQLlamaQuantizer import FireQLlamaQuantizer
from mcomp.fused.modules.models.llama.fuser import LlamaFuser
from mcomp.utils import timing_utils

from benchmarks.utils.mockup_mlp_llama2 import MLPINT4FP8Llama2


os.environ["CUDA_VISIBLE_DEVICES"] = '4'
num_warmup_steps = 5
num_bench_steps = 10

def module_benchmark(module, *args):
    # Warm-up runs
    for i in range(num_warmup_steps):
        out = module(*args)

    print("\tmodule_benchmark: Warm-up finish!")
    # Set up timing utils for measurements
    timing_utils.Timer.change_setup(1)
    timing_utils.Timer.tick("mlp")
    for i in tqdm.tqdm(range(num_bench_steps)):
        out = module(*args)
    timing_utils.Timer.tock("mlp")
    

def build_mockup_mlp_llama2(hidden_size, intermediate_size, is_bf16, group_size = 128):
    print(f"Mockup Llama2 MLP hidden_size: {hidden_size}, intermediate_size: {intermediate_size}")
    model = MLPINT4FP8Llama2(hidden_size, intermediate_size, is_bf16, group_size)
    return model

def build_mlp_llama3(model):
    mlp = model.model.layers[-1].mlp
    return mlp
    

def mlp_benchmark(args):
    model_name = args.model
    bsz = args.bsz
    seq_len = args.seq_len
    is_bf16 = args.is_bf16
        

    print(f"------------------------------------Run benchmark {model_name} --------------------------------------")
    if any(x in model_name.lower() for x in ["llama2-7b", "llama-2-7b", "llama-7b"]):
        hidden_size = 4096
        intermediate_size = 11008
        model = build_mockup_mlp_llama2(hidden_size, intermediate_size, is_bf16)
    elif any(x in model_name.lower() for x in ["llama2-13b", "llama-2-13b", "llama-13b"]):
        hidden_size = 5120
        intermediate_size = 13824
        model = build_mockup_mlp_llama2(hidden_size, intermediate_size, is_bf16)
    elif any(x in model_name.lower() for x in ["llama-30b"]):
        hidden_size = 6656
        intermediate_size = 17920
        model = build_mockup_mlp_llama2(hidden_size, intermediate_size, is_bf16)
    else:
        print(f"\tModel load start {model_name}")
        quantizer, model, tokenizer = FireQLlamaQuantizer.load_quantized(model_name)
        fuser = LlamaFuser(model, quantizer.config)
        fuser.fuse()
        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        model = build_mlp_llama3(model)

    print("\tModel load completed.")
    model.cuda()

    
    print(f"\tbsz: {bsz} seq_len: {seq_len}")
    if is_bf16:
        test_input = torch.randn((bsz, seq_len, hidden_size)).to(torch.bfloat16).cuda().contiguous()
        residual = torch.randn((bsz, seq_len, hidden_size)).to(torch.bfloat16).cuda().contiguous()
    else:
        test_input = torch.randn((bsz, seq_len, hidden_size)).to(torch.float8_e4m3fn).cuda().contiguous()
        residual = torch.randn((bsz, seq_len, hidden_size)).to(torch.bfloat16).cuda().contiguous()
        
    # Run benchmark
    module_benchmark(model, test_input, residual)

    avg_time, _ = timing_utils.Timer.get_avg_elapsed_time_ms("mlp")
    avg_time = avg_time / num_bench_steps
    if not avg_time:
        print(f"\tMLP time is not measured.")
    else:
        print(f"\tMLP Average time: {avg_time:.3f}ms")
        print(f"\tMLP Throughput: {(bsz * seq_len / avg_time): .3f}tok/s")
    
    timing_utils.Timer.clear_time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='Benchmark_model',
        default='llama3-8B-i4f8-pre-post-chwise',
        # default='llama3-8B-i4bf16fp8-linear-kv-rope-chwise',
    )

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size for benchmarks',
        default = 16,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Input sequence length list for benchmakrs',
        default=1024,
    )
    parser.add_argument(
        '--is_bf16', action="store_true",
        help='Whether run I4BF16 linear kernels',
    )

    args = parser.parse_args()
    pprint.pprint(vars(args))
    mlp_benchmark(args)

    
    

