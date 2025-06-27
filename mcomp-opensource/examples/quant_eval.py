import argparse
import yaml
import os
import lm_eval
import json
from lm_eval.models import huggingface
from lm_eval.utils import (
    handle_non_serializable
)
import urllib3
urllib3.disable_warnings()

from mcomp.quant.modules.cache import PseudoQuantStaticCache

import torch
from test_utils import (
    TestConfig,
    get_config,
    get_kvcache_layer_device_map_from_model_and_attn_name,
    get_kv_wraped_model,
    eval_ppl_with_gptq_evaluator,
    get_base_path,
    quantize,
    load_model,
    save_test_config,
    CausalModelWrapperForKVCache,
)

from mcomp.utils.logging import get_test_logger

logger = get_test_logger()

def get_test_config(args):
    
    if args.yaml:
        config_dict = yaml.safe_load(open(args.yaml))
        config = TestConfig(**config_dict)
        return config

    config = TestConfig()
    config.kv_quant = args.kv_quant
    config.i4fp8_linear = args.linear_quant
    config.group_size = args.group_size
    config.qk_pre_scale = args.qk_pre_scale
    config.alpha = float(args.alpha)
    config.qk_post_scale = args.qk_post_scale
    config.beta = float(args.beta)
    config.chwise_scale = args.chwise_scale
    config.scale_type = int(args.scale_type)
    config.qkv_scale = float(args.qkv_scale)
    config.o_scale = float(args.o_scale)
    config.ug_scale = float(args.ug_scale)
    config.d_scale = float(args.d_scale)
    config.v_mul = float(args.v_mul)
    config.top_k = int(args.top_k)
    config.model_path = args.model
    config.attn_name = args.model_attn_name
    config.mlp_name = args.model_mlp_name
    config.calib_data = args.calib_data
    config.split = args.split
    config.text_col = args.text_col
    config.batch = args.batch
    config.seq_len = args.seq_len
    config.awq_smoothing = args.awq_smoothing
    config.orthogonal = args.orthogonal
    config.no_scaled_linear = args.no_scaled_linear
    config.quant_yaml = args.quant_yaml
    
    config.kv_cache_key = args.kv_cache_key
    config.max_cache_len = args.max_cache_len
    config.task = args.task

    return config

def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(description="TestConfig Arguments")
    parser.add_argument("--model", "-m", help="model_path", required=True)
    parser.add_argument("--yaml", help="Specific yaml_path already saved", default="")
    
    parser.add_argument("--kv-quant", 
                        action='store_true',
                        help="I4F8 KV Quantization", 
                        )

    parser.add_argument("--linear-quant", action='store_true', help="I4F8_F8 Linear quantization")
    parser.add_argument("--group-size", "-g", type=int, default=128, help="group-size of int quantization") ## TODO group-size 연계
    
    parser.add_argument("--qk-pre-scale", action='store_true', help="Apply qk-pre-scale")
    parser.add_argument("--alpha", type=float, default=1.0, help="multiplier for qk-pre-scale")
    
    parser.add_argument("--qk-post-scale", action='store_true', help="Apply qk-post-scale")
    parser.add_argument("--beta", type=float, default=1.0, help="multiplier for qk-post-scale")
    parser.add_argument("--top-k", type=int, default=3, help="top-k for outlier selection in qk-post-scale")

    parser.add_argument("--chwise-scale", action='store_true', help="Apply chwise-scale")
    parser.add_argument("--scale-type", type=int, default=0, help="scaling type")

    parser.add_argument("--qkv-scale", type=float, default=0.0, help="Apply chwise-scale for qkv proj weight")
    parser.add_argument("--o-scale", type=float, default=0.0, help="Apply chwise-scale for o proj weight")
    parser.add_argument("--ug-scale", type=float, default=0.0, help="Apply chwise-scale for up&gate proj weight")
    parser.add_argument("--d-scale", type=float, default=0.0, help="Apply chwise-scale for down proj weight")
    parser.add_argument("--v-mul", type=float, default=1.0, help="Apply scalar multiplication for v weight")
    
    parser.add_argument("--model-attn-name", type=str, default='self_attn', help="The attention module name in the model")
    parser.add_argument("--model-mlp-name", type=str, default='mlp', help="The mlp module name in the model")
    
    parser.add_argument("--calib-data", type=str, default='mit-han-lab/pile-val-backup', help="Calibration Datasets name (huggingface)")
    parser.add_argument("--split", type=str, default='validation', help="The split of the calibration Datasets (huggingface)")
    parser.add_argument("--text-col", type=str, default='text', help="The text column key for the Datasets")
    
    parser.add_argument("--batch", "-b", type=int, default=10, help="Total batch size of the calibration data")
    parser.add_argument("--seq-len", "-s", type=int, default=2048, help="Sequence length of the calibration data")
    
    parser.add_argument("--device", "-d", type=str, default='cuda', help="Device for calibration to be conducted")
    parser.add_argument("--save", action='store_true', help="Saving Test Model")
    parser.add_argument("--save-path", type=str, default='default', help="Model path to be saved", required=True)
    parser.add_argument("--output-path", type=str, default='default_test_result', help="Test output path to be saved")
    parser.add_argument("--test-batch", type=int, default=4, help="Total batch size for Tests (StaticCache)")
    parser.add_argument("--orthogonal", action='store_true', help="Hadamard Orthogonal Transform")
    parser.add_argument("--awq-smoothing", action='store_true', help="AWQ Smoothing")
    parser.add_argument("--no-scaled-linear", action='store_true', help="Do not use SCALED_I4F8_F8, but I4F8_F8")
    parser.add_argument("--quant-yaml", type=str, default=None, help="Specific quant yaml path saved in quant_configs")
    
    parser.add_argument("--kv-cache-key", type=str, default='i4fp8pseudo', help="KV Cache Type key")
    parser.add_argument("--max-cache-len", type=int, default=16384, help="max cache length for StaticCache")
    parser.add_argument("--task", type=str, default='wikitext', help="lm_eval task name or wikitext for ppl")
    
    args = parser.parse_args()
    logger.debug("arguments:")
    logger.debug(args)
    
    test_config = get_test_config(args)
    return test_config, args

def main(test_config, device, save_path, args):
    
    config_dict = get_config(save_path)
    
    # quantizer
    
    if config_dict is None:
        quantizer, model, tokenizer = quantize(test_config, device)
    else:
        quantizer, model, tokenizer = load_model(save_path)
        args.save = False
    
    if args.save:
        quantizer.save_quantized(save_path)
        
        save_test_config(save_path, test_config)
    

    model = model.to('cuda:0') # single process for now
    
    layer_device_map = get_kvcache_layer_device_map_from_model_and_attn_name(model, test_config.attn_name)
    
    if test_config.kv_quant:
        scale_dtype=torch.float8_e4m3fn
        if test_config.kv_cache_key == 'i4f16pseudo':
            scale_dtype=model.dtype
        model = get_kv_wraped_model(model, 
                                      kv_cache_key=test_config.kv_cache_key, 
                                      model_config=model.config,
                                      batch_size=args.test_batch,
                                      max_cache_len=test_config.max_cache_len,
                                      dtype=model.dtype,
                                      scale_dtype=scale_dtype,
                                      layer_device_map = layer_device_map
                                     )
    else:
        pass
    
    logger.debug(model)
    if isinstance(model, CausalModelWrapperForKVCache):
        logger.debug(f"KV Cache_Key: {test_config.kv_cache_key}")
        logger.debug(f"KV max_cache_len: {test_config.max_cache_len}")
        logger.debug(f"value cache dtype: {model.kv_cache.value_cache[0].dtype}")
        logger.debug(f"value scale cache dtype: {model.kv_cache.value_scale_cache[0].dtype}")
        logger.debug(f"key cache dtype: {model.kv_cache.key_cache[0].dtype}")
        logger.debug(f"key scale cache dtype: {model.kv_cache.key_scale_cache[0].dtype}")
    
    
    task_list = test_config.task.split(',')
    wiki_ppl = False
    wiki_idx = -1
    for idx, task_name in enumerate(task_list):
        if task_name == 'wikitext':
            wiki_idx = idx
            wiki_ppl = True
            break
        
    task_list.pop(wiki_idx)
    
    if wiki_ppl:
        task = 'wikitext'
        res = eval_ppl_with_gptq_evaluator(
            model=model,
            tokenizer=tokenizer,
            task=task
        )
        
        print(args.output_path)
        os.makedirs(args.output_path, exist_ok=True)
        # with open(os.path.join(args.output_path, 'wikitext_preplexity.txt'), 'w') as f:
        #     f.write(str(res))
        res_str = ""
        if test_config.qk_pre_scale:
            res_str += f"* pre-RoPE: {args.alpha} | "
        if test_config.qk_post_scale:
            res_str += f"topk: {args.top_k} | post-RoPE: {args.beta} | "
        if test_config.v_mul != 1:
            res_str += f"vmul: {args.v_mul} | "
        if test_config.qkv_scale != 0.0:
            res_str += f"qkv: {args.qkv_scale} | "
        if test_config.o_scale != 0.0:
            res_str += f"o: {args.o_scale} | "
        if test_config.ug_scale != 0.0:
            res_str += f"up & gate: {args.ug_scale} | "
        if test_config.d_scale != 0.0:
            res_str += f"down: {args.d_scale} | "
        
        res_str += f"PPL: {res}\n"
        
        file_name = 'wikitext_preplexity.txt'
        with open(os.path.join(args.output_path, file_name), 'a') as g:
            g.write(res_str)
            
    if task_list:

        hflm_model = huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend='causal',
            use_cache=True,
            batch_size=4,
            max_batch_size=4,
            device='cuda:0' ## for now, single process
        )
        #task_list = test_config.task.split(',')
        ## conduct Test
        results = lm_eval.simple_evaluate(
            model=hflm_model,
            tasks=task_list,
            num_fewshot=0,
            batch_size=4
        )
        
        os.makedirs(args.output_path, exist_ok=True)
        dumped = json.dumps(
            results,
            indent=2,
            default=handle_non_serializable,
            ensure_ascii=False,
        )
        
        with open(os.path.join(args.output_path, f'lm_eval_{test_config.task}.json'), 'w') as f:
            f.write(dumped)
    
        
    from mcomp.modules.linears import LinearScaled_I4F8_F8
    logger.debug(f"---[per-tensor-scale]---")
    for n, m in model.named_modules():
        if isinstance(m, LinearScaled_I4F8_F8):
            logger.debug(f"{n}: {(1/m.per_tensor_scale_inv).item()}")
            
    logger.debug(f"Attn_Impl: {model.config._attn_implementation}")


if __name__ == '__main__':
    
    test_config, args = parse_args()    
    main(test_config, args.device, args.save_path, args)
    
    logger.debug(f"Test result has been saved on {args.output_path}")
    