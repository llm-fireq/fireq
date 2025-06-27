#!/bin/bash

## I4_F8 Pre-ROPE Scale + Post-ROPE Scale + Chwise Scale
gpu_device="3"
script_path="quant_eval.py"
model="meta-llama/Meta-Llama-3-8B"

qkv=0
o=0
ug=0.0096
d=0.015
vmul=0.9

topk=8
beta=8
alpha=8

task="wikitext,piqa,hellaswag,ai2_arc,winogrande,boolq"
#task="wikitext"

test_case="llama3-8B-i4-pre-post-chwise"
CUDA_VISIBLE_DEVICES=$gpu_device python3 $script_path -m $model \
    --save-path "$test_case" \
    --save \
    --linear-quant \
    --kv-quant \
    --kv-cache-key 'i4f16pseudo' \
    --qk-pre-scale \
    --alpha $alpha \
    --qk-post-scale \
    --beta $beta \
    --top-k $topk \
    --chwise-scale \
    --scale-type=2 \
    --qkv-scale $qkv \
    --o-scale $o \
    --ug-scale $ug \
    --d-scale $d \
    --v-mul $vmul \
    --task $task \
    --quant-yaml "llama3_quant_i4_f8_linear_kv_pre_post.yaml" \
    --output-path "result/$test_case"

#> "result/$test_case.log" 2>&1
