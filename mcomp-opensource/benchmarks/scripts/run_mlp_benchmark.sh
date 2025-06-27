#!/bin/bash

models=("llama-2-7b" "llama-2-13b" "llama3-8B-i4f8-pre-post-chwise" "llama3-8B-i4-pre-post-chwise")
bszs=("1" "4" "16" "64")
seq_lens=("1024")

script="mlp_benchmark.py"

log_file_path="./logs/mlp_benchmark-$(date +"%m-%d %H:%M:%S").csv"
mkdir -p "./logs"

echo "$(date): Run MLP benchmark "
echo "$log_file_path"
echo "model,seq_len,bsz,time" > "${log_file_path}"
for model in "${models[@]}"
do
    for seq_len in "${seq_lens[@]}"    
    do
        for bsz in "${bszs[@]}"
        do
            cmd="python $script --model ${model} --seq_len ${seq_len} --bsz ${bsz}"
            for _ in $(seq 1 200); do echo -n "-"; done
            echo -e "\nBenchamrk $model batch_size: $bsz seq_len: $seq_len"
            if [[ "$model" == *"-i4-"* ]]; then
                cmd="${cmd} --is_bf16"
            fi
            echo "cmd: ${cmd}"
            #time=$(python $script --model $model --seq_len $seq_len --bsz $bsz | grep "MLP Average time" | awk '{print $4}' | sed 's/ms//g')
            time=$( $cmd | grep "MLP Average time" | awk '{print $4}' | sed 's/ms//g')
            echo "$model MLP average latency: ${time}ms"
            echo "$model,$seq_len,$bsz,$time" >> "${log_file_path}"
        done
    done
done