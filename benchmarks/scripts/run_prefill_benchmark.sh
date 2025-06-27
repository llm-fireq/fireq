#!/bin/bash

models=("llama3-8B-i4f8-pre-post-chwise" )
bszs=("1" "4" "16" )
seq_lens=("1024")

script="prefill_benchmark.py"

log_file_path="./logs/prefill_benchmark-$(date +"%m-%d %H:%M:%S").csv"
mkdir -p "./logs"

echo "$(date): Run Prefill benchmark "
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

            #time=$(python $script --model $model --seq_len $seq_len --bsz $bsz | grep "MLP Average time" | awk '{print $4}' | sed 's/ms//g')
            time=$( $cmd | grep "Prefill Average time" | awk '{print $4}' | sed 's/ms//g')
            echo "$model Prefill average latency: ${time}ms"
            echo "$model,$seq_len,$bsz,$time" >> "${log_file_path}"
        done
    done
done