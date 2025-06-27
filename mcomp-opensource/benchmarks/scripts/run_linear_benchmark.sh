#!/bin/bash

bszs=("1" "4" "16" "64")
seq_lens=("1024")

script="linear_kernel_benchmark.py"

echo "$(date): Run Linear benchmark "


for seq_len in "${seq_lens[@]}"    
do
    for bsz in "${bszs[@]}"
    do
        cmd="python $script --seq_len ${seq_len} --bsz ${bsz}"
        for _ in $(seq 1 200); do echo -n "-"; done
        echo -e "\nBenchamrk $model batch_size: $bsz seq_len: $seq_len"

        $cmd
    done
done
