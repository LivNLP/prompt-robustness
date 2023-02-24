#!/bin/bash

task="cb" # "mnli"
gpu=0
modeltype="roberta"
model="roberta-large"
dp=200

for seed in 0 1 2 3
do
    echo "\nExperiment: seed: $seed , num_datapoint: $dp"
    python run_mp_finetuning.py  \
    --task $task \
    --seed $seed \
    --dp $dp \
    --out_path "model/$task/$model/$seed/dp-$dp/plm" \
    --bsz 4  \
    --iters 1000  \
    --eval_interval 200 \
    --model_type $modeltype \
    --model_name $model \
    --seed $seed \
    --gpu $gpu
done
