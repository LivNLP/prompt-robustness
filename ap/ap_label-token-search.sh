#!/bin/bash

task="cb"
datapath="../data/superglue"
gpu=0
model="roberta-large"
xtrig=10
dp=200

echo "Label search"
for seed in 0 1 2 3
do
    labelfile="model/$task/$model/$seed/result_label-tokens_xtrig-${xtrig}/label-tokens_datapoint-$dp.json"  
    
    echo "\nExperiment: xtrig = $xtrig , seed = $seed , num_datapiint = $dp"
    
    python run_label_search.py \
    --train $datapath/$task/datapoint-random-$seed/train_dp-$dp.tsv \
    --template '<s> {sentence_A} [P] [T] [T] [T] [T] [T] [T] [T] [T] [T] [T] {sentence_B} </s>'  \
    --label-map '{"entailment": 0, "contradiction": 1, "neutral": 2}' \
    --result-filename $labelfile \
    --iters 500 \
    --bsz 32  \
    --model-name $model \
    --seed $seed \
    --gpu $gpu
done



