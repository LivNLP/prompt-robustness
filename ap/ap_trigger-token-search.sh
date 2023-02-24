#!/bin/bash

task="cb"
datapath="../data/superglue"
gpu=0 # GPU ID
model="roberta-large"
xtrig=10
vy=3
vcand=10
devfile="dev.tsv" # 50 instances sampled for the development dataset
dp=200

echo "Create trigger tokens"
for seed in 0 1 2 3
do
    labelfile="model/$task/$model/$seed/result_label-tokens_xtrig-${xtrig}/label-tokens_datapoint-$dp.json"
    resultfile="model/$task/$model/$seed/result_trigger-tokens_xtrig-${xtrig}_vy-${vy}_vcand-${vcand}/trigger-tokens_datapoint-$dp.json"
    
    echo "\nExperiment xtrig = $xtrig , vy = $vy , vcand = $vcand , seed = $seed , num_datapoint = $dp"

    python run_trigger_search.py  \
    --train $datapath/$task/datapoint-random-$seed/train_dp-$dp.tsv \
    --dev $datapath/$task/$devfile \
    --label-token-file $labelfile \
    --result-filename $resultfile \
    --n-label-tokens $vy \
    --template '<s> {sentence_A} [P] [T] [T] [T] [T] [T] [T] [T] [T] [T] [T] {sentence_B} </s>'  \
    --iters 300 \
    --num-cand $vcand \
    --accumulation-steps 15 \
    --bsz 16 \
    --eval-size 32 \
    --model-name $model \
    --seed $seed \
    --gpu $gpu 
done