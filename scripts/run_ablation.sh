#!/bin/bash
# Module ablation: full / w-o CD / w-o RevIN / w-o adaptive-fusion,
# on ETTh2 (low-dim), Weather (medium), Traffic (high-dim), single seed 2021.
# Requires the four model files copied into the TSLib models/ directory.
#   bash run_ablation.sh
export CUDA_VISIBLE_DEVICES=0

COMMON="--task_name long_term_forecast --is_training 1 --features M \
  --seq_len 96 --label_len 48 --learning_rate 0.001 --train_epochs 50 --patience 10 \
  --des Exp --itr 1 --seed 2021"

MODELS="FreqNetCD FreqNetCD_noCD FreqNetCD_noRevIN FreqNetCD_noFusion"

run_ds () {   # $1=root $2=data_path $3=data $4=channels $5=name
  for MODEL in $MODELS; do
    for PL in 96 192 336 720; do
      python -u run.py $COMMON --model $MODEL \
        --root_path $1 --data_path $2 --data $3 \
        --enc_in $4 --dec_in $4 --c_out $4 \
        --model_id $5_96_$PL --pred_len $PL
    done
  done
}

run_ds ./dataset/ETT-small/ ETTh2.csv   ETTh2   7   ETTh2
run_ds ./dataset/weather/   weather.csv custom  21  Weather
run_ds ./dataset/traffic/   traffic.csv custom  862 Traffic
