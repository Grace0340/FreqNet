#!/bin/bash
# Hyperparameter sensitivity: sweep K (low-freq coeffs), hidden dim, and lambda
# (CD L1) on ETTh2 (low-dim) and Traffic (high-dim) at pred_len=96.
# Requires run.py to expose --freqnet_k / --freqnet_hidden / --freqnet_cd_l1.
#   bash run_sensitivity.sh
export CUDA_VISIBLE_DEVICES=0

C="--task_name long_term_forecast --is_training 1 --features M --seq_len 96 --label_len 48 \
   --pred_len 96 --learning_rate 0.001 --train_epochs 50 --patience 10 --des Exp --itr 1 --seed 2021 --model FreqNetCD"

sweep () {  # $1 root $2 data_path $3 data $4 channels $5 name
  for K in 10 20 30 40 50; do
    python -u run.py $C --root_path $1 --data_path $2 --data $3 --enc_in $4 --dec_in $4 --c_out $4 \
      --model_id $5__K__$K --freqnet_k $K
  done
  for H in 32 64 128 256; do
    python -u run.py $C --root_path $1 --data_path $2 --data $3 --enc_in $4 --dec_in $4 --c_out $4 \
      --model_id $5__H__$H --freqnet_hidden $H
  done
  for L in 0 0.001 0.005 0.01 0.05; do
    python -u run.py $C --root_path $1 --data_path $2 --data $3 --enc_in $4 --dec_in $4 --c_out $4 \
      --model_id $5__L__$L --freqnet_cd_l1 $L
  done
}

sweep ./dataset/ETT-small/ ETTh2.csv   ETTh2  7   ETTh2
sweep ./dataset/traffic/   traffic.csv custom 862 Traffic
