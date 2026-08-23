#!/bin/bash
# Multi-seed module ablation: full / w-o CD / w-o RevIN / w-o adaptive-fusion
# on ETTh2, Weather, Traffic, seeds 2021/2022/2023.
# Usage: nohup bash run_ablation_seeds.sh > ablation_seeds.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting multi-seed ablation"

COMMON="--task_name long_term_forecast --is_training 1 --features M \
  --seq_len 96 --label_len 48 --learning_rate 0.001 --train_epochs 50 --patience 10 \
  --des Exp --itr 1"

MODELS="FreqNetCD FreqNetCD_noCD FreqNetCD_noRevIN FreqNetCD_noFusion"

run_ds () {   # $1=root $2=data_path $3=data $4=channels $5=name $6=extra
  for SEED in 2021 2022 2023; do
    for MODEL in $MODELS; do
      for PL in 96 192 336 720; do
        python -u run.py $COMMON --seed $SEED --model $MODEL \
          --root_path $1 --data_path $2 --data $3 \
          --enc_in $4 --dec_in $4 --c_out $4 \
          --model_id $5_96_${PL}_abl_seed${SEED} --pred_len $PL $6
      done
    done
  done
}

run_ds ./dataset/ETT-small/ ETTh2.csv   ETTh2   7   ETTh2   ""
run_ds ./dataset/weather/   weather.csv custom  21  Weather ""
run_ds ./dataset/traffic/   traffic.csv custom  862 Traffic "--batch_size 16"

echo "==== multi-seed ablation finished ===="
