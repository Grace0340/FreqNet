#!/bin/bash
# FreqNet (CI) multi-seed fill-in: ECL + ILI, seeds 2022/2023 (seed 2021 = the
# original single-seed runs, since run.py defaults to fix_seed=2021).
# Hyperparameters identical to run_freqnet_all.sh / run_freqnetcd_seeds.sh.
# Usage: nohup bash run_freqnet_ecl_ili_seeds.sh > freqnet_ecl_ili_seeds.log 2>&1 &

cd /root/autodl-tmp/Time-Series-Library

COMMON="--task_name long_term_forecast --is_training 1 --model FreqNet --features M \
  --des Exp --itr 1 --learning_rate 0.001 --train_epochs 50 --patience 10"

# ILI first (fast, 7 channels, short sequences)
for SEED in 2022 2023; do
  for PL in 24 36 48 60; do
    python -u run.py $COMMON --seed ${SEED} \
      --root_path ./dataset/illness/ --data_path national_illness.csv \
      --model_id ILI_36_${PL}_seed${SEED} --data custom \
      --seq_len 36 --label_len 18 --pred_len ${PL} \
      --enc_in 7 --dec_in 7 --c_out 7
  done
done

# ECL (321 channels)
for SEED in 2022 2023; do
  for PL in 96 192 336 720; do
    python -u run.py $COMMON --seed ${SEED} \
      --root_path ./dataset/electricity/ --data_path electricity.csv \
      --model_id ECL_96_${PL}_seed${SEED} --data custom \
      --seq_len 96 --label_len 48 --pred_len ${PL} \
      --enc_in 321 --dec_in 321 --c_out 321
  done
done

echo "==== FreqNet ECL/ILI seed2022/2023 ALL DONE ===="
