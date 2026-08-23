#!/bin/bash
# FreqNet (channel-independent base) on all nine benchmarks.
# Run from the Time-Series-Library root after copying models/ into TSLib.
#   bash run_freqnet_all.sh

COMMON="--task_name long_term_forecast --is_training 1 --model FreqNet --features M \
  --des Exp --itr 1 --learning_rate 0.001 --train_epochs 50 --patience 10"

for DS in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PL in 96 192 336 720; do
    python -u run.py $COMMON \
      --root_path ./dataset/ETT-small/ --data_path ${DS}.csv \
      --model_id ${DS}_96_${PL} --data ${DS} \
      --seq_len 96 --label_len 48 --pred_len ${PL} \
      --enc_in 7 --dec_in 7 --c_out 7
  done
done

for PL in 96 192 336 720; do
  python -u run.py $COMMON --root_path ./dataset/weather/ --data_path weather.csv \
    --model_id Weather_96_${PL} --data custom --seq_len 96 --label_len 48 --pred_len ${PL} \
    --enc_in 21 --dec_in 21 --c_out 21
done

for PL in 96 192 336 720; do
  python -u run.py $COMMON --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv \
    --model_id Exchange_96_${PL} --data custom --seq_len 96 --label_len 48 --pred_len ${PL} \
    --enc_in 8 --dec_in 8 --c_out 8
done

for PL in 96 192 336 720; do
  python -u run.py $COMMON --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id ECL_96_${PL} --data custom --seq_len 96 --label_len 48 --pred_len ${PL} \
    --enc_in 321 --dec_in 321 --c_out 321
done

for PL in 96 192 336 720; do
  python -u run.py $COMMON --root_path ./dataset/traffic/ --data_path traffic.csv \
    --model_id Traffic_96_${PL} --data custom --seq_len 96 --label_len 48 --pred_len ${PL} \
    --enc_in 862 --dec_in 862 --c_out 862 --batch_size 16
done

for PL in 24 36 48 60; do
  python -u run.py $COMMON --root_path ./dataset/illness/ --data_path national_illness.csv \
    --model_id ILI_36_${PL} --data custom --seq_len 36 --label_len 18 --pred_len ${PL} \
    --enc_in 7 --dec_in 7 --c_out 7
done
