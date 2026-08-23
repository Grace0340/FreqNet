#!/bin/bash
# P1: 2024/2025-generation baselines under the unified lookback-96 protocol.
#   TimeXer (NeurIPS 2024)  -- official TSLib per-dataset architecture params
#   WPMixer (AAAI 2025)     -- no official lookback-96 config; unified light
#                              protocol (d_model 128, patch 16)
# Datasets: ETTh2, Weather, Traffic, PM25, NASDAQ x horizons {96,192,336,720}.
# Usage: nohup bash run_newbaselines.sh > newbaselines.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting new baselines"

# Shorter schedule for these baselines (official TimeXer scripts omit
# epochs; TSLib default is long). 20/5 is enough for ranking tables.
BASE="--task_name long_term_forecast --is_training 1 --features M --des Exp --itr 1 \
  --train_epochs 20 --patience 5"

skip () {  # $1=name $2=sl $3=pl $4=model $5=data
  grep -q "long_term_forecast_${1}_${2}_${3}_${4}_${5}_" result_long_term_forecast.txt 2>/dev/null
}

# ---------------- TimeXer ----------------
for PL in 96 192 336 720; do
  skip ETTh2 96 $PL TimeXer ETTh2 && echo ">> skip TimeXer ETTh2 $PL" || \
  python -u run.py $BASE --model TimeXer \
    --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 \
    --model_id ETTh2_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
    --e_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 256 --d_ff 1024 --batch_size 16
done
for PL in 96 192 336 720; do
  skip Weather 96 $PL TimeXer custom && echo ">> skip TimeXer Weather $PL" || \
  python -u run.py $BASE --model TimeXer \
    --root_path ./dataset/weather/ --data_path weather.csv --data custom \
    --model_id Weather_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
    --e_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 \
    --d_model 256 --d_ff 512 --batch_size 4
done
for PL in 96 192 336 720; do
  skip Traffic 96 $PL TimeXer custom && echo ">> skip TimeXer Traffic $PL" || \
  python -u run.py $BASE --model TimeXer \
    --root_path ./dataset/traffic/ --data_path traffic.csv --data custom \
    --model_id Traffic_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
    --e_layers 3 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 \
    --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.001
done
for PL in 96 192 336 720; do
  skip PM25 96 $PL TimeXer custom && echo ">> skip TimeXer PM25 $PL" || \
  python -u run.py $BASE --model TimeXer \
    --root_path ./dataset/pm25/ --data_path pm25.csv --data custom --freq h \
    --model_id PM25_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
    --e_layers 1 --factor 3 --enc_in 11 --dec_in 11 --c_out 11 \
    --d_model 256 --d_ff 512 --batch_size 32
done
for PL in 96 192 336 720; do
  skip NASDAQ 96 $PL TimeXer custom && echo ">> skip TimeXer NASDAQ $PL" || \
  python -u run.py $BASE --model TimeXer \
    --root_path ./dataset/nasdaq/ --data_path nasdaq100.csv --data custom --freq t \
    --model_id NASDAQ_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
    --e_layers 1 --factor 3 --enc_in 82 --dec_in 82 --c_out 82 \
    --d_model 256 --d_ff 512 --batch_size 32
done

# ---------------- WPMixer ----------------
wpm () {  # $1=root $2=csv $3=data $4=C $5=name $6=freq
  for PL in 96 192 336 720; do
    skip $5 96 $PL WPMixer $3 && { echo ">> skip WPMixer $5 $PL"; continue; }
    python -u run.py $BASE --model WPMixer \
      --root_path $1 --data_path $2 --data $3 --freq $6 \
      --model_id ${5}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --enc_in $4 --dec_in $4 --c_out $4 \
      --d_model 128 --patch_len 16 --batch_size 32 \
      --learning_rate 0.001 --train_epochs 50 --patience 10
  done
}
wpm ./dataset/ETT-small/ ETTh2.csv   ETTh2  7   ETTh2   h
wpm ./dataset/weather/   weather.csv custom 21  Weather 10min
wpm ./dataset/traffic/   traffic.csv custom 862 Traffic h
wpm ./dataset/pm25/      pm25.csv    custom 11  PM25    h
wpm ./dataset/nasdaq/    nasdaq100.csv custom 82 NASDAQ t

echo "==== new baselines finished ===="
