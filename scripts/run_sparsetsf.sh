#!/bin/bash
# P2: SparseTSF (ICML 2024 Oral) -- the most direct ultra-lightweight rival
# (<1k parameters). Official training protocol (lr 0.02, batch 256, type3),
# unified lookback 96, period_len 24.
# Usage: nohup bash run_sparsetsf.sh > sparsetsf.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting SparseTSF runs"

BASE="--task_name long_term_forecast --is_training 1 --features M --des Exp --itr 1 \
  --learning_rate 0.02 --batch_size 256 --train_epochs 30 --patience 5 --lradj type3"

sp () {  # $1=root $2=csv $3=data $4=C $5=name $6=freq
  for PL in 96 192 336 720; do
    if grep -q "long_term_forecast_${5}_96_${PL}_SparseTSF_${3}_" result_long_term_forecast.txt 2>/dev/null; then
      echo ">> skip SparseTSF $5 $PL"; continue
    fi
    python -u run.py $BASE --model SparseTSF \
      --root_path $1 --data_path $2 --data $3 --freq $6 \
      --model_id ${5}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --enc_in $4 --dec_in $4 --c_out $4
  done
}
sp ./dataset/ETT-small/ ETTh2.csv     ETTh2  7   ETTh2   h
sp ./dataset/weather/   weather.csv   custom 21  Weather 10min
sp ./dataset/traffic/   traffic.csv   custom 862 Traffic h
sp ./dataset/pm25/      pm25.csv      custom 11  PM25    h
sp ./dataset/nasdaq/    nasdaq100.csv custom 82  NASDAQ  t

echo "==== SparseTSF finished ===="
