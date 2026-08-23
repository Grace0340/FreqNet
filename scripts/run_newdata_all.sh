#!/bin/bash
# Experiments on the two additional datasets:
#   PM25   = Beijing Multi-Site Air Quality, Aotizhongxin station (11 channels, hourly)
#   NASDAQ = NASDAQ 100 stock data, DA-RNN version (82 channels, minute bars)
# Our models run over 3 seeds; baselines follow TSLib official Weather-script
# architecture hyperparameters with a single run (same protocol as the main runs).
# Usage: nohup bash run_newdata_all.sh > newdata.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting new-dataset runs"

# ---------- our models, 3 seeds ----------
OURS="--task_name long_term_forecast --is_training 1 --features M \
  --des Exp --itr 1 --learning_rate 0.001 --train_epochs 50 --patience 10"

for SEED in 2021 2022 2023; do
  for MODEL in FreqNetCD FreqNet; do
    # PM2.5 (hourly, C=11)
    for PL in 96 192 336 720; do
      python -u run.py $OURS --seed $SEED --model $MODEL \
        --root_path ./dataset/pm25/ --data_path pm25.csv --data custom --freq h \
        --model_id PM25_96_${PL}_seed${SEED} \
        --seq_len 96 --label_len 48 --pred_len $PL \
        --enc_in 11 --dec_in 11 --c_out 11
    done
    # NASDAQ100 (minute bars, C=82)
    for PL in 96 192 336 720; do
      python -u run.py $OURS --seed $SEED --model $MODEL \
        --root_path ./dataset/nasdaq/ --data_path nasdaq100.csv --data custom --freq t \
        --model_id NASDAQ_96_${PL}_seed${SEED} \
        --seq_len 96 --label_len 48 --pred_len $PL \
        --enc_in 82 --dec_in 82 --c_out 82
    done
  done
done

# ---------- lightweight baselines (single run, same protocol) ----------
LIGHT="--task_name long_term_forecast --is_training 1 --features M \
  --des Exp --itr 1 --learning_rate 0.001 --train_epochs 50 --patience 10"

for MODEL in FITS DLinear; do
  for PL in 96 192 336 720; do
    python -u run.py $LIGHT --model $MODEL \
      --root_path ./dataset/pm25/ --data_path pm25.csv --data custom --freq h \
      --model_id PM25_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --enc_in 11 --dec_in 11 --c_out 11
    python -u run.py $LIGHT --model $MODEL \
      --root_path ./dataset/nasdaq/ --data_path nasdaq100.csv --data custom --freq t \
      --model_id NASDAQ_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --enc_in 82 --dec_in 82 --c_out 82
  done
done

# ---------- heavy baselines (TSLib Weather-script architecture params) ----------
run_heavy () {  # $1=root $2=csv $3=freq $4=C $5=name
  local ROOT=$1 CSV=$2 FRQ=$3 C=$4 NAME=$5
  for PL in 96 192 336 720; do
    # iTransformer
    python -u run.py --task_name long_term_forecast --is_training 1 --model iTransformer \
      --root_path $ROOT --data_path $CSV --data custom --freq $FRQ --features M \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --e_layers 3 --d_layers 1 --factor 3 --enc_in $C --dec_in $C --c_out $C \
      --d_model 512 --d_ff 512 --des Exp --itr 1
    # PatchTST
    python -u run.py --task_name long_term_forecast --is_training 1 --model PatchTST \
      --root_path $ROOT --data_path $CSV --data custom --freq $FRQ --features M \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --e_layers 2 --d_layers 1 --factor 3 --enc_in $C --dec_in $C --c_out $C \
      --n_heads 4 --des Exp --itr 1
    # Crossformer
    python -u run.py --task_name long_term_forecast --is_training 1 --model Crossformer \
      --root_path $ROOT --data_path $CSV --data custom --freq $FRQ --features M \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --e_layers 2 --d_layers 1 --factor 3 --enc_in $C --dec_in $C --c_out $C \
      --d_model 32 --d_ff 32 --top_k 5 --des Exp --itr 1
    # TimesNet
    python -u run.py --task_name long_term_forecast --is_training 1 --model TimesNet \
      --root_path $ROOT --data_path $CSV --data custom --freq $FRQ --features M \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len $PL \
      --e_layers 2 --d_layers 1 --factor 3 --enc_in $C --dec_in $C --c_out $C \
      --d_model 32 --d_ff 32 --top_k 5 --des Exp --itr 1
    # TimeMixer
    python -u run.py --task_name long_term_forecast --is_training 1 --model TimeMixer \
      --root_path $ROOT --data_path $CSV --data custom --freq $FRQ --features M \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 0 --pred_len $PL \
      --e_layers 3 --d_layers 1 --factor 3 --enc_in $C --dec_in $C --c_out $C \
      --d_model 16 --d_ff 32 --batch_size 128 --learning_rate 0.01 \
      --train_epochs 20 --patience 10 \
      --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
      --des Exp --itr 1
  done
}

run_heavy ./dataset/pm25/   pm25.csv      h 11 PM25
run_heavy ./dataset/nasdaq/ nasdaq100.csv t 82 NASDAQ

echo "==== new-dataset runs finished ===="
