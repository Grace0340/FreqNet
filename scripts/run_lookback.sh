#!/bin/bash
# P1: lookback-length sensitivity study.
# Lookback {192,336,720} x models {FreqNetCD,FITS,DLinear,PatchTST} x
# datasets {ETTh2,Weather,Traffic} x horizons {96,192,336,720}.
# Lookback-96 results already exist from the main experiments.
# Skip-guard: reruns are skipped if the setting is already in the result file.
# Usage: nohup bash run_lookback.sh > lookback.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting lookback study"

# Sensitivity study: shorter schedule is enough (early-stop usually fires
# well before 20). Saves ~2x wall time vs the main-table 50/10 protocol.
COMMON="--task_name long_term_forecast --is_training 1 --features M \
  --learning_rate 0.001 --train_epochs 20 --patience 5 --des Exp --itr 1"

run_one () {  # $1=model $2=root $3=csv $4=data $5=C $6=name $7=SL $8=PL $9=extra
  local MODEL=$1 ROOT=$2 CSV=$3 DATA=$4 C=$5 NAME=$6 SL=$7 PL=$8; shift 8; local EXTRA="$@"
  local LL=$((SL / 2))
  local TAG="long_term_forecast_${NAME}_${SL}_${PL}_lb_${MODEL}_${DATA}_"
  if grep -q "$TAG" result_long_term_forecast.txt 2>/dev/null; then
    echo ">> skip (done): $TAG"; return
  fi
  local ARCH=""
  if [ "$MODEL" = "PatchTST" ]; then
    ARCH="--e_layers 2 --d_layers 1 --factor 3 --n_heads 4"
  fi
  if [ "$MODEL" = "FreqNetCD" ]; then
    ARCH="--seed 2021"
  fi
  python -u run.py $COMMON --model $MODEL $ARCH \
    --root_path $ROOT --data_path $CSV --data $DATA \
    --enc_in $C --dec_in $C --c_out $C \
    --model_id ${NAME}_${SL}_${PL}_lb \
    --seq_len $SL --label_len $LL --pred_len $PL $EXTRA
}

for SL in 192 336 720; do
  for MODEL in FreqNetCD FITS DLinear PatchTST; do
    for PL in 96 192 336 720; do
      run_one $MODEL ./dataset/ETT-small/ ETTh2.csv ETTh2 7 ETTh2 $SL $PL
    done
  done
done

for SL in 192 336 720; do
  for MODEL in FreqNetCD FITS DLinear PatchTST; do
    for PL in 96 192 336 720; do
      run_one $MODEL ./dataset/weather/ weather.csv custom 21 Weather $SL $PL
    done
  done
done

# Traffic light models only here. PatchTST is launched in a parallel
# stream (run_lookback_traffic_patchtst.sh) to use idle GPU capacity.
for SL in 192 336 720; do
  for MODEL in FreqNetCD FITS DLinear; do
    for PL in 96 192 336 720; do
      run_one $MODEL ./dataset/traffic/ traffic.csv custom 862 Traffic $SL $PL --batch_size 16
    done
  done
done
echo "==== lookback light models finished (PatchTST Traffic runs in parallel stream) ===="
# If the parallel PatchTST stream is not running, fall back to serial here.
if ! pgrep -af 'run_lookback_traffic_patchtst' | grep -v pgrep >/dev/null; then
  echo ">> PatchTST parallel stream not found; running PatchTST Traffic serially"
  for SL in 192 336 720; do
    for PL in 96 192 336 720; do
      run_one PatchTST ./dataset/traffic/ traffic.csv custom 862 Traffic $SL $PL --batch_size 16
    done
  done
fi
echo "==== lookback study finished ===="
