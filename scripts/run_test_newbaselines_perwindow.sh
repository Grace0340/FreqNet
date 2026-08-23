#!/bin/bash
# Test-only re-evaluation of TimeXer / WPMixer / SparseTSF on PM2.5 and
# NASDAQ 100 to export per-window squared errors for the DM tests. The
# checkpoints from run_newbaselines.sh / run_sparsetsf.sh are reused
# (is_training 0), so no retraining happens; architecture flags must match
# the training commands exactly for the setting string to resolve.
# Usage: nohup bash run_test_newbaselines_perwindow.sh > test_perwindow_nb.log 2>&1 &

cd /root/autodl-tmp/Time-Series-Library
export PATH=/root/miniconda3/envs/timesnet1/bin:$PATH

TEST="--task_name long_term_forecast --is_training 0 --features M --des Exp --itr 1"

for CFG in "pm25 pm25.csv h 11 PM25" "nasdaq nasdaq100.csv t 82 NASDAQ"; do
  set -- $CFG
  DIR=$1; CSV=$2; FRQ=$3; C=$4; NAME=$5
  for PL in 96 192 336 720; do
    python -u run.py $TEST --model TimeXer \
      --root_path ./dataset/${DIR}/ --data_path ${CSV} --data custom --freq ${FRQ} \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len ${PL} \
      --e_layers 1 --factor 3 --enc_in ${C} --dec_in ${C} --c_out ${C} \
      --d_model 256 --d_ff 512 --batch_size 32

    python -u run.py $TEST --model WPMixer \
      --root_path ./dataset/${DIR}/ --data_path ${CSV} --data custom --freq ${FRQ} \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len ${PL} \
      --enc_in ${C} --dec_in ${C} --c_out ${C} \
      --d_model 128 --patch_len 16 --batch_size 32

    python -u run.py $TEST --model SparseTSF \
      --root_path ./dataset/${DIR}/ --data_path ${CSV} --data custom --freq ${FRQ} \
      --model_id ${NAME}_96_${PL} --seq_len 96 --label_len 48 --pred_len ${PL} \
      --enc_in ${C} --dec_in ${C} --c_out ${C} --batch_size 256
  done
done

echo "==== perwindow test export done ===="
