#!/bin/bash
# Test-only reruns to export per-window errors from EXISTING checkpoints
# (no retraining). Produces perwindow_mse.npy / perwindow_mae.npy in each
# results/<setting>/ dir, used by tools/significance_tests.py for the paired
# FreqNetCD-vs-FITS tests.
# Requires apply_perwindow_patch.py to have been applied first.
# Usage: nohup bash run_test_perwindow.sh > perwindow.log 2>&1 &

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "!! no GPU available, aborting"; exit 1; }
echo ">> GPU OK, starting test-only per-window export"

# FreqNetCD, seed 2021 (checkpoints named <DS>_96_<PL>_seed2021)
CD="--task_name long_term_forecast --is_training 0 --model FreqNetCD --features M \
  --des Exp --itr 1 --seed 2021"
# FITS (checkpoints named <DS>_96_<PL>)
FT="--task_name long_term_forecast --is_training 0 --model FITS --features M \
  --des Exp --itr 1"

for DS in ETTh1 ETTh2 ETTm1 ETTm2; do
  for PL in 96 192 336 720; do
    python -u run.py $CD --root_path ./dataset/ETT-small/ --data_path ${DS}.csv \
      --model_id ${DS}_96_${PL}_seed2021 --data ${DS} \
      --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 --dec_in 7 --c_out 7
    python -u run.py $FT --root_path ./dataset/ETT-small/ --data_path ${DS}.csv \
      --model_id ${DS}_96_${PL} --data ${DS} \
      --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 7 --dec_in 7 --c_out 7
  done
done

for PL in 96 192 336 720; do
  python -u run.py $CD --root_path ./dataset/weather/ --data_path weather.csv \
    --model_id Weather_96_${PL}_seed2021 --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 21 --dec_in 21 --c_out 21
  python -u run.py $FT --root_path ./dataset/weather/ --data_path weather.csv \
    --model_id Weather_96_${PL} --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 21 --dec_in 21 --c_out 21
done

for PL in 96 192 336 720; do
  python -u run.py $CD --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv \
    --model_id Exchange_96_${PL}_seed2021 --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 8 --dec_in 8 --c_out 8
  python -u run.py $FT --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv \
    --model_id Exchange_96_${PL} --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 8 --dec_in 8 --c_out 8
done

for PL in 96 192 336 720; do
  python -u run.py $CD --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id ECL_96_${PL}_seed2021 --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 321 --dec_in 321 --c_out 321
  python -u run.py $FT --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id ECL_96_${PL} --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 321 --dec_in 321 --c_out 321
done

for PL in 96 192 336 720; do
  python -u run.py $CD --root_path ./dataset/traffic/ --data_path traffic.csv \
    --model_id Traffic_96_${PL}_seed2021 --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 862 --dec_in 862 --c_out 862 --batch_size 16
  python -u run.py $FT --root_path ./dataset/traffic/ --data_path traffic.csv \
    --model_id Traffic_96_${PL} --data custom \
    --seq_len 96 --label_len 48 --pred_len ${PL} --enc_in 862 --dec_in 862 --c_out 862 --batch_size 16
done

for PL in 24 36 48 60; do
  python -u run.py $CD --root_path ./dataset/illness/ --data_path national_illness.csv \
    --model_id ILI_36_${PL}_seed2021 --data custom \
    --seq_len 36 --label_len 18 --pred_len ${PL} --enc_in 7 --dec_in 7 --c_out 7
  python -u run.py $FT --root_path ./dataset/illness/ --data_path national_illness.csv \
    --model_id ILI_36_${PL} --data custom \
    --seq_len 36 --label_len 18 --pred_len ${PL} --enc_in 7 --dec_in 7 --c_out 7
done

echo "==== per-window export finished ===="
