#!/bin/bash
# P2: noise-robustness study. Re-tests existing checkpoints (FreqNetCD seed2021
# and FITS on the nine standard benchmarks) with Gaussian input noise of
# sigma x per-channel std, via the TSLIB_NOISE_SIGMA hook installed by
# apply_noise_patch.py. Noisy metrics go to result_noise.txt and to
# results/<setting>_noise<s>/ folders; clean files are untouched.
# Requires: apply_noise_patch.py applied first.
# Usage: nohup bash run_noise.sh > noise.log 2>&1 &

export TSLIB_RESULT_FILE=result_noise.txt
for SIG in 0.1 0.2 0.4; do
  echo "########## noise sigma = $SIG ##########"
  export TSLIB_NOISE_SIGMA=$SIG
  bash run_test_perwindow.sh
done
unset TSLIB_NOISE_SIGMA TSLIB_RESULT_FILE
echo "==== noise robustness finished ===="
