# FreqNet / FreqNetCD

Lightweight frequency-domain models for long-term multivariate time series
forecasting, implemented as drop-in modules for the
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib).

FreqNet is a compact frequency-domain forecaster: after reversible instance
normalization it keeps only the lowest rFFT coefficients and fuses a frequency
branch and a linear branch with an input-adaptive gate. FreqNetCD adds a
cross-channel self-attention branch whose parameter count is independent of the
number of variates, attached through a ReZero gate initialized at zero, so the
model is channel-independent at initialization and turns on cross-channel mixing
only where it helps. The single learned gate magnitude `|gamma|` reports, per
dataset, how much the model relies on cross-variate information.

## Repository layout

```
FreqNet/
├── README.md
├── LICENSE
├── requirements.txt
├── models/
│   ├── FreqNetCD.py            # main model (FreqNet = CD branch off)
│   ├── FITS.py                 # frequency baseline used for comparison
│   ├── SparseTSF.py            # ultra-lightweight baseline (ICML 2024)
│   ├── FreqNetCD_noCD.py       # ablation: no cross-channel branch
│   ├── FreqNetCD_noFusion.py   # ablation: fixed equal-weight fusion
│   └── FreqNetCD_noRevIN.py    # ablation: no instance normalization
├── scripts/
│   ├── run_freqnetcd_seeds.sh  # FreqNetCD over 3 seeds
│   ├── run_freqnet_all.sh      # channel-independent base
│   ├── run_fits_all.sh         # FITS baseline
│   ├── run_ablation.sh         # module ablation (seed 2021)
│   ├── run_ablation_seeds.sh   # module ablation over 3 seeds
│   ├── run_sensitivity.sh      # K / hidden / lambda sweep
│   ├── run_newdata_all.sh      # PM2.5 and NASDAQ 100
│   ├── run_lookback.sh         # lookback-length sensitivity
│   ├── run_newbaselines.sh     # TimeXer / WPMixer
│   ├── run_sparsetsf.sh        # SparseTSF baseline
│   ├── run_test_perwindow.sh   # export per-window errors for DM tests
│   ├── run_test_newbaselines_perwindow.sh
│   ├── run_freqnet_ecl_ili_seeds.sh
│   └── run_noise.sh            # Gaussian-noise robustness
├── tools/
│   ├── prepare_pm25.py         # convert UCI Beijing air-quality to TSLib CSV
│   ├── prepare_nasdaq.py       # convert DA-RNN NASDAQ 100 to TSLib CSV
│   ├── aggregate_seeds.py      # mean ± std from multi-seed logs
│   ├── significance_tests.py   # Diebold–Mariano / Wilcoxon / paired t
│   ├── significance_all.py     # batch DM tests over result folders
│   ├── significance_ili.py     # DM tests for the ILI dataset
│   └── gate_cost.py            # parameter and latency cost of the gates
├── patches/
│   ├── apply_perwindow_patch.py
│   └── apply_noise_patch.py
└── data/
    └── README.md               # how to obtain the datasets
```

## Environment

- Python 3.12 (Ubuntu 22.04)
- PyTorch 2.3.0, CUDA 12.1
- A single GPU with 32 GB of memory

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Download the nine standard benchmarks (ETT, Weather, Exchange, Electricity,
Traffic, ILI) from Time-Series-Library and place them as described in
[`data/README.md`](data/README.md). Prepare the two additional datasets with:

```bash
python tools/prepare_pm25.py PRSA_Data.zip ./dataset/pm25/pm25.csv
python tools/prepare_nasdaq.py nasdaq100_padding.csv ./dataset/nasdaq/nasdaq100.csv
```

## Training and evaluation

The model files are drop-in modules for the
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib).

1. Clone TSLib and prepare its environment and datasets.
2. Copy the files in `models/` into the TSLib `models/` directory.
3. Add the following arguments to TSLib's `run.py` (argument parser):

   ```python
   parser.add_argument('--freqnet_k', type=int, default=30)
   parser.add_argument('--freqnet_hidden', type=int, default=64)
   parser.add_argument('--freqnet_cd_l1', type=float, default=5e-3)
   parser.add_argument('--seed', type=int, default=2021)
   parser.add_argument('--period_len', type=int, default=24)  # SparseTSF
   ```

   The remaining `freqnet_*` switches are read via `getattr` with defaults, so
   the default configuration is the standard one; the ablation variants set
   them in code.
4. (Optional, for significance and noise studies.) From the TSLib root:

   ```bash
   python patches/apply_perwindow_patch.py .
   python patches/apply_noise_patch.py .
   ```

5. Run the scripts in `scripts/` from the TSLib root, e.g.:

   ```bash
   bash scripts/run_freqnetcd_seeds.sh   # main FreqNetCD runs (3 seeds)
   bash scripts/run_freqnet_all.sh       # channel-independent base
   bash scripts/run_fits_all.sh          # FITS baseline
   bash scripts/run_ablation_seeds.sh    # three-seed ablation
   bash scripts/run_sensitivity.sh       # sensitivity study
   bash scripts/run_newdata_all.sh       # PM2.5 and NASDAQ 100
   bash scripts/run_lookback.sh          # lookback-length study
   bash scripts/run_test_perwindow.sh    # per-window errors for DM tests
   python tools/significance_all.py      # Diebold–Mariano summary CSV
   ```

   A single run is also possible directly, e.g. ETTh1 / predict-96:

   ```bash
   python -u run.py --task_name long_term_forecast --is_training 1 \
     --model FreqNetCD --data ETTh1 \
     --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
     --features M --seq_len 96 --label_len 48 --pred_len 96 \
     --enc_in 7 --dec_in 7 --c_out 7 --des Exp
   ```

The heavy Transformer baselines (iTransformer, PatchTST, Crossformer,
TimesNet, TimeMixer, TimeXer, WPMixer) are run with their official
Time-Series-Library implementations; `scripts/run_newbaselines.sh` and
`scripts/run_newdata_all.sh` record the exact flags used.

## Acknowledgment

The training and evaluation pipeline and the baseline implementations are from
the [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
PM2.5 uses the Beijing Multi-Site Air-Quality Data Set (Liang et al., 2015);
NASDAQ 100 uses the DA-RNN release (Qin et al., IJCAI 2017).

## License

Released under the [MIT License](LICENSE).
