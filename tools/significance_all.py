"""Batch significance tests: FreqNetCD (seed2021) vs each competitor with
perwindow_mse.npy available, for every dataset/horizon. Writes a CSV summary.

Run from TSLib root:  python significance_all.py
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path("results")

# Lookback token is 96 on the standard benchmarks and 36 on ILI.
OURS_RE = re.compile(
    r"^long_term_forecast_(?P<ds>[A-Za-z0-9]+)_(?:96|36)_(?P<pl>\d+)_seed2021_FreqNetCD_"
)
COMP_RE = re.compile(
    r"^long_term_forecast_(?P<ds>[A-Za-z0-9]+)_(?:96|36)_(?P<pl>\d+)_(?P<model>FITS|DLinear|PatchTST|iTransformer|Crossformer|TimesNet|TimeMixer|TimeXer|WPMixer|SparseTSF)_"
)


def dm_test(err_a, err_b):
    d = np.asarray(err_a, dtype=float) - np.asarray(err_b, dtype=float)
    n = len(d)
    dbar = d.mean()
    lag = max(1, int(round(n ** (1 / 3))))
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        var += 2.0 * w * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    dm = dbar / np.sqrt(var / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return dm, p


ours, comps = {}, {}
for d in RESULTS.iterdir():
    if not (d / "perwindow_mse.npy").exists():
        continue
    if "_noise" in d.name:  # noise-robustness runs are evaluated separately
        continue
    m = OURS_RE.match(d.name)
    if m:
        ours[(m["ds"], int(m["pl"]))] = d
        continue
    m = COMP_RE.match(d.name)
    if m:
        comps[(m["ds"], int(m["pl"]), m["model"])] = d

rows = []
for (ds, pl, model), cdir in sorted(comps.items()):
    if (ds, pl) not in ours:
        continue
    a = np.load(ours[(ds, pl)] / "perwindow_mse.npy")
    b = np.load(cdir / "perwindow_mse.npy")
    if a.shape != b.shape:
        print(f"!! shape mismatch {ds} {pl} {model}: {a.shape} vs {b.shape}")
        continue
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    dm, dm_p = dm_test(a, b)
    rows.append({
        "dataset": ds, "horizon": pl, "competitor": model,
        "ours_mse": f"{a.mean():.4f}", "comp_mse": f"{b.mean():.4f}",
        "dm_stat": f"{dm:+.3f}", "dm_p": f"{dm_p:.2e}",
        "wilcoxon_p": f"{w_p:.2e}", "ttest_p": f"{t_p:.2e}",
        "ours_better": "yes" if a.mean() < b.mean() else "no",
        "sig_0.05": "yes" if dm_p < 0.05 else "no",
    })

with open("significance_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"{'ds':<8} {'pl':>4} {'vs':<13} {'ours':>8} {'comp':>8} {'DM':>8} {'p(DM)':>9} {'sig':>4}")
for r in rows:
    print(f"{r['dataset']:<8} {r['horizon']:>4} {r['competitor']:<13} {r['ours_mse']:>8} "
          f"{r['comp_mse']:>8} {r['dm_stat']:>8} {r['dm_p']:>9} {r['sig_0.05']:>4}")
print(f"\n{len(rows)} pairs written to significance_summary.csv")
