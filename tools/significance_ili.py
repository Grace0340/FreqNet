"""DM significance test for ILI: FreqNetCD (seed2021) vs FITS, clean runs only.

The original significance_all.py hard-coded the `_96_` lookback token in its
regex, which silently skipped ILI (whose tags use `_36_`). This script covers
exactly those missing pairs. Run from TSLib root: python significance_ili.py
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path("results")

OURS_RE = re.compile(
    r"^long_term_forecast_ILI_36_(?P<pl>\d+)_seed2021_FreqNetCD_.*_Exp_0$"
)
COMP_RE = re.compile(
    r"^long_term_forecast_ILI_36_(?P<pl>\d+)_FITS_.*_Exp_0$"
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
    m = OURS_RE.match(d.name)
    if m:
        ours[int(m["pl"])] = d
        continue
    m = COMP_RE.match(d.name)
    if m:
        comps[int(m["pl"])] = d

rows = []
for pl in sorted(set(ours) & set(comps)):
    a = np.load(ours[pl] / "perwindow_mse.npy")
    b = np.load(comps[pl] / "perwindow_mse.npy")
    if a.shape != b.shape:
        print(f"!! shape mismatch ILI {pl}: {a.shape} vs {b.shape}")
        continue
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    dm, dm_p = dm_test(a, b)
    rows.append({
        "dataset": "ILI", "horizon": pl, "competitor": "FITS",
        "ours_mse": f"{a.mean():.4f}", "comp_mse": f"{b.mean():.4f}",
        "dm_stat": f"{dm:+.3f}", "dm_p": f"{dm_p:.2e}",
        "wilcoxon_p": f"{w_p:.2e}", "ttest_p": f"{t_p:.2e}",
        "ours_better": "yes" if a.mean() < b.mean() else "no",
        "sig_0.05": "yes" if dm_p < 0.05 else "no",
        "n_windows": len(a),
    })

with open("significance_ili.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"{'pl':>4} {'ours':>8} {'FITS':>8} {'DM':>8} {'p(DM)':>9} {'n':>5} {'sig':>4}")
for r in rows:
    print(f"{r['horizon']:>4} {r['ours_mse']:>8} {r['comp_mse']:>8} "
          f"{r['dm_stat']:>8} {r['dm_p']:>9} {r['n_windows']:>5} {r['sig_0.05']:>4}")
print(f"\n{len(rows)} pairs written to significance_ili.csv")
