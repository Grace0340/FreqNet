"""Statistical significance tests for paired forecast comparisons.

Two modes:

1) Per-window paired tests between two models on one setting
   (requires perwindow_mse.npy exported by the patched test()):

   python significance_tests.py windows <results_dir_model_A> <results_dir_model_B>

   Reports paired t-test, Wilcoxon signed-rank, and the Diebold-Mariano
   statistic with a Newey-West (HAC) variance correction, since forecast
   errors of overlapping windows are autocorrelated.

2) Cross-setting Wilcoxon signed-rank over per-horizon results
   (uses a CSV with columns: dataset,horizon,<modelA>_MSE,<modelB>_MSE,...):

   python significance_tests.py table <results_per_horizon.csv> FreqNetCD_MSE FITS_MSE
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def dm_test(err_a, err_b, h=1):
    """Diebold-Mariano test on loss differentials with HAC variance.

    err_a, err_b: per-window losses (e.g. squared errors). h: forecast
    horizon proxy for the truncation lag of the Newey-West estimator.
    Negative DM favours model A (its loss is smaller).
    """
    d = np.asarray(err_a, dtype=float) - np.asarray(err_b, dtype=float)
    n = len(d)
    dbar = d.mean()
    lag = max(1, int(round(n ** (1 / 3))))
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * w * cov
    dm = dbar / np.sqrt(var / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return dm, p


def mode_windows(dir_a, dir_b):
    a = np.load(Path(dir_a) / "perwindow_mse.npy")
    b = np.load(Path(dir_b) / "perwindow_mse.npy")
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    dm, dm_p = dm_test(a, b)
    print(f"n windows     : {len(a)}")
    print(f"mean MSE A    : {a.mean():.6f}  ({dir_a})")
    print(f"mean MSE B    : {b.mean():.6f}  ({dir_b})")
    print(f"paired t      : t={t_stat:+.3f}, p={t_p:.2e}")
    print(f"Wilcoxon      : W={w_stat:.1f}, p={w_p:.2e}")
    print(f"Diebold-Mariano (HAC): DM={dm:+.3f}, p={dm_p:.2e}"
          f"  ({'A better' if dm < 0 else 'B better'})")


def mode_table(csv_path, col_a, col_b):
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = [r for r in rows if r.get("horizon", "").strip().lower() != "avg"]
    a = np.array([float(r[col_a]) for r in rows])
    b = np.array([float(r[col_b]) for r in rows])
    w_stat, w_p = stats.wilcoxon(a, b)
    wins = int((a < b).sum())
    print(f"settings      : {len(a)}")
    print(f"{col_a} wins  : {wins}/{len(a)}")
    print(f"mean          : {a.mean():.4f} vs {b.mean():.4f}")
    print(f"Wilcoxon      : W={w_stat:.1f}, p={w_p:.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "windows" and len(sys.argv) == 4:
        mode_windows(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "table" and len(sys.argv) == 5:
        mode_table(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
