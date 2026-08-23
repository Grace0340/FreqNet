"""Aggregate FreqNetCD three-seed results into mean +/- std tables.

Parses TSLib's result_long_term_forecast.txt, keeps entries whose model_id
contains seed2021/2022/2023, and reports per dataset/horizon:
  MSE mean, MSE std, MAE mean, MAE std, and n_seeds found.

Usage:
  python aggregate_seeds.py <result_long_term_forecast.txt> [out_csv]
"""
import re
import sys
from collections import defaultdict
from statistics import mean, stdev

SETTING_RE = re.compile(
    r"long_term_forecast_(?P<name>[A-Za-z0-9]+)_(?P<sl>\d+)_(?P<pl>\d+)_seed(?P<seed>\d{4})_(?P<model>[A-Za-z_]+?)_"
)
# Metric lines may carry a "time:...," prefix, so search rather than match.
METRIC_RE = re.compile(r"mse:(?P<mse>[0-9.eE+-]+), mae:(?P<mae>[0-9.eE+-]+)")


def main(path, out_csv=None):
    text = open(path, encoding="utf-8", errors="replace").read()
    # Entries appear as: <setting line> \n <metrics line>
    entries = defaultdict(dict)  # (dataset, pl, model) -> {seed: (mse, mae)}
    pending = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SETTING_RE.match(line)
        if m:
            pending = (m.group("name"), int(m.group("pl")), m.group("model"),
                       int(m.group("seed")))
            continue
        mm = METRIC_RE.search(line)
        if mm and pending:
            ds, pl, model, seed = pending
            # Later duplicates overwrite earlier ones (reruns supersede).
            entries[(ds, pl, model)][seed] = (float(mm.group("mse")),
                                              float(mm.group("mae")))
            pending = None

    rows = []
    for (ds, pl, model), seeds in sorted(entries.items()):
        mses = [v[0] for v in seeds.values()]
        maes = [v[1] for v in seeds.values()]
        rows.append({
            "dataset": ds, "horizon": pl, "model": model,
            "n_seeds": len(seeds),
            "mse_mean": mean(mses),
            "mse_std": stdev(mses) if len(mses) > 1 else 0.0,
            "mae_mean": mean(maes),
            "mae_std": stdev(maes) if len(maes) > 1 else 0.0,
        })

    header = f"{'dataset':10s} {'H':>4s} {'model':12s} {'n':>2s} " \
             f"{'MSE mean':>9s} {'MSE std':>8s} {'MAE mean':>9s} {'MAE std':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['dataset']:10s} {r['horizon']:4d} {r['model']:12s} "
              f"{r['n_seeds']:2d} {r['mse_mean']:9.4f} {r['mse_std']:8.4f} "
              f"{r['mae_mean']:9.4f} {r['mae_std']:8.4f}")

    if out_csv:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
