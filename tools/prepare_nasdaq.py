"""Prepare the NASDAQ 100 stock dataset (DA-RNN, Qin et al., IJCAI 2017)
for the Time-Series-Library 'custom' loader.

Input: nasdaq100_padding.csv (81 constituent stocks + NDX index, minute bars,
105 trading days, 2016-07-26 .. 2016-12-22).

The raw file has no timestamp column, so synthetic minute-spaced timestamps
are attached (TSLib only uses the date column to derive time features; a
uniform minute grid is the standard treatment for this dataset). The NDX
index is renamed to OT and moved last (TSLib target convention).

Usage:
  python prepare_nasdaq.py <nasdaq100_padding.csv> <out_csv>
"""
import sys

import pandas as pd


def main(src, out_csv):
    df = pd.read_csv(src)
    assert "NDX" in df.columns, "expected NDX index column"

    stocks = [c for c in df.columns if c != "NDX"]
    out = df[stocks].copy()
    out["OT"] = df["NDX"]

    # Uniform minute grid starting at the first trading day's open (09:30 ET).
    dates = pd.date_range("2016-07-26 09:30", periods=len(out), freq="min")
    out.insert(0, "date", dates)

    out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Rows: {len(out)}, Columns: {len(out.columns)}")
    print(f"Channels (C): {len(out.columns) - 1}")
    print(f"Stocks: {len(stocks)}, target: NDX -> OT")
    print(f"NaNs: {int(out.isna().sum().sum())}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
