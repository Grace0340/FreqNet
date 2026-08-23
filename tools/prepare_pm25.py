"""Prepare the Beijing Multi-Site Air-Quality (UCI ID 501) PM2.5 dataset
for the Time-Series-Library 'custom' loader.

Uses the Aotizhongxin station (the station most commonly used in the
PM2.5-forecasting literature). Output CSV columns:
  date, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, OT
where OT is the PM2.5 concentration (TSLib convention: target last, named OT).

Missing values are filled by time interpolation, then forward/backward fill.

Usage:
  python prepare_pm25.py <beijing_multisite.zip or extracted csv> <out_csv>
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

STATION = "Aotizhongxin"
FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET = "PM2.5"


def load_station_csv(src: Path) -> pd.DataFrame:
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            name = next(n for n in z.namelist()
                        if STATION in n and n.endswith(".csv"))
            # The UCI archive nests a second zip in some mirrors; handle both.
            with z.open(name) as f:
                return pd.read_csv(f)
    return pd.read_csv(src)


def main(src, out_csv):
    src = Path(src)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            inner_zips = [n for n in z.namelist() if n.endswith(".zip")]
        if inner_zips:
            import io
            with zipfile.ZipFile(src) as z:
                inner = zipfile.ZipFile(io.BytesIO(z.read(inner_zips[0])))
                name = next(n for n in inner.namelist()
                            if STATION in n and n.endswith(".csv"))
                df = pd.read_csv(inner.open(name))
        else:
            df = load_station_csv(src)
    else:
        df = load_station_csv(src)

    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("date").reset_index(drop=True)

    cols = FEATURES + [TARGET]
    out = df[["date"] + cols].copy()
    out = out.set_index("date")
    n_missing = out.isna().sum()
    out = out.interpolate(method="time").ffill().bfill()
    out = out.rename(columns={TARGET: "OT"}).reset_index()

    out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Station: {STATION}")
    print(f"Rows: {len(out)}, Columns: {list(out.columns)}")
    print(f"Channels (C): {len(out.columns) - 1}")
    print(f"Date range: {out['date'].min()} .. {out['date'].max()}")
    print("Missing filled per column:")
    print(n_missing.to_string())
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
