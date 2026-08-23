# Datasets

FreqNet / FreqNetCD uses the Time-Series-Library (`custom` / ETT) loaders.
Place every CSV under `./dataset/` of the TSLib root.

## Original nine benchmarks

Download from the Time-Series-Library data release and unpack as:

```
dataset/
  ETT-small/ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
  weather/weather.csv
  exchange_rate/exchange_rate.csv
  electricity/electricity.csv
  traffic/traffic.csv
  illness/national_illness.csv
```

Official source: https://github.com/thuml/Time-Series-Library

## Additional datasets

### PM2.5 (Beijing multi-site air quality, Aotizhongxin station)

- 11 variates, 35,064 hourly records.
- Source: UCI Machine Learning Repository, dataset ID 501
  (Liang et al., *Scientific Data*, 2015).
- Prepare:

```bash
python tools/prepare_pm25.py <PRSA_Data.zip or station CSV> ./dataset/pm25/pm25.csv
```

Expected layout after preparation:

```
dataset/pm25/pm25.csv     # columns: date, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, OT
```

`OT` is the PM2.5 concentration (TSLib target-last convention).

### NASDAQ 100 (DA-RNN minute bars)

- 82 variates (81 constituents + NDX index), 40,560 minute bars.
- Source: Qin et al., IJCAI 2017 (DA-RNN), file `nasdaq100_padding.csv`.
- Prepare:

```bash
python tools/prepare_nasdaq.py nasdaq100_padding.csv ./dataset/nasdaq/nasdaq100.csv
```

Expected layout:

```
dataset/nasdaq/nasdaq100.csv     # columns: date, <81 stocks>, OT
```

`OT` is the NDX index. The raw file has no timestamps; the script attaches a
uniform minute grid, which is the standard treatment for this dataset.
