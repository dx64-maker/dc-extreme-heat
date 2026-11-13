# Data dictionary

## Source
NOAA NCEI Global Historical Climatology Network – Daily (GHCN-Daily), station
**USW00013743, Washington Reagan National Airport, Arlington, VA**
(38.84721 N, 77.03454 W, elevation 4.1 m). File: NCEI "access" CSV,
`https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00013743.csv`.
The station is in Virginia, not inside the District; results describe this one
station as a case study for the DC area.

## Raw elements used (`data/raw/USW00013743.csv`)
| Column | Meaning | Raw units | `*_ATTRIBUTES` |
|---|---|---|---|
| `DATE` | observation date | YYYY-MM-DD | – |
| `TMAX` | daily maximum temperature | tenths of °C | `MFLAG,QFLAG,SFLAG` |
| `TMIN` | daily minimum temperature | tenths of °C | `MFLAG,QFLAG,SFLAG` |
| `PRCP` | daily precipitation | tenths of mm | `MFLAG,QFLAG,SFLAG,OBS_TIME` |

A non-blank **QFLAG** (2nd attribute) means the value failed a NOAA quality check;
these values are treated as missing. All other columns in the raw file are ignored.

## Cleaned daily table (`data/processed/daily_clean.csv`)
One row per calendar day, 1990-01-01 to 2025-12-31 (complete calendar; days with no
record are NaN rows).
| Column | Units |
|---|---|
| `date` | date |
| `TMAX`, `TMIN` | °C |
| `PRCP` | mm |

## Model table (`data/processed/model_table.csv`)
One row per **target day T** in May–September with observed TMAX.

| Column | Definition | Model input? |
|---|---|---|
| `y` | 1 if TMAX(T) > threshold (P95 of training-period May–Sep TMAX) | label |
| `target_tmax` | TMAX(T), °C | no (evaluation only) |
| `tmax_lag1..3`, `tmin_lag1..3`, `prcp_lag1..3` | value on day T-1, T-2, T-3 | yes |
| `tmax_mean7`, `tmax_max7`, `tmin_mean7` | mean / max over T-7..T-1 (≥4 valid days) | yes |
| `prcp_sum7` | total precipitation T-7..T-1 | yes |
| `tmax_change1` | TMAX(T-1) − TMAX(T-2) | yes |
| `dtr_lag1` | TMAX(T-1) − TMIN(T-1) (diurnal range) | yes |
| `n_extreme7` | number of extreme days in T-7..T-1 | yes |
| `doy_sin`, `doy_cos` | seasonal cycle of day-of-year of T | yes |
| `month` | month of T | seasonal baseline only |
| `prev_extreme` | 1 if day T-1 was extreme | persistence rule / onset analysis |
| `year`, `split` | year of T; train 1990–2015, val 2016–2020, test 2021–2025 | no |

Missing predictors are median-imputed (training medians) and given 0/1 missingness
indicator columns inside the scikit-learn pipeline.
