"""
[COMPONENT 1c: DATA -- target definition, features, chronological split]

Input : data/processed/daily_clean.csv
Output: data/processed/model_table.csv   one row per TARGET day (May-Sep)

Framing (retrospective next-day prediction):
  * Each row is a target day T. The label is 1 if TMAX(T) > threshold.
  * Every predictor uses only days T-1 and earlier (lags and rolling windows
    are shifted by one day), plus calendar features of T, which are known
    in advance. No information from day T leaks into the features.
  * threshold = 95th percentile of May-Sep TMAX in the TRAINING years only.
  * Target days with missing TMAX are dropped (never imputed).
  * Missing PREDICTORS are left as NaN here; they are median-imputed with
    missingness indicators INSIDE the model pipeline (fit on training data),
    see models.py.
"""
import numpy as np
import pandas as pd

from src import config


def extreme_threshold(daily: pd.DataFrame) -> float:
    """95th percentile of May-Sep TMAX in training years (training data only)."""
    d = daily
    mask = (d.index.year >= config.TRAIN_YEARS[0]) & (d.index.year <= config.TRAIN_YEARS[1]) \
        & d.index.month.isin(config.WARM_MONTHS)
    return float(np.nanpercentile(d.loc[mask, "TMAX"], config.EXTREME_PERCENTILE))


def split_name(year: int) -> str:
    if config.TRAIN_YEARS[0] <= year <= config.TRAIN_YEARS[1]:
        return "train"
    if config.VAL_YEARS[0] <= year <= config.VAL_YEARS[1]:
        return "val"
    if config.TEST_YEARS[0] <= year <= config.TEST_YEARS[1]:
        return "test"
    return "unused"


def build_features(daily: pd.DataFrame | None = None):
    if daily is None:
        daily = pd.read_csv(config.PROCESSED_DIR / "daily_clean.csv", parse_dates=["date"]).set_index("date")

    thr = extreme_threshold(daily)
    print(f"[data] extreme-heat threshold (train May-Sep P{config.EXTREME_PERCENTILE}): "
          f"{thr:.1f} C = {thr * 9 / 5 + 32:.1f} F")

    f = pd.DataFrame(index=daily.index)
    f["target_tmax"] = daily["TMAX"]
    f["y"] = (daily["TMAX"] > thr).astype(float).where(daily["TMAX"].notna())

    # --- lag features: day T-1, T-2, T-3 (calendar days, thanks to the complete index)
    for lag in (1, 2, 3):
        for el in ("TMAX", "TMIN", "PRCP"):
            f[f"{el.lower()}_lag{lag}"] = daily[el].shift(lag)

    # --- trailing 7-day summaries over T-7..T-1 (shift(1) ends the window BEFORE T)
    past = daily.shift(1)
    roll = past.rolling(7, min_periods=4)
    f["tmax_mean7"] = roll["TMAX"].mean()
    f["tmax_max7"] = roll["TMAX"].max()
    f["tmin_mean7"] = roll["TMIN"].mean()
    f["prcp_sum7"] = roll["PRCP"].sum()
    f["tmax_change1"] = f["tmax_lag1"] - f["tmax_lag2"]
    f["dtr_lag1"] = f["tmax_lag1"] - f["tmin_lag1"]   # diurnal range yesterday
    # count of extreme days in the past 7 days (uses the training threshold)
    f["n_extreme7"] = (past["TMAX"] > thr).astype(float).where(past["TMAX"].notna()) \
        .rolling(7, min_periods=4).sum()

    # --- seasonal calendar features of the target day (known in advance)
    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["month"] = f.index.month

    # --- helper columns used by baselines / evaluation (NOT model inputs)
    f["prev_extreme"] = (daily["TMAX"].shift(1) > thr).astype(int)   # persistence rule
    f["prev_tmax_missing"] = daily["TMAX"].shift(1).isna().astype(int)
    f["year"] = f.index.year
    f["split"] = [split_name(y) for y in f.index.year]

    # keep warm-season target days with an observed target
    n_before = int(f.index.month.isin(config.WARM_MONTHS).sum())
    f = f[f.index.month.isin(config.WARM_MONTHS) & f["y"].notna()].copy()
    f["y"] = f["y"].astype(int)
    print(f"[data] warm-season target days: {n_before:,}; dropped for missing target TMAX: "
          f"{n_before - len(f):,}; kept: {len(f):,}")
    print("[data] rows / events by split:\n" +
          f.groupby("split")["y"].agg(rows="size", events="sum", rate="mean").to_string())

    f.index.name = "date"
    f.reset_index().to_csv(config.PROCESSED_DIR / "model_table.csv", index=False)
    return f, thr


FEATURES = (
    [f"{el}_lag{l}" for l in (1, 2, 3) for el in ("tmax", "tmin", "prcp")]
    + ["tmax_mean7", "tmax_max7", "tmin_mean7", "prcp_sum7",
       "tmax_change1", "dtr_lag1", "n_extreme7", "doy_sin", "doy_cos"]
)


if __name__ == "__main__":
    build_features()
