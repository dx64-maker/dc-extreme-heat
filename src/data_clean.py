"""
[COMPONENT 1b: DATA -- refinement / cleaning]

Input : data/raw/USW00013743.csv   (NOAA GHCN-Daily "access" CSV, wide format)
Output: data/processed/daily_clean.csv   one row per calendar day, 1990-2025
        outputs/cleaning_log.txt         counts for every cleaning decision

Cleaning steps (each one is logged):
  1. keep only TMAX, TMIN, PRCP for the configured date range
  2. remove duplicate dates (keep first)
  3. values that FAILED a NOAA quality check (non-blank QFLAG) -> missing
  4. unit conversion per GHCN-Daily docs: TMAX/TMIN tenths of deg C -> deg C,
     PRCP tenths of mm -> mm
  5. physical-range sanity check (anything outside is set to missing)
  6. re-index onto a COMPLETE daily calendar so lag-1 always means "yesterday";
     dates with no record become rows of NaN (they are NOT imputed here)

Also accepts NOAA's long-format by_station CSV (AWS mirror) as a fallback.

Run alone:  python -m src.data_clean
"""
import pandas as pd

from src import config

ELEMENTS = ["TMAX", "TMIN", "PRCP"]


def _read_access_csv(path) -> pd.DataFrame:
    """Wide 'access' format: DATE, TMAX, TMAX_ATTRIBUTES, ... (attributes = 'M,Q,S,T')."""
    df = pd.read_csv(path, low_memory=False, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({"date": pd.to_datetime(df["DATE"].str.strip())})
    qc_failed = {}
    for el in ELEMENTS:
        vals = pd.to_numeric(df.get(el), errors="coerce") if el in df else pd.Series(float("nan"), index=df.index)
        attrs = df.get(f"{el}_ATTRIBUTES", pd.Series("", index=df.index)).fillna("")
        qflag = attrs.str.split(",").str[1].fillna("").str.strip()
        failed = (qflag != "") & vals.notna()
        qc_failed[el] = int(failed.sum())
        out[el] = vals.where(~failed)
    return out, qc_failed


def _read_long_csv(path) -> pd.DataFrame:
    """Long by_station format (no header): ID,DATE,ELEMENT,VALUE,MFLAG,QFLAG,SFLAG,OBSTIME."""
    df = pd.read_csv(path, header=None, dtype=str,
                     names=["ID", "DATE", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBSTIME"])
    df = df[df["ELEMENT"].isin(ELEMENTS)].copy()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    failed = df["QFLAG"].fillna("").str.strip() != ""
    qc_failed = df[failed].groupby("ELEMENT").size().reindex(ELEMENTS, fill_value=0).astype(int).to_dict()
    df.loc[failed, "VALUE"] = float("nan")
    df["date"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    wide = df.pivot_table(index="date", columns="ELEMENT", values="VALUE", aggfunc="first").reset_index()
    for el in ELEMENTS:
        if el not in wide:
            wide[el] = float("nan")
    return wide[["date"] + ELEMENTS], qc_failed


def clean() -> pd.DataFrame:
    log = []
    first_line = config.RAW_FILE.open().readline()
    if "DATE" in first_line.upper():
        raw, qc_failed = _read_access_csv(config.RAW_FILE)
        log.append(f"format: NOAA access CSV (wide)")
    else:
        raw, qc_failed = _read_long_csv(config.RAW_FILE)
        log.append(f"format: NOAA by_station CSV (long)")
    log.append(f"raw rows (all years): {len(raw):,}")

    # 1. date range
    raw = raw[(raw["date"] >= config.START_DATE) & (raw["date"] <= config.END_DATE)]
    log.append(f"rows in {config.START_DATE}..{config.END_DATE}: {len(raw):,}")

    # 2. duplicates
    n_dup = int(raw["date"].duplicated().sum())
    raw = raw.drop_duplicates("date", keep="first")
    log.append(f"duplicate dates removed: {n_dup}")

    # 3. QC flags (already applied while reading) -> log counts
    for el, n in qc_failed.items():
        log.append(f"{el}: values with failed NOAA QC flag set to missing (all years): {n}")

    # 4. units: tenths -> whole units
    for el in ELEMENTS:
        raw[el] = raw[el] / 10.0
    log.append("units: TMAX/TMIN tenths degC -> degC; PRCP tenths mm -> mm")

    # 5. physical-range sanity check
    bounds = {"TMAX": (-40, 50), "TMIN": (-40, 40), "PRCP": (0, 500)}
    for el, (lo, hi) in bounds.items():
        bad = raw[el].notna() & ~raw[el].between(lo, hi)
        raw.loc[bad, el] = float("nan")
        log.append(f"{el}: outside [{lo},{hi}] set to missing: {int(bad.sum())}")

    # 6. complete daily calendar
    full = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    daily = raw.set_index("date").reindex(full)
    daily.index.name = "date"
    log.append(f"complete calendar days: {len(full):,}; days with no record at all: "
               f"{int(daily[ELEMENTS].isna().all(axis=1).sum())}")
    for el in ELEMENTS:
        log.append(f"{el}: missing days after cleaning: {int(daily[el].isna().sum())} "
                   f"({daily[el].isna().mean():.2%})")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(config.PROCESSED_DIR / "daily_clean.csv", index=False)
    (config.OUTPUT_DIR / "cleaning_log.txt").write_text(
        f"Cleaning log for {config.STATION_ID} ({config.STATION_NAME})\n" + "\n".join(log) + "\n")
    print("[data] " + "\n[data] ".join(log))
    return daily


if __name__ == "__main__":
    clean()
