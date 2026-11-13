"""
config.py -- All project settings in one place.

Changing a value here (station, years, split boundaries, percentile) changes
the whole pipeline; nothing else in the code hard-codes these choices.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # untouched NOAA downloads
PROCESSED_DIR = ROOT / "data" / "processed"  # cleaned daily table + feature table
OUTPUT_DIR = ROOT / "outputs"              # metrics, figures, predictions, logs

# ---------------------------------------------------------------- DATA
# GHCN-Daily station: Washington Reagan National Airport, Arlington, VA
STATION_ID = "USW00013743"
STATION_NAME = "Washington Reagan National Airport (Arlington, VA)"
# NOAA NCEI "access" CSV: one row per day, wide format, values in tenths.
NOAA_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
    f"access/{STATION_ID}.csv"
)
RAW_FILE = RAW_DIR / f"{STATION_ID}.csv"

START_DATE = "1990-01-01"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------- target definition
WARM_MONTHS = [5, 6, 7, 8, 9]     # May-September
EXTREME_PERCENTILE = 95           # of training-period May-Sep TMAX

# ---------------------------------------------------------------- chronological split
# Split is by the TARGET day's year (the day being predicted).
TRAIN_YEARS = (1990, 2015)
VAL_YEARS = (2016, 2020)
TEST_YEARS = (2021, 2025)

RANDOM_STATE = 5550
