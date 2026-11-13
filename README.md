# Next-day extreme-heat prediction for the Washington, DC area

DSAN 5550 project: can simple ML models (logistic regression, random forest)
predict **next-day extreme heat** at Reagan National Airport (Arlington, VA) better
than seasonal and persistence baselines, at low computational and carbon cost?

This repository is the **end-to-end working pipeline** for the intermediate check-in.
One command runs it from the raw NOAA file to evaluation outputs.

```bash
pip install -r requirements.txt
python run_pipeline.py          # ~30 s on a laptop
```

---

## Where each component is

| Component | File | What it does | Writes |
|---|---|---|---|
| **DATA: acquisition** | `src/data_download.py` | Downloads the raw GHCN-Daily record for station USW00013743 from NOAA NCEI (skipped if `data/raw/USW00013743.csv` already exists) | `data/raw/USW00013743.csv`, `data/raw/DOWNLOAD_LOG.txt` |
| **DATA: refinement** | `src/data_clean.py` | Keeps TMAX/TMIN/PRCP for 1990–2025; removes duplicate dates; sets values that failed NOAA QC to missing; converts tenths → °C / mm; physical-range check; builds a complete daily calendar | `data/processed/daily_clean.csv`, `outputs/cleaning_log.txt` |
| **DATA: features & split** | `src/features.py` | Extreme threshold = P95 of **training-period** May–Sep TMAX; label = next day exceeds it; 1–3-day lags, trailing 7-day summaries (all ending at T-1), seasonal features; chronological split by target year | `data/processed/model_table.csv` |
| **METHOD** | `src/models.py` | Seasonal (monthly-frequency) baseline, persistence rule, logistic regression (standardized), random forest. Median imputation + missingness indicators fit on training data inside each pipeline. Small grid tuned on validation AP; decision threshold = best validation F1, then frozen | – |
| **OUTPUT** | `src/evaluate.py` | Test-set AP, Brier, precision/recall/F1, confusion matrices, onset-day subset, year-by-year scores, PR curve, reliability plot, LR coefficients, RF permutation importance, example hits/misses | `outputs/*` (listed below) |
| **Carbon cost** | `src/carbon.py` | CodeCarbon tracker around each stage (preprocessing, tuning, final training, batch inference); appends, so all runs accumulate | `outputs/emissions.csv`, `outputs/runtime_log.csv` |
| **Orchestration** | `run_pipeline.py` | Runs the steps above in order, labelled `# DATA / # METHOD / # OUTPUT` | `outputs/run_summary.json` |
| Settings | `src/config.py` | Station, dates, percentile, split years, paths | – |
| Data docs | `docs/data_dictionary.md` | Every raw and derived column | – |

Each module can also be run on its own, e.g. `python -m src.data_clean`.

## Data

* **Source:** NOAA NCEI GHCN-Daily, station **USW00013743, Washington Reagan National
  Airport, Arlington, VA** (38.847 N, 77.035 W). The station is in Virginia; results apply
  to this station as a case study of the DC area.
* **Period:** 1990-01-01 to 2025-12-31 (the raw file runs from 1936, and the extra years are ignored).
* **Split (by target-day year):** train 1990–2015 · validation 2016–2020 · test 2021–2025.
* The raw file is kept unmodified in `data/raw/`; all cleaning decisions are counted in
  `outputs/cleaning_log.txt`.
* The raw CSV itself is **not** versioned here (12 MB). `python -m src.data_download` fetches
  it from NOAA NCEI into `data/raw/`; `run_pipeline.py` does this automatically on first run.
  The exact source URL, station and download timestamp are recorded in `data/raw/DOWNLOAD_LOG.txt`.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `metrics_test.csv`, `metrics_val.csv` | AP, Brier, precision, recall, F1, TP/FP/FN/TN, event rate per model |
| `confusion_matrices_test.txt` | confusion matrices for all four predictors |
| `metrics_onset_test.csv` | metrics on days whose previous day was *not* extreme (onset events) |
| `yearly_test.csv`, `fig_yearly_ap.png` | per-year scores and event counts |
| `fig_pr_curve.png` | precision–recall curves (persistence shown as a point) |
| `fig_reliability.png` | reliability (calibration) plot |
| `logreg_coefficients.csv`, `rf_permutation_importance.csv`, `fig_importance.png` | interpretability |
| `case_examples.csv` | strongest hits, missed events and false alarms with recent weather |
| `predictions_test.csv` | per-day probabilities and decisions |
| `tuning_log.csv` | every grid setting and its validation AP |
| `emissions.csv`, `runtime_log.csv` | CodeCarbon energy (kWh) and CO2e per stage |
| `run_summary.json` | threshold, row/event counts, chosen hyperparameters and thresholds |

## Notes / limitations
* Retrospective experiment: it assumes day T-1's observations are available when the forecast
  for day T is made. Operational use would need to check real data latency.
* No humidity or numerical-weather-prediction inputs; few extreme events per year.
* The threshold is a research definition, not an NWS heat-advisory criterion.
* CodeCarbon estimates (offline mode, US grid average) cover tracked computation only, not
  the hardware's embodied footprint.
