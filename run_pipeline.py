"""
run_pipeline.py -- ONE command runs the whole project end-to-end.

    python run_pipeline.py            # download (if needed) -> clean -> features -> models -> outputs
    python run_pipeline.py --force    # re-download the raw NOAA file first

  STEP 1  DATA    src/data_download.py  raw NOAA GHCN-Daily file -> data/raw/
  STEP 2  DATA    src/data_clean.py     QC, units, duplicates, full calendar -> data/processed/daily_clean.csv
  STEP 3  DATA    src/features.py       threshold, lags, rolling windows, split -> data/processed/model_table.csv
  STEP 4  METHOD  src/models.py         seasonal + persistence baselines, logistic regression, random forest
  STEP 5  OUTPUT  src/evaluate.py       metrics, figures, interpretability -> outputs/
  (all stages)    src/carbon.py         CodeCarbon energy/emissions -> outputs/emissions.csv
"""
import json
import sys

from src import config
from src.carbon import track
from src.data_clean import clean
from src.data_download import download
from src.evaluate import write_all
from src.features import FEATURES, build_features
from src.models import GRIDS, SeasonalBaseline, best_f1_threshold, persistence_predict, tune_and_fit


def main():
    # ============================ DATA ============================
    with track("preprocessing"):
        download(force="--force" in sys.argv)            # STEP 1
        daily = clean()                                   # STEP 2
        table, thr = build_features(daily)                # STEP 3

    train = table[table.split == "train"]
    val = table[table.split == "val"]
    test = table[table.split == "test"]
    X_tr, y_tr = train[FEATURES], train["y"]
    X_va, y_va = val[FEATURES], val["y"]

    # ============================ METHOD ==========================
    with track("model_tuning"):                           # STEP 4a: small grids, scored on validation
        tuned = {name: tune_and_fit(name, X_tr, y_tr, X_va, y_va) for name in GRIDS}

    with track("final_training"):                         # STEP 4b: refit frozen settings on train
        models, thresholds = {}, {}
        for name, (_, params, thr_, _) in tuned.items():
            models[name] = GRIDS[name][0](**params).fit(X_tr, y_tr)
            thresholds[name] = thr_
        seasonal = SeasonalBaseline().fit(train)
        thresholds["seasonal_baseline"] = best_f1_threshold(y_va, seasonal.predict_proba(val))
        thresholds["persistence"] = 0.5                   # already 0/1

    with track("batch_inference"):                        # STEP 4c: predict val + test once
        def predict_all(df):
            p = {"seasonal_baseline": seasonal.predict_proba(df),
                 "persistence": persistence_predict(df).astype(float)}
            for name, m in models.items():
                p[name] = m.predict_proba(df[FEATURES])[:, 1]
            return p
        probs_val, probs_test = predict_all(val), predict_all(test)

    # ============================ OUTPUT ==========================
    import pandas as pd
    tuning_log = pd.concat([t[3] for t in tuned.values()], ignore_index=True)
    write_all(test, val, probs_test, probs_val, thresholds, models, FEATURES, tuning_log)   # STEP 5

    (config.OUTPUT_DIR / "run_summary.json").write_text(json.dumps({
        "station": f"{config.STATION_ID} {config.STATION_NAME}",
        "extreme_threshold_C": round(thr, 2), "extreme_threshold_F": round(thr * 9 / 5 + 32, 1),
        "rows": {s: int((table.split == s).sum()) for s in ["train", "val", "test"]},
        "events": {s: int(table.loc[table.split == s, "y"].sum()) for s in ["train", "val", "test"]},
        "selected_params": {n: {k: str(v) for k, v in t[1].items()} for n, t in tuned.items()},
        "decision_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
    }, indent=2))
    print(f"[output] all results written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
