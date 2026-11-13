"""
[COMPONENT 2: METHOD -- baselines and machine-learning models]

Four predictors of P(extreme heat tomorrow):
  1. SeasonalBaseline   -- training-period event frequency for the target month
  2. persistence rule   -- predict extreme if yesterday was extreme (0/1)
  3. Logistic regression (scikit-learn), standardized inputs, class weighting
  4. Random forest      (scikit-learn), class weighting

Both ML models sit inside a scikit-learn Pipeline whose first step is
SimpleImputer(strategy="median", add_indicator=True). The pipeline is fit on
TRAINING rows only, so the medians and the scaler never see validation/test data.

Tuning: a small grid per model, each candidate fit on train and scored on
validation average precision (AP). The best setting is then frozen. Each
model's decision threshold is chosen to maximize validation F1 and is also
frozen before the test set is touched.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


# ------------------------------------------------------------------ baselines
class SeasonalBaseline:
    """Probability = training event frequency for the target day's month."""

    def fit(self, df: pd.DataFrame):
        self.rate_by_month = df.groupby("month")["y"].mean()
        self.overall = df["y"].mean()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return df["month"].map(self.rate_by_month).fillna(self.overall).to_numpy()


def persistence_predict(df: pd.DataFrame) -> np.ndarray:
    """1 if the day before the target day was extreme, else 0 (missing -> 0)."""
    return df["prev_extreme"].to_numpy()


# ------------------------------------------------------------------ ML models
def make_logreg(C=1.0, class_weight="balanced"):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)),
    ])


def make_rf(max_depth=None, min_samples_leaf=5, class_weight="balanced_subsample"):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight=class_weight, n_jobs=-1, random_state=config.RANDOM_STATE)),
    ])


GRIDS = {
    "logistic_regression": (make_logreg, {"C": [0.01, 0.1, 1.0, 10.0],
                                          "class_weight": [None, "balanced"]}),
    "random_forest": (make_rf, {"max_depth": [4, 8, None],
                                "min_samples_leaf": [5, 20],
                                "class_weight": [None, "balanced_subsample"]}),
}


def best_f1_threshold(y_true, prob) -> float:
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5


def tune_and_fit(name, X_tr, y_tr, X_va, y_va):
    """Small grid search on validation AP; returns (fitted model, params, threshold, log)."""
    factory, grid = GRIDS[name]
    keys = list(grid)
    log, best = [], None
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        model = factory(**params).fit(X_tr, y_tr)
        ap = average_precision_score(y_va, model.predict_proba(X_va)[:, 1])
        log.append({"model": name, **{k: str(v) for k, v in params.items()}, "val_AP": ap})
        if best is None or ap > best[2]:
            best = (model, params, ap)
    model, params, ap = best
    thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
    print(f"[method] {name}: best params {params}  val AP={ap:.3f}  F1-threshold={thr:.3f}")
    return model, params, thr, pd.DataFrame(log)
