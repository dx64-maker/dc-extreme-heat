"""
[COMPONENT 3: OUTPUT -- evaluation tables, figures, interpretability]

Everything is written to outputs/ :
  metrics_test.csv / metrics_val.csv  AP, Brier, precision, recall, F1, confusion counts, event rate
  metrics_onset_test.csv              same metrics on ONSET days only (first extreme after a non-extreme day)
  yearly_test.csv                     per-year AP / F1 / event count
  confusion_matrices_test.txt
  predictions_test.csv                per-day probabilities and 0/1 decisions
  tuning_log.csv                      every grid setting tried and its validation AP
  logreg_coefficients.csv             standardized logistic-regression coefficients
  rf_permutation_importance.csv       validation-set permutation importance (random forest)
  case_examples.csv                   selected hits / misses with the preceding 3 days of weather
  fig_pr_curve.png, fig_reliability.png, fig_yearly_ap.png, fig_importance.png

(The assignment does not require analysing these outputs; they are produced
so later stages of the project can do so.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score)

from src import config


def score(y, prob, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y)) > 1
    return {
        "n": len(y), "events": int(np.sum(y)), "event_rate": float(np.mean(y)),
        "AP": average_precision_score(y, prob) if has_both else np.nan,
        "Brier": brier_score_loss(y, prob),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def metrics_table(df, probs, thresholds) -> pd.DataFrame:
    rows = []
    for name, p in probs.items():
        rows.append({"model": name, "threshold": thresholds[name],
                     **score(df["y"].to_numpy(), p, (p >= thresholds[name]).astype(int))})
    return pd.DataFrame(rows)


def write_all(test, val, probs_test, probs_val, thresholds, models, features, tuning_log):
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---- headline metrics
    mt = metrics_table(test, probs_test, thresholds)
    mv = metrics_table(val, probs_val, thresholds)
    mt.to_csv(out / "metrics_test.csv", index=False)
    mv.to_csv(out / "metrics_val.csv", index=False)
    print("[output] TEST metrics:\n" + mt[["model", "AP", "Brier", "precision", "recall", "F1",
                                          "TP", "FP", "FN", "event_rate"]].round(3).to_string(index=False))

    # ---- onset days: target extreme AND previous day not extreme  (+ all non-extreme days,
    #      so precision is still meaningful): evaluate on days whose previous day was NOT extreme
    onset_mask = (test["prev_extreme"] == 0).to_numpy()
    mo = metrics_table(test[onset_mask], {k: v[onset_mask] for k, v in probs_test.items()}, thresholds)
    mo.insert(1, "subset", "previous day not extreme (onset events)")
    mo.to_csv(out / "metrics_onset_test.csv", index=False)

    # ---- confusion matrices text
    with open(out / "confusion_matrices_test.txt", "w") as fh:
        for _, r in mt.iterrows():
            fh.write(f"{r['model']} (threshold {r['threshold']:.3f})\n"
                     f"              pred 0   pred 1\n"
                     f"  actual 0  {r['TN']:7d}  {r['FP']:7d}\n"
                     f"  actual 1  {r['FN']:7d}  {r['TP']:7d}\n\n")

    # ---- year-by-year
    rows = []
    for yr, idx in test.groupby("year").indices.items():
        y = test["y"].to_numpy()[idx]
        for name, p in probs_test.items():
            s = score(y, p[idx], (p[idx] >= thresholds[name]).astype(int))
            rows.append({"year": yr, "model": name, "events": s["events"], "AP": s["AP"], "F1": s["F1"]})
    yearly = pd.DataFrame(rows)
    yearly.to_csv(out / "yearly_test.csv", index=False)

    # ---- per-day predictions
    pred = test[["target_tmax", "y", "prev_extreme"]].copy()
    for name, p in probs_test.items():
        pred[f"prob_{name}"] = p
        pred[f"pred_{name}"] = (p >= thresholds[name]).astype(int)
    pred.to_csv(out / "predictions_test.csv")
    tuning_log.to_csv(out / "tuning_log.csv", index=False)

    # ---- interpretability: LR standardized coefficients
    lr = models["logistic_regression"]
    names = lr.named_steps["impute"].get_feature_names_out(features)
    coefs = pd.DataFrame({"feature": names, "std_coef": lr.named_steps["clf"].coef_[0]}) \
        .sort_values("std_coef", key=np.abs, ascending=False)
    coefs.to_csv(out / "logreg_coefficients.csv", index=False)

    # ---- interpretability: RF permutation importance on VALIDATION set
    rf = models["random_forest"]
    pi = permutation_importance(rf, val[features], val["y"], scoring="average_precision",
                                n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": features, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(out / "rf_permutation_importance.csv", index=False)

    # ---- case examples: a few hits and misses for the RF, with recent weather
    cols = ["target_tmax", "tmax_lag1", "tmax_lag2", "tmax_lag3", "tmin_lag1", "prcp_lag1", "prev_extreme"]
    ce = test[cols + ["y"]].copy()
    ce["prob_rf"] = probs_test["random_forest"]
    ce["pred_rf"] = (ce["prob_rf"] >= thresholds["random_forest"]).astype(int)
    hits = ce[(ce.y == 1) & (ce.pred_rf == 1)].nlargest(3, "prob_rf").assign(case="correct (hit)")
    misses = ce[(ce.y == 1) & (ce.pred_rf == 0)].nsmallest(3, "prob_rf").assign(case="missed event")
    falarm = ce[(ce.y == 0) & (ce.pred_rf == 1)].nlargest(3, "prob_rf").assign(case="false alarm")
    pd.concat([hits, misses, falarm]).to_csv(out / "case_examples.csv")

    _figures(test, probs_test, yearly, coefs, imp)
    return mt


def _figures(test, probs, yearly, coefs, imp):
    out = config.OUTPUT_DIR
    y = test["y"].to_numpy()
    prob_models = [m for m in probs if m != "persistence"]

    # precision-recall curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in prob_models:
        p, r, _ = precision_recall_curve(y, probs[m])
        ax.step(r, p, where="post", label=f"{m} (AP={average_precision_score(y, probs[m]):.2f})")
    pp = probs["persistence"]
    ax.scatter(recall_score(y, pp), precision_score(y, pp, zero_division=0), color="k", zorder=5,
               label="persistence rule")
    ax.axhline(y.mean(), ls=":", color="grey", label=f"event rate={y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test set (2021-2025) precision-recall")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_pr_curve.png", dpi=150); plt.close(fig)

    # reliability plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    for m in prob_models:
        fr, mp = calibration_curve(y, probs[m], n_bins=8, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=m)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Reliability (test)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_reliability.png", dpi=150); plt.close(fig)

    # year-by-year AP
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in prob_models:
        d = yearly[yearly.model == m]
        ax.plot(d.year, d.AP, marker="o", label=m)
    ev = yearly.drop_duplicates("year")
    ax2 = ax.twinx(); ax2.bar(ev.year, ev.events, alpha=0.2, color="grey"); ax2.set_ylabel("events")
    ax.set(xlabel="Year", ylabel="Average precision", title="Year-by-year test AP")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_yearly_ap.png", dpi=150); plt.close(fig)

    # importance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    c = coefs.head(12).iloc[::-1]
    axes[0].barh(c.feature, c.std_coef); axes[0].set_title("Logistic regression: standardized coef.")
    i = imp.head(12).iloc[::-1]
    axes[1].barh(i.feature, i.importance_mean, xerr=i.importance_std)
    axes[1].set_title("Random forest: validation permutation importance (AP)")
    fig.tight_layout(); fig.savefig(out / "fig_importance.png", dpi=150); plt.close(fig)
