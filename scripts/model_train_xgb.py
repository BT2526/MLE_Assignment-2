"""
XGBoost champion training script.

Trains XGBoost with RandomizedSearchCV + Platt calibration, saves artefact to
model_bank/, logs to MLflow. model_select.py runs after this (and after
model_train_logreg.py) to compare OOT AUC and write champion.txt.

    python3 model_train_xgb.py --snapshotdate 2024-09-01
"""

import argparse
import glob
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, f1_score, make_scorer,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow

from pipeline_common import get_spark

MODEL_BANK = "model_bank/"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-ui:5000")
EXPERIMENT_NAME = "credit_default_prediction"
CLASSIFICATION_THRESHOLD = 0.5


def _gini(auc):
    return round(2 * auc - 1, 4)


def _classification_metrics(y_true, y_proba, threshold=CLASSIFICATION_THRESHOLD):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _plot_reliability_diagram(y_true, proba_raw, proba_cal, title, out_path, n_bins=10):
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, proba_raw, n_bins=n_bins, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, proba_cal,  n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", color="#dc2626", label="raw XGBoost")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", color="#2563eb", label="Platt-calibrated")
    ax.set_xlabel("mean predicted probability (per bin)")
    ax.set_ylabel("observed default rate (per bin)")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def _load_aligned(spark, start, end):
    """Join gold feature store x gold label store, aligned by the 6-month MOB offset.
    [start, end] are applied to the LABEL date, consistent with model_train.py."""
    from pyspark.sql.functions import col, add_months
    feat_files  = glob.glob("datamart/gold/feature_store/*.parquet")
    label_files = glob.glob("datamart/gold/label_store/*.parquet")
    feats  = spark.read.parquet(*feat_files)
    labels = spark.read.parquet(*label_files)
    feats  = feats.withColumn("label_date", add_months(col("snapshot_date"), 6))
    labels = (labels.select("Customer_ID",
                             col("snapshot_date").alias("label_date"), "label")
              .filter((col("label_date") >= start) & (col("label_date") <= end)))
    df = feats.join(labels, on=["Customer_ID", "label_date"], how="inner")
    df = df.drop("snapshot_date").withColumnRenamed("label_date", "snapshot_date")
    return df.toPandas()


def main(snapshotdate, tt_months=12):
    print(f"\n--- XGBoost training {snapshotdate} (tt_months={tt_months}) ---\n")

    train_date  = datetime.strptime(snapshotdate, "%Y-%m-%d")
    oot_months  = 2
    oot_end     = train_date - timedelta(days=1)
    oot_start   = train_date - relativedelta(months=oot_months)
    tt_end      = oot_start  - timedelta(days=1)
    tt_start    = oot_start  - relativedelta(months=tt_months)
    test_cutoff = tt_end     - relativedelta(months=2)

    cfg = {"train_date": snapshotdate, "tt_start": tt_start, "tt_end": tt_end,
           "oot_start": oot_start, "oot_end": oot_end, "tt_months": tt_months}

    spark = get_spark("model_train_xgb")
    data  = _load_aligned(spark, tt_start.strftime("%Y-%m-%d"),
                          oot_end.strftime("%Y-%m-%d"))
    spark.stop()
    print(f"aligned rows: {len(data)}  bad rate: {round(data['label'].mean(), 3)}")

    feature_cols = [c for c in data.columns
                    if c not in ("Customer_ID", "snapshot_date", "label",
                                 "loan_id", "label_def")]
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

    oot = data[(data.snapshot_date >= oot_start) & (data.snapshot_date <= oot_end)]
    tt  = data[(data.snapshot_date >= tt_start)  & (data.snapshot_date <= tt_end)
               ].sort_values("snapshot_date")

    train_df = tt[tt.snapshot_date <= test_cutoff]
    test_df  = tt[tt.snapshot_date  > test_cutoff]

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test,  y_test  = test_df[feature_cols],  test_df["label"]
    X_oot,   y_oot   = oot[feature_cols],      oot["label"]

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte, Xoo = (scaler.transform(X_train),
                     scaler.transform(X_test),
                     scaler.transform(X_oot))

    os.makedirs(MODEL_BANK, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"xgb_champion_{snapshotdate}_tt{tt_months}m"):
        param_dist = {
            "n_estimators":    [50, 100, 200],
            "max_depth":       [2, 3, 4],
            "learning_rate":   [0.01, 0.05, 0.1],
            "subsample":       [0.6, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
            "gamma":           [0, 0.1, 0.3],
            "min_child_weight":[1, 3, 5],
            "reg_alpha":       [0, 0.1, 1],
            "reg_lambda":      [1, 1.5, 2],
        }
        search = RandomizedSearchCV(
            xgb.XGBClassifier(eval_metric="logloss", random_state=88),
            param_distributions=param_dist,
            scoring=make_scorer(roc_auc_score, response_method="predict_proba"),
            n_iter=40, cv=TimeSeriesSplit(n_splits=3),
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(Xtr, y_train)
        best = search.best_estimator_

        auc = {k: roc_auc_score(y, best.predict_proba(X)[:, 1])
               for k, (X, y) in {"train": (Xtr, y_train),
                                  "test":  (Xte, y_test),
                                  "oot":   (Xoo, y_oot)}.items()}
        print("XGB AUC:", {k: round(v, 4) for k, v in auc.items()})

        clf_metrics = {k: _classification_metrics(y, best.predict_proba(X)[:, 1])
                       for k, (X, y) in {"train": (Xtr, y_train),
                                          "test":  (Xte, y_test),
                                          "oot":   (Xoo, y_oot)}.items()}
        print("XGB precision/recall/F1 (oot):", clf_metrics["oot"])

        mlflow.log_params(search.best_params_)
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        for k, v in auc.items():
            mlflow.log_metric(f"auc_{k}", v)
            mlflow.log_metric(f"gini_{k}", _gini(v))
        for k, m in clf_metrics.items():
            for mn, mv in m.items():
                mlflow.log_metric(f"{mn}_{k}", mv)

        # Platt calibration — fit on test only, OOT stays untouched
        calibrator = CalibratedClassifierCV(best, method="sigmoid", cv="prefit")
        calibrator.fit(Xte, y_test)

        oot_proba_raw = best.predict_proba(Xoo)[:, 1]
        oot_proba_cal = calibrator.predict_proba(Xoo)[:, 1]
        brier_raw = round(float(brier_score_loss(y_oot, oot_proba_raw)), 4)
        brier_cal = round(float(brier_score_loss(y_oot, oot_proba_cal)), 4)
        auc_cal_oot = round(float(roc_auc_score(y_oot, oot_proba_cal)), 4)
        print(f"XGB calibration (oot) - Brier raw: {brier_raw}  "
              f"Brier calibrated: {brier_cal}  AUC calibrated: {auc_cal_oot}")

        mlflow.log_param("calibration_method", "platt_sigmoid")
        mlflow.log_param("calibration_fit_split", "test")
        mlflow.log_metric("brier_oot_raw", brier_raw)
        mlflow.log_metric("brier_oot_calibrated", brier_cal)
        mlflow.log_metric("auc_oot_calibrated", auc_cal_oot)

        os.makedirs("reports/calibration", exist_ok=True)
        reliability_png = (f"reports/calibration/"
                           f"xgb_{snapshotdate.replace('-','_')}_reliability.png")
        _plot_reliability_diagram(
            y_oot, oot_proba_raw, oot_proba_cal,
            title=f"XGBoost calibration (OOT) - {snapshotdate}",
            out_path=reliability_png,
        )
        mlflow.log_artifact(reliability_png)
        print(f"saved reliability diagram: {reliability_png}")

    # Persist artefact
    window_suffix = "" if tt_months == 12 else f"_tt{tt_months}m"
    tag = snapshotdate.replace("-", "_") + window_suffix
    artefact = {
        "model":                    best,
        "model_version":            f"credit_xgb_{tag}",
        "model_type":               "xgb",
        "feature_cols":             feature_cols,
        "preprocessing_transformers": {"stdscaler": scaler},
        "data_dates":               {k: (v.strftime("%Y-%m-%d")
                                         if isinstance(v, datetime) else v)
                                     for k, v in cfg.items()},
        "results":                  {"auc_train": auc["train"],
                                     "auc_test":  auc["test"],
                                     "auc_oot":   auc["oot"],
                                     "gini_oot":  _gini(auc["oot"]),
                                     "brier_oot_raw":        brier_raw,
                                     "brier_oot_calibrated": brier_cal},
        "hp_params":                search.best_params_,
        "calibrator":               calibrator,
    }
    path = os.path.join(MODEL_BANK, f"credit_xgb_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"saved XGBoost artefact: {path}")

    # ---- Save OOT1 calibrated scores to the predictions datamart -----------
    # This makes the OOT1 score distribution the PSI/CSI baseline in
    # model_monitor.py, replacing the first arbitrary production month.
    # OOT1 is the validated, outcome-verified reference: we know the model
    # performed well on it (AUC = ' + str(round(auc["oot"], 4)) + '), making it a principled
    # anchor for all future drift comparisons. model_monitor.py's loop picks
    # up the earliest prediction file as its baseline — by saving OOT1 scores
    # here (which predate any production inference month) that earliest file
    # will always be an OOT1 month, not the first arbitrary production month.
    # Scores are calibrated (Platt-scaled) to be consistent with what
    # model_inference.py produces in production — PSI baseline and production
    # scores must use the same scale for the comparison to be meaningful.
    oot_out = oot[["Customer_ID", "snapshot_date"]].copy()
    oot_out["model_name"] = f"credit_xgb_{tag}"
    oot_out["model_predictions"] = oot_proba_cal
    oot_out["prediction_label"] = (oot_proba_cal >= CLASSIFICATION_THRESHOLD).astype(int)
    oot_pred_dir = f"datamart/gold/model_predictions/credit_xgb_{tag}/"
    os.makedirs(oot_pred_dir, exist_ok=True)
    # Save one parquet per OOT snapshot month so model_monitor.py can read
    # them exactly like regular monthly inference outputs.
    for snap_date, grp in oot_out.groupby("snapshot_date"):
        snap_str = pd.Timestamp(snap_date).strftime("%Y_%m_%d")
        oot_path = os.path.join(oot_pred_dir,
                                f"credit_xgb_{tag}_predictions_{snap_str}.parquet")
        grp.to_parquet(oot_path, index=False)
        print(f"saved OOT1 predictions: {oot_path} ({len(grp)} rows)")

    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--tt_months", type=int, default=12)
    a = p.parse_args()
    main(a.snapshotdate, a.tt_months)
