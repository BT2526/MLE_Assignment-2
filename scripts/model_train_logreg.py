"""
Logistic Regression challenger training script.

Trains Logistic Regression, saves artefact to model_bank/, logs to MLflow.
model_select.py runs after this (and after model_train_xgb.py) to compare
OOT AUC and write champion.txt.

    python3 model_train_logreg.py --snapshotdate 2024-09-01
"""

import argparse
import glob
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, brier_score_loss, make_scorer,
                              precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
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


def _load_aligned(spark, start, end):
    """Join gold feature store x gold label store, aligned by the 6-month MOB offset."""
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
    print(f"\n--- Logistic Regression training {snapshotdate} (tt_months={tt_months}) ---\n")

    train_date  = datetime.strptime(snapshotdate, "%Y-%m-%d")
    oot_months  = 2
    oot_end     = train_date - timedelta(days=1)
    oot_start   = train_date - relativedelta(months=oot_months)
    tt_end      = oot_start  - timedelta(days=1)
    tt_start    = oot_start  - relativedelta(months=tt_months)
    test_cutoff = tt_end     - relativedelta(months=2)

    cfg = {"train_date": snapshotdate, "tt_start": tt_start, "tt_end": tt_end,
           "oot_start": oot_start, "oot_end": oot_end, "tt_months": tt_months}

    spark = get_spark("model_train_logreg")
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

    # Scaler fit on TRAIN only — same deterministic result as model_train_xgb.py
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte, Xoo = (scaler.transform(X_train),
                     scaler.transform(X_test),
                     scaler.transform(X_oot))

    os.makedirs(MODEL_BANK, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"logreg_challenger_{snapshotdate}_tt{tt_months}m"):
        lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=88)
        lr.fit(Xtr, y_train)

        auc = {k: roc_auc_score(y, lr.predict_proba(X)[:, 1])
               for k, (X, y) in {"train": (Xtr, y_train),
                                  "test":  (Xte, y_test),
                                  "oot":   (Xoo, y_oot)}.items()}
        print("LogReg AUC:", {k: round(v, 4) for k, v in auc.items()})

        # Brier score on OOT — LR is calibrated by construction via log-loss,
        # so no separate calibration step is needed. One Brier value suffices.
        oot_proba = lr.predict_proba(Xoo)[:, 1]
        brier_oot = round(float(brier_score_loss(y_oot, oot_proba)), 4)
        print(f"LogReg Brier (OOT): {brier_oot}")
        mlflow.log_metric("brier_oot", brier_oot)

        clf_metrics = {k: _classification_metrics(y, lr.predict_proba(X)[:, 1])
                       for k, (X, y) in {"train": (Xtr, y_train),
                                          "test":  (Xte, y_test),
                                          "oot":   (Xoo, y_oot)}.items()}
        print("LogReg precision/recall/F1 (oot):", clf_metrics["oot"])

        mlflow.log_params({"max_iter": 1000, "class_weight": "balanced"})
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        for k, v in auc.items():
            mlflow.log_metric(f"auc_{k}", v)
            mlflow.log_metric(f"gini_{k}", _gini(v))
        for k, m in clf_metrics.items():
            for mn, mv in m.items():
                mlflow.log_metric(f"{mn}_{k}", mv)

    # Persist artefact — no calibrator for LR (calibrated by construction via log-loss)
    window_suffix = "" if tt_months == 12 else f"_tt{tt_months}m"
    tag = snapshotdate.replace("-", "_") + window_suffix
    artefact = {
        "model":                    lr,
        "model_version":            f"credit_logreg_{tag}",
        "model_type":               "logreg",
        "feature_cols":             feature_cols,
        "preprocessing_transformers": {"stdscaler": scaler},
        "data_dates":               {k: (v.strftime("%Y-%m-%d")
                                         if isinstance(v, datetime) else v)
                                     for k, v in cfg.items()},
        "results":                  {"auc_train": auc["train"],
                                     "auc_test":  auc["test"],
                                     "auc_oot":   auc["oot"],
                                     "gini_oot":  _gini(auc["oot"]),
                                     "brier_oot_calibrated": brier_oot},
        "hp_params":                {"max_iter": 1000, "class_weight": "balanced"},
        "calibrator":               None,
    }
    path = os.path.join(MODEL_BANK, f"credit_logreg_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"saved LogReg artefact: {path}")

    # ---- Save OOT1 raw scores to the predictions datamart -----------------
    # Same design as model_train_xgb.py: OOT1 scores become the PSI/CSI
    # baseline in model_monitor.py, anchoring drift comparisons to a
    # validated, outcome-verified reference distribution.
    # Logistic Regression does not need calibration — it is calibrated by
    # construction via log-loss optimisation, so raw predict_proba() outputs
    # are already true probabilities. Using raw scores here is consistent
    # with what model_inference.py produces for Logistic Regression in
    # production (where no calibrator is present in the artefact).
    oot_proba_raw = lr.predict_proba(Xoo)[:, 1]
    oot_out = oot[["Customer_ID", "snapshot_date"]].copy()
    oot_out["model_name"] = f"credit_logreg_{tag}"
    oot_out["model_predictions"] = oot_proba_raw
    oot_out["prediction_label"] = (oot_proba_raw >= CLASSIFICATION_THRESHOLD).astype(int)
    oot_pred_dir = f"datamart/gold/model_predictions/credit_logreg_{tag}/"
    os.makedirs(oot_pred_dir, exist_ok=True)
    for snap_date, grp in oot_out.groupby("snapshot_date"):
        snap_str = pd.Timestamp(snap_date).strftime("%Y_%m_%d")
        oot_path = os.path.join(oot_pred_dir,
                                f"credit_logreg_{tag}_predictions_{snap_str}.parquet")
        grp.to_parquet(oot_path, index=False)
        print(f"saved OOT1 predictions: {oot_path} ({len(grp)} rows)")

    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--tt_months", type=int, default=12)
    a = p.parse_args()
    main(a.snapshotdate, a.tt_months)
