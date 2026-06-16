"""
Model training (AutoML) entry point.

Trains TWO models on an aligned (feature store x label store) dataset and writes
both artefacts to the model bank, logging every run to MLflow (file backend).

    Champion  : XGBoost  (RandomizedSearchCV hyper-parameter tuning)
    Challenger: Logistic Regression  (interpretable linear baseline)

    python3 model_train.py --snapshotdate 2024-09-01

Train / OOT split is driven by the snapshot date, mirroring the Main notebook:
    train+test window : 12 months ending 2 months before the train date
    OOT window        : the final 2 months before the train date (held out)

ANTI-LEAKAGE: features come from the gold feature store (all as-of application),
the StandardScaler is fit on TRAIN only, and the OOT window is strictly later
than train/test so no future information leaks into model selection.
"""

import argparse
import glob
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb

import mlflow

from pipeline_common import get_spark

MODEL_BANK = "model_bank/"
MLFLOW_DB = "mlflow.db"
MLFLOW_ARTIFACTS = "mlruns"


def _load_aligned(spark, start, end):
    """Join gold feature store x gold label store into pandas, aligned by the
    6-month month-on-book offset.

    A feature row is observed at the APPLICATION month (mob=0); its label is
    observed 6 months later (mob=6).  We therefore add 6 months to each feature
    row's snapshot_date and join that to the label's snapshot_date on the same
    Customer_ID.  The [start, end] window is applied to the LABEL date so that
    train/OOT splits line up with realised outcomes.
    """
    import glob as _glob
    from pyspark.sql.functions import col, add_months

    feat_files = _glob.glob("datamart/gold/feature_store/*.parquet")
    label_files = _glob.glob("datamart/gold/label_store/*.parquet")
    feats = spark.read.parquet(*feat_files)
    labels = spark.read.parquet(*label_files)

    # Map each feature row forward to the month its label is observed (mob=6).
    feats = feats.withColumn("label_date", add_months(col("snapshot_date"), 6))

    labels = (labels.select("Customer_ID",
                            col("snapshot_date").alias("label_date"), "label")
              .filter((col("label_date") >= start) & (col("label_date") <= end)))

    df = feats.join(labels, on=["Customer_ID", "label_date"], how="inner")
    # Keep the label_date as the modelling snapshot for OOT windowing.
    df = df.drop("snapshot_date").withColumnRenamed("label_date", "snapshot_date")
    return df.toPandas()


def _gini(auc):
    return round(2 * auc - 1, 4)


def main(snapshotdate):
    print(f"\n--- model training {snapshotdate} ---\n")

    # ---- window config (mirrors Main notebook) ----
    train_date = datetime.strptime(snapshotdate, "%Y-%m-%d")
    oot_months, tt_months = 2, 12
    oot_end = train_date - timedelta(days=1)
    oot_start = train_date - relativedelta(months=oot_months)
    tt_end = oot_start - timedelta(days=1)
    tt_start = oot_start - relativedelta(months=tt_months)
    cfg = {"train_date": snapshotdate, "tt_start": tt_start, "tt_end": tt_end,
           "oot_start": oot_start, "oot_end": oot_end}
    pprint.pprint(cfg)

    spark = get_spark("model_train")
    data = _load_aligned(spark, tt_start.strftime("%Y-%m-%d"),
                         oot_end.strftime("%Y-%m-%d"))
    spark.stop()
    print(f"aligned rows: {len(data)}  bad rate: {round(data['label'].mean(), 3)}")

    feature_cols = [c for c in data.columns
                    if c not in ("Customer_ID", "snapshot_date", "label",
                                 "loan_id", "label_def")]
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

    oot = data[(data.snapshot_date >= oot_start) & (data.snapshot_date <= oot_end)]
    tt = data[(data.snapshot_date >= tt_start)
              & (data.snapshot_date <= tt_end)].sort_values("snapshot_date")

    X_oot, y_oot = oot[feature_cols], oot["label"]

    # Chronological split: the last 2 months of the train/test window become the
    # test set; everything earlier is training.  No shuffling - the data stays in
    # time order so the test set is strictly later than train (no temporal leakage)
    # and TimeSeriesSplit below sees correctly ordered rows.
    test_cutoff = tt_end - relativedelta(months=2)
    train_df = tt[tt.snapshot_date <= test_cutoff]
    test_df = tt[tt.snapshot_date > test_cutoff]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]
    print(f"train {len(X_train)}  test {len(X_test)}  oot {len(X_oot)}")

    # ---- scaler fit on TRAIN only (no leakage) ----
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte, Xoo = (scaler.transform(X_train),
                     scaler.transform(X_test),
                     scaler.transform(X_oot))

    os.makedirs(MODEL_BANK, exist_ok=True)
    os.makedirs(MLFLOW_ARTIFACTS, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(MLFLOW_DB)}")
    mlflow.set_experiment("credit_default_prediction")

    results = {}

    # =====================================================================
    # CHAMPION: XGBoost with RandomizedSearchCV
    # =====================================================================
    with mlflow.start_run(run_name=f"xgb_champion_{snapshotdate}"):
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2],
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
                                  "test": (Xte, y_test),
                                  "oot": (Xoo, y_oot)}.items()}
        print("XGB AUC:", {k: round(v, 4) for k, v in auc.items()})

        mlflow.log_params(search.best_params_)
        for k, v in auc.items():
            mlflow.log_metric(f"auc_{k}", v)
            mlflow.log_metric(f"gini_{k}", _gini(v))
        results["xgb"] = {"auc": auc, "model": best, "params": search.best_params_}

    # =====================================================================
    # CHALLENGER: Logistic Regression
    # =====================================================================
    with mlflow.start_run(run_name=f"logreg_challenger_{snapshotdate}"):
        lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=88)
        lr.fit(Xtr, y_train)
        auc = {k: roc_auc_score(y, lr.predict_proba(X)[:, 1])
               for k, (X, y) in {"train": (Xtr, y_train),
                                  "test": (Xte, y_test),
                                  "oot": (Xoo, y_oot)}.items()}
        print("LogReg AUC:", {k: round(v, 4) for k, v in auc.items()})

        mlflow.log_params({"max_iter": 1000, "class_weight": "balanced"})
        for k, v in auc.items():
            mlflow.log_metric(f"auc_{k}", v)
            mlflow.log_metric(f"gini_{k}", _gini(v))
        results["logreg"] = {"auc": auc, "model": lr, "params": {}}

    # =====================================================================
    # Persist both artefacts to the model bank
    # =====================================================================
    tag = snapshotdate.replace("-", "_")
    for name, key in [("credit_xgb", "xgb"), ("credit_logreg", "logreg")]:
        r = results[key]
        artefact = {
            "model": r["model"],
            "model_version": f"{name}_{tag}",
            "model_type": key,
            "feature_cols": feature_cols,
            "preprocessing_transformers": {"stdscaler": scaler},
            "data_dates": {k: (v.strftime("%Y-%m-%d")
                               if isinstance(v, datetime) else v)
                           for k, v in cfg.items()},
            "results": {"auc_train": r["auc"]["train"],
                        "auc_test": r["auc"]["test"],
                        "auc_oot": r["auc"]["oot"],
                        "gini_oot": _gini(r["auc"]["oot"])},
            "hp_params": r["params"],
        }
        path = os.path.join(MODEL_BANK, f"{name}_{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(artefact, f)
        print(f"saved model artefact: {path}")

    # ---- champion selection by OOT AUC ----
    champ = max(results, key=lambda k: results[k]["auc"]["oot"])
    print(f"\nCHAMPION by OOT AUC: {champ} "
          f"(oot_auc={round(results[champ]['auc']['oot'], 4)})")
    with open(os.path.join(MODEL_BANK, "champion.txt"), "w") as f:
        cname = "credit_xgb" if champ == "xgb" else "credit_logreg"
        f.write(f"{cname}_{tag}.pkl\n")

    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.snapshotdate)
