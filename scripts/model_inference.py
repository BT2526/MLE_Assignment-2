"""
Model inference entry point.

Loads a trained artefact from the model bank, scores the current snapshot's
feature-store partition, and writes per-customer calibrated probabilities
and binary default/no-default verdicts to the gold predictions datamart.

    python3 model_inference.py --snapshotdate 2024-10-01 --modelname credit_xgb_2024_09_01.pkl
"""

import argparse
import os
import pickle

import pandas as pd

from pipeline_common import get_spark
from pyspark.sql.functions import col

# Decision threshold applied to the calibrated probability to produce an
# explicit default/no-default verdict per customer. 0.5 is used as the
# standard midpoint — customers scoring at or above this are classified as
# likely to default. This is a business-level input, not a modelling
# decision: a risk-averse lender would lower this (catch more defaulters
# at the cost of more false positives); a growth-focused lender would
# raise it (approve more customers, accept more missed defaults).
CLASSIFICATION_THRESHOLD = 0.5


def main(snapshotdate, modelname, labeldate=None):
    """
    snapshotdate: application date — which feature store partition to score
    labeldate:    optional label date — if provided, stamps output snapshot_date
                  with this value instead of snapshotdate, so the prediction file
                  aligns with the label store partition for that label date.
                  labeldate = snapshotdate + 6 months (MOB=6 convention).
                  If not provided, snapshotdate is used for both loading and stamping.
    """
    effective_date = labeldate if labeldate else snapshotdate
    print(f"\n--- inference {snapshotdate} (label date: {effective_date}) / {modelname} ---\n")

    model_path = os.path.join("model_bank", modelname)
    if not os.path.exists(model_path):
        print(f"model {modelname} not in bank yet (trained on a later month); "
              f"skipping inference for {snapshotdate}.")
        return
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    print(f"loaded model: {model_path}")

    feat_path = (f"datamart/gold/feature_store/"
                 f"gold_feature_store_{snapshotdate.replace('-', '_')}.parquet")
    if not os.path.exists(feat_path):
        print(f"no feature partition for {snapshotdate}, nothing to score.")
        return

    spark = get_spark("inference")
    fdf = spark.read.parquet(feat_path).toPandas()
    spark.stop()
    if len(fdf) == 0:
        print("empty feature partition, nothing to score.")
        return

    feature_cols = artefact["feature_cols"]
    # Align columns (defensive: fill any missing with 0, keep order)
    for c in feature_cols:
        if c not in fdf.columns:
            fdf[c] = 0
    X = fdf[feature_cols]
    X = artefact["preprocessing_transformers"]["stdscaler"].transform(X)

    # Use the calibrated probability when a calibrator was saved into this
    # artefact (currently XGBoost only - see model_train.py's calibration
    # block for why Logistic Regression doesn't need one). This is the step
    # that makes calibration actually take effect on real predictions,
    # rather than the correction being computed and evaluated at training
    # time but never actually applied to what gets scored each month.
    calibrator = artefact.get("calibrator")
    if calibrator is not None:
        proba = calibrator.predict_proba(X)[:, 1]
        print("using CALIBRATED probabilities (Platt-scaled)")
    else:
        proba = artefact["model"].predict_proba(X)[:, 1]
        print("using raw model probabilities (no calibrator in this artefact)")
    # Stamp output with label date (effective_date) so prediction files align
    # with the label store for monitoring/F1 computation. When labeldate is
    # provided, this overwrites the application-date snapshot_date from the
    # feature store with the corresponding label date (application + 6 months).
    out = fdf[["Customer_ID", "snapshot_date"]].copy()
    if labeldate:
        out["snapshot_date"] = pd.to_datetime(labeldate)
    out["model_name"] = artefact["model_version"]
    out["model_predictions"] = proba
    # Binary verdict: 1 = predicted to default, 0 = predicted not to default.
    # Applied on top of the calibrated probability using CLASSIFICATION_THRESHOLD.
    # Stored alongside the raw probability so downstream consumers can use
    # either: the verdict for simple approve/reject decisions, or the
    # probability for risk-based pricing, tiering, or further analysis.
    out["prediction_label"] = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    n_default = out["prediction_label"].sum()
    print(f"predicted defaults: {n_default} / {len(out)} "
          f"({round(100 * n_default / len(out), 1)}%) "
          f"at threshold={CLASSIFICATION_THRESHOLD}")

    model_tag = artefact["model_version"]
    out_dir = f"datamart/gold/model_predictions/{model_tag}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{model_tag}_predictions_{effective_date.replace('-', '_')}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"saved predictions: {out_path} ({len(out)} rows, "
          f"mean_score={round(proba.mean(), 4)})")
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True,
                    help="application date — which feature store partition to score (YYYY-MM-DD)")
    p.add_argument("--modelname", required=True)
    p.add_argument("--labeldate", default=None,
                    help="optional label date to stamp output with (YYYY-MM-DD). "
                         "Use when snapshotdate is an application date but the output "
                         "should align with the label store (labeldate = snapshotdate + 6mo).")
    a = p.parse_args()
    main(a.snapshotdate, a.modelname, a.labeldate)
