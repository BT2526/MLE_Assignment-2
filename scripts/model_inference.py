"""
Model inference entry point.

Loads a trained artefact from the model bank, scores the current snapshot's
feature-store partition, and writes per-customer probabilities to the gold
predictions datamart.

    python3 model_inference.py --snapshotdate 2024-10-01 --modelname credit_xgb_2024_09_01.pkl
"""

import argparse
import os
import pickle

import pandas as pd

from pipeline_common import get_spark
from pyspark.sql.functions import col


def main(snapshotdate, modelname):
    print(f"\n--- inference {snapshotdate} / {modelname} ---\n")

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

    proba = artefact["model"].predict_proba(X)[:, 1]
    out = fdf[["Customer_ID", "snapshot_date"]].copy()
    out["model_name"] = artefact["model_version"]
    out["model_predictions"] = proba

    model_tag = artefact["model_version"]
    out_dir = f"datamart/gold/model_predictions/{model_tag}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{model_tag}_predictions_{snapshotdate.replace('-', '_')}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"saved predictions: {out_path} ({len(out)} rows, "
          f"mean_score={round(proba.mean(), 4)})")
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--modelname", required=True)
    a = p.parse_args()
    main(a.snapshotdate, a.modelname)
