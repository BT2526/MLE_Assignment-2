"""
Model monitoring entry point.

Reads all gold prediction partitions for a model, joins them to realised labels
where available, and produces TWO things across the full time period:

  1. A gold monitoring table  (datamart/gold/model_monitoring/<model>/)
        per snapshot: n, mean_score, PSI (vs first scored month), AUC, GINI
  2. PNG charts                (datamart/gold/monitoring_plots/<model>/)
        - score-distribution stability via PSI over time
        - discrimination (AUC / GINI) over time

PSI (Population Stability Index) measures distribution drift of the score:
        PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift.

    python3 model_monitor.py --snapshotdate 2024-12-01 --modelname credit_xgb_2024_09_01.pkl
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def _psi(expected, actual, bins=10):
    """Population Stability Index between an expected and an actual score array."""
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return np.nan
    e_pct = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a_pct = np.histogram(actual, bins=quantiles)[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-4, None)
    a_pct = np.clip(a_pct, 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main(snapshotdate, modelname):
    print(f"\n--- monitoring {modelname} ---\n")

    model_file = os.path.join("model_bank", modelname)
    if not os.path.exists(model_file):
        print(f"model {modelname} not in bank yet; skipping monitoring.")
        return
    with open(model_file, "rb") as f:
        artefact = pickle.load(f)
    model_tag = artefact["model_version"]

    pred_dir = f"datamart/gold/model_predictions/{model_tag}/"
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if not pred_files:
        print(f"no predictions found for {model_tag}, nothing to monitor.")
        return

    # realised labels (may not exist for the most recent months yet).
    # A prediction made at application month M is judged against the label
    # observed at M+6 (mob=6).  We shift the label date back by 6 months so it
    # aligns to the application/prediction month.
    label_files = glob.glob("datamart/gold/label_store/*.parquet")
    if label_files:
        labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        labels["snapshot_date"] = (
            pd.to_datetime(labels["snapshot_date"]) - pd.DateOffset(months=6)
        )
        labels = labels[["Customer_ID", "snapshot_date", "label"]]
    else:
        labels = pd.DataFrame(columns=["Customer_ID", "snapshot_date", "label"])

    rows, baseline = [], None
    for pf in pred_files:
        pdf = pd.read_parquet(pf)
        pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])
        snap = pdf["snapshot_date"].iloc[0]
        scores = pdf["model_predictions"].values
        if baseline is None:
            baseline = scores  # first scored month is the PSI reference

        psi = _psi(baseline, scores)

        auc = gini = np.nan
        if len(labels):
            merged = pdf.merge(labels, on=["Customer_ID", "snapshot_date"], how="inner")
            if len(merged) > 0 and merged["label"].nunique() == 2:
                auc = roc_auc_score(merged["label"], merged["model_predictions"])
                gini = 2 * auc - 1

        rows.append({"snapshot_date": snap.strftime("%Y-%m-%d"),
                     "n_scored": len(pdf),
                     "mean_score": round(float(scores.mean()), 4),
                     "psi": round(psi, 4) if psi == psi else np.nan,
                     "auc": round(auc, 4) if auc == auc else np.nan,
                     "gini": round(gini, 4) if gini == gini else np.nan})

    mon = pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)
    print(mon.to_string(index=False))

    # ---- write gold monitoring table ----
    mon_dir = f"datamart/gold/model_monitoring/{model_tag}/"
    os.makedirs(mon_dir, exist_ok=True)
    mon_path = os.path.join(mon_dir, f"{model_tag}_monitoring.parquet")
    mon.to_parquet(mon_path, index=False)
    print(f"saved monitoring table: {mon_path}")

    # ---- charts ----
    plot_dir = f"datamart/gold/monitoring_plots/{model_tag}/"
    os.makedirs(plot_dir, exist_ok=True)
    x = pd.to_datetime(mon["snapshot_date"])

    # stability: PSI
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mon["psi"], marker="o", color="#2563eb", label="PSI")
    ax.axhline(0.1, ls="--", c="orange", lw=1, label="0.10 (moderate)")
    ax.axhline(0.25, ls="--", c="red", lw=1, label="0.25 (significant)")
    ax.set_title(f"Score Stability (PSI vs baseline) - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("PSI")
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    psi_png = os.path.join(plot_dir, "psi_stability.png")
    fig.savefig(psi_png, dpi=120); plt.close(fig)

    # performance: AUC / GINI
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mon["auc"], marker="o", color="#16a34a", label="AUC")
    ax.plot(x, mon["gini"], marker="s", color="#9333ea", label="GINI")
    ax.set_title(f"Discrimination over time - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("score")
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    perf_png = os.path.join(plot_dir, "auc_gini_performance.png")
    fig.savefig(perf_png, dpi=120); plt.close(fig)

    print(f"saved plots: {psi_png}, {perf_png}")
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--modelname", required=True)
    a = p.parse_args()
    main(a.snapshotdate, a.modelname)
