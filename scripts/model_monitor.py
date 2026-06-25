"""
Model monitoring entry point.

Reads all gold prediction partitions for a model, joins them to realised labels
where available, and produces TWO things across the full time period:

  1. A gold monitoring table  (datamart/gold/model_monitoring/<model>/)
        per snapshot: n, mean_score, PSI (vs first scored month), AUC, GINI,
        CSI_max (worst feature-level drift among top-N important features),
        auc_drop_from_oot1 (this month's AUC vs the model's original OOT1)
  2. PNG charts                (datamart/gold/monitoring_plots/<model>/)
        - score-distribution stability via PSI over time
        - discrimination (AUC / GINI) over time, with the OOT1 baseline drawn in
        - feature-level stability via CSI over time

PSI (Population Stability Index) measures distribution drift of the score:
        PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift.

CSI (Characteristic Stability Index) is the SAME formula, applied to an
individual input feature instead of the model's output score. Where PSI asks
"has the model's overall risk score shifted", CSI asks "has this specific
feature's distribution shifted" - it can localise drift to a cause, where PSI
can only say something drifted without saying what.

CSI is computed only for the TOP_N_CSI_FEATURES most important features (per
feature_importance_report.py's saved ranking), not all 84 - drift in a feature
the model barely uses (e.g. a single one-hot occupation category at <2%
importance) shouldn't carry the same governance weight as drift in the
model's top driver (e.g. Outstanding_Debt at ~10% importance). The reported
CSI_max is the WORST (maximum) CSI among those top-N features for that month,
so one badly-drifted important feature cannot be hidden/diluted by averaging
against many stable ones.

OOT1 BASELINE TRACKING: the model's original OOT AUC (computed once, at
training time, the final pre-deployment checkpoint - see model_train.py) is
pulled from the saved artefact and used as a fixed reference point on the
AUC/GINI chart. Each month's realised AUC is compared against it
(auc_drop_from_oot1 = oot1_auc - this_month_auc); a drop exceeding
AUC_DROP_THRESHOLD fires an [ALERT], mirroring the PSI/CSI pattern. Unlike
PSI/CSI - which can react to THIS month's incoming data immediately - this
check is necessarily lagged by the label's 6-month maturity (mob=6), so it
is a slower, retrospective health check, not a fast-reacting tripwire. This
is informational/diagnostic only: it does NOT feed the automated retrain
trigger in the DAG (see _should_train in dags/dag.py, which checks PSI/CSI
only) - a human reviewing this chart can choose to act on a flagged AUC
drop, but the pipeline does not retrain automatically on this signal alone.

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
from sklearn.metrics import (roc_auc_score, f1_score, brier_score_loss,
                              precision_score, recall_score)

TOP_N_CSI_FEATURES = 15
AUC_DROP_THRESHOLD = 0.05
# Threshold used to convert calibrated probabilities into binary predictions
# for F1/precision/recall computation. Must match model_inference.py's
# CLASSIFICATION_THRESHOLD so monitored metrics reflect real deployed decisions.
CLASSIFICATION_THRESHOLD = 0.5


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


def _get_top_csi_features(model_tag, feature_cols, top_n=TOP_N_CSI_FEATURES):
    """
    Scope CSI to the top-N features by importance, reusing the ranking already
    saved by feature_importance_report.py. Falls back to the first top_n
    columns in feature_cols (arbitrary, but still bounded) if that report
    hasn't been run yet for this model, so monitoring never hard-fails just
    because the interpretability report happens to run separately.
    """
    csv_path = f"reports/feature_importance/{model_tag}_feature_importance.csv"
    if os.path.exists(csv_path):
        ranked = pd.read_csv(csv_path)
        return ranked["feature"].head(top_n).tolist()
    print(f"  [csi] no feature_importance_report.py output found for {model_tag} "
          f"at {csv_path}; falling back to first {top_n} feature_cols as a "
          f"placeholder scope (run feature_importance_report.py for a proper ranking).")
    return feature_cols[:top_n]


def _compute_csi_for_snapshot(feature_baseline_df, feature_actual_df, csi_features):
    """
    CSI per feature for one snapshot vs the baseline snapshot, using the same
    PSI formula applied to each feature's own raw values instead of the score.
    Returns (csi_max, csi_max_feature, per_feature_dict).
    """
    per_feature = {}
    for feat in csi_features:
        if feat not in feature_baseline_df.columns or feat not in feature_actual_df.columns:
            continue
        per_feature[feat] = _psi(
            feature_baseline_df[feat].dropna().values,
            feature_actual_df[feat].dropna().values,
        )
    valid = {k: v for k, v in per_feature.items() if v == v}  # drop NaNs
    if not valid:
        return np.nan, None, per_feature
    worst_feat = max(valid, key=valid.get)
    return valid[worst_feat], worst_feat, per_feature


def main(snapshotdate, modelname):
    print(f"\n--- monitoring {modelname} ---\n")

    model_file = os.path.join("model_bank", modelname)
    if not os.path.exists(model_file):
        print(f"model {modelname} not in bank yet; skipping monitoring.")
        return
    with open(model_file, "rb") as f:
        artefact = pickle.load(f)
    model_tag = artefact["model_version"]

    # OOT1 baseline: the model's original, pre-deployment AUC, computed once
    # at training time (see model_train.py). Read directly from the saved
    # artefact rather than recomputed here, so this always matches exactly
    # what's already reported in MLflow and the deck - never a second,
    # potentially-inconsistent calculation of the same number.
    oot1_auc   = artefact.get("results", {}).get("auc_oot")
    oot1_brier = artefact.get("results", {}).get("brier_oot_calibrated")
    if oot1_auc is not None:
        print(f"OOT1 baseline AUC for {model_tag}: {oot1_auc}")
    if oot1_brier is not None:
        print(f"OOT1 baseline Brier for {model_tag}: {oot1_brier}")
    else:
        print(f"[warning] no OOT1 baseline found in artefact for {model_tag}; "
              f"AUC-drop tracking will be skipped for this model.")

    pred_dir = f"datamart/gold/model_predictions/{model_tag}/"
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if not pred_files:
        print(f"no predictions found for {model_tag}, nothing to monitor.")
        return

    # Realised labels (may not exist for the most recent months yet).
    # Predictions use LABEL DATES as snapshot_date (set by _load_aligned which
    # renames label_date -> snapshot_date). The label store also uses label dates.
    # So the join is label-date to label-date — no offset needed.
    # (The old 6-month subtraction was for an earlier design where predictions
    # stored application dates; that design has since changed.)
    label_files = glob.glob("datamart/gold/label_store/*.parquet")
    if label_files:
        labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"])
        labels["snapshot_date"] = labels["snapshot_date"].astype("datetime64[ns]")
        labels = labels[["Customer_ID", "snapshot_date", "label"]]
    else:
        labels = pd.DataFrame(columns=["Customer_ID", "snapshot_date", "label"])

    rows, baseline, feat_baseline_df = [], None, None
    feature_files_by_date = {
        os.path.basename(f).replace("gold_feature_store_", "").replace(".parquet", "").replace("_", "-"): f
        for f in glob.glob("datamart/gold/feature_store/*.parquet")
    }

    artefact_feature_cols = artefact.get("feature_cols", [])
    csi_features = _get_top_csi_features(model_tag, artefact_feature_cols)
    print(f"[csi] scoping to top {len(csi_features)} features: {csi_features}\n")

    for pf in pred_files:
        pdf = pd.read_parquet(pf)
        # Cast to datetime64[ns] explicitly — prediction files may store
        # snapshot_date as datetime.date (object) depending on how they were
        # written (model_inference.py vs OOT1 saving in model_train_xgb.py).
        # Forcing both sides to the same type before any merge prevents the
        # "merging object and datetime64[ns]" ValueError.
        pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])
        snap = pd.Timestamp(pdf["snapshot_date"].iloc[0])
        scores = pdf["model_predictions"].values
        if baseline is None:
            baseline = scores  # first scored month is the PSI reference

        psi = _psi(baseline, scores)

        auc = gini = f1 = precision = recall = brier = np.nan
        if len(labels):
            # Try direct join first (works for OOT1 months saved with label dates).
            # If that produces no rows, try joining via label_date = snap + 6 months
            # (works for inference months saved with application dates, where the
            # corresponding label lives in the partition 6 months later).
            merged = pdf.merge(labels, on=["Customer_ID", "snapshot_date"], how="inner")
            if len(merged) == 0:
                label_date = snap + pd.DateOffset(months=6)
                labels_offset = labels[labels["snapshot_date"] == label_date].copy()
                labels_offset = labels_offset.rename(columns={"snapshot_date": "label_snapshot_date"})
                merged = pdf.merge(
                    labels_offset[["Customer_ID", "label_snapshot_date", "label"]],
                    on="Customer_ID", how="inner"
                )
            if len(merged) > 0 and merged["label"].nunique() == 2:
                auc = roc_auc_score(merged["label"], merged["model_predictions"])
                gini = 2 * auc - 1
                y_pred = (merged["model_predictions"] >= CLASSIFICATION_THRESHOLD).astype(int)
                f1        = f1_score(merged["label"], y_pred, zero_division=0)
                precision = precision_score(merged["label"], y_pred, zero_division=0)
                recall    = recall_score(merged["label"], y_pred, zero_division=0)
                # Brier score: mean squared error between predicted probability
                # and actual outcome. Lower is better; 0 = perfect, 0.25 = random.
                # Measures calibration quality in addition to discrimination.
                brier = brier_score_loss(merged["label"], merged["model_predictions"])

        # ---- CSI: feature-level drift, scoped to top-N important features ----
        csi_max = np.nan
        csi_max_feature = None
        csi_per_feature = {}
        feat_path = feature_files_by_date.get(snap.strftime("%Y-%m-%d"))
        if feat_path and os.path.exists(feat_path):
            feat_df = pd.read_parquet(feat_path)
            if feat_baseline_df is None:
                feat_baseline_df = feat_df  # first scored month is the CSI reference too
            csi_max, csi_max_feature, csi_per_feature = _compute_csi_for_snapshot(
                feat_baseline_df, feat_df, csi_features
            )

        # ---- AUC drop vs OOT1 baseline (only computable once labels exist) ----
        auc_drop_from_oot1 = np.nan
        if oot1_auc is not None and auc == auc:  # auc == auc excludes NaN
            auc_drop_from_oot1 = round(oot1_auc - auc, 4)

        row = {"snapshot_date": snap.strftime("%Y-%m-%d"),
               "n_scored": len(pdf),
               "mean_score": round(float(scores.mean()), 4),
               "psi": round(psi, 4) if psi == psi else np.nan,
               "f1": round(f1, 4) if f1 == f1 else np.nan,
               "precision": round(precision, 4) if precision == precision else np.nan,
               "recall": round(recall, 4) if recall == recall else np.nan,
               "brier": round(brier, 4) if brier == brier else np.nan,
               "auc": round(auc, 4) if auc == auc else np.nan,
               "gini": round(gini, 4) if gini == gini else np.nan,
               "auc_drop_from_oot1": auc_drop_from_oot1,
               "csi_max": round(csi_max, 4) if csi_max == csi_max else np.nan,
               "csi_max_feature": csi_max_feature}
        # One column per top-N feature (csi_<feature_name>), so the full
        # breakdown is preserved in the saved table, not just the worst
        # feature. csi_max/csi_max_feature above are kept unchanged so the
        # existing [ALERT] logic and the single-line CSI chart below don't
        # need to change at all - this is purely additive.
        for feat in csi_features:
            val = csi_per_feature.get(feat, np.nan)
            row[f"csi_{feat}"] = round(val, 4) if val == val else np.nan
        rows.append(row)

    mon = pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)
    print(mon.to_string(index=False))

    # ---- explicit, auditable log line whenever PSI or CSI breaches 0.25 ----
    # This is the audit trail referenced in the DAG's retrain-trigger docstring:
    # a human reviewing logs after the fact can see exactly which signal fired
    # and which feature was responsible, even though the retrain itself (in
    # the DAG) acts on this automatically.
    DRIFT_THRESHOLD = 0.25
    latest = mon.iloc[-1]
    if latest["psi"] == latest["psi"] and latest["psi"] > DRIFT_THRESHOLD:
        print(f"\n[ALERT] PSI breached {DRIFT_THRESHOLD} for {model_tag} "
              f"at {latest['snapshot_date']} (PSI={latest['psi']}); "
              f"this would trigger a scheduled retrain.")
    if latest["csi_max"] == latest["csi_max"] and latest["csi_max"] > DRIFT_THRESHOLD:
        print(f"\n[ALERT] CSI breached {DRIFT_THRESHOLD} for {model_tag} "
              f"at {latest['snapshot_date']} on feature '{latest['csi_max_feature']}' "
              f"(CSI={latest['csi_max']}); this would trigger a scheduled retrain.")
    if latest["auc_drop_from_oot1"] == latest["auc_drop_from_oot1"] and \
            latest["auc_drop_from_oot1"] > AUC_DROP_THRESHOLD:
        print(f"\n[ALERT] AUC dropped {latest['auc_drop_from_oot1']} below the "
              f"OOT1 baseline ({oot1_auc}) for {model_tag} at {latest['snapshot_date']} "
              f"(threshold: {AUC_DROP_THRESHOLD}). This is INFORMATIONAL ONLY - "
              f"unlike PSI/CSI, this does NOT automatically trigger a retrain "
              f"(see module docstring for why: this signal is lagged by the "
              f"label's 6-month maturity, so it reacts to performance from 6 "
              f"months ago, not current conditions). Recommend manual review.")

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

    # stability: CSI (feature-level drift, worst of the top-N important features)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mon["csi_max"], marker="o", color="#dc2626", label="CSI (max over top-N features)")
    ax.axhline(0.1, ls="--", c="orange", lw=1, label="0.10 (moderate)")
    ax.axhline(0.25, ls="--", c="red", lw=1, label="0.25 (significant)")
    ax.set_title(f"Feature Stability (CSI vs baseline, top {len(csi_features)} features) - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("CSI")
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    csi_png = os.path.join(plot_dir, "csi_stability.png")
    fig.savefig(csi_png, dpi=120); plt.close(fig)

    # stability: CSI breakdown - ALL top-N features individually, not just
    # the worst one. Lets a viewer see whether drift is concentrated in one
    # feature or broad-based across several, which csi_max alone can't show.
    csi_cols = [f"csi_{feat}" for feat in csi_features if f"csi_{feat}" in mon.columns]
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, col in enumerate(csi_cols):
        feat_name = col[len("csi_"):]
        ax.plot(x, mon[col], marker="o", markersize=4, lw=1.3,
                color=cmap(i % 20), label=feat_name)
    ax.axhline(0.1, ls="--", c="orange", lw=1, label="0.10 (moderate)")
    ax.axhline(0.25, ls="--", c="red", lw=1, label="0.25 (significant)")
    ax.set_title(f"Feature Stability breakdown - all top {len(csi_features)} features - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("CSI")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1)
    ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    csi_breakdown_png = os.path.join(plot_dir, "csi_breakdown_all_features.png")
    fig.savefig(csi_breakdown_png, dpi=120); plt.close(fig)

    # performance: AUC over time with OOT1 baseline reference.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mon["auc"], marker="s", color="#9333ea", lw=2, label="AUC")
    if oot1_auc is not None:
        ax.axhline(oot1_auc, ls="--", c="gray", lw=1,
                    label=f"OOT1 baseline AUC ({oot1_auc:.4f})")
        ax.axhline(oot1_auc - AUC_DROP_THRESHOLD, ls=":", c="red", lw=1,
                    label=f"AUC alert threshold (OOT1 - {AUC_DROP_THRESHOLD})")
    ax.set_title(f"Model Discrimination (AUC) over time - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("AUC")
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    perf_png = os.path.join(plot_dir, "auc_performance.png")
    fig.savefig(perf_png, dpi=120); plt.close(fig)

    # Brier score: lower is better (0 = perfect, 1 = worst).
    # Benchmarked against OOT1 Brier — alert threshold = OOT1 + 0.05,
    # meaning the model's calibration has meaningfully degraded relative
    # to its validated baseline.
    BRIER_ALERT_THRESHOLD = 0.05
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mon["brier"], marker="o", color="#0891b2", lw=2, label="Brier score")
    if oot1_brier is not None:
        ax.axhline(oot1_brier, ls="--", c="gray", lw=1,
                    label=f"OOT1 baseline Brier ({oot1_brier:.4f})")
        ax.axhline(oot1_brier + BRIER_ALERT_THRESHOLD, ls=":", c="red", lw=1,
                    label=f"Alert threshold (OOT1 + {BRIER_ALERT_THRESHOLD})")
    ax.set_title(f"Model Calibration (Brier Score) over time - {model_tag}")
    ax.set_xlabel("snapshot date"); ax.set_ylabel("Brier score")
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    brier_png = os.path.join(plot_dir, "brier_score.png")
    fig.savefig(brier_png, dpi=120); plt.close(fig)

    print(f"saved plots: {psi_png}, {csi_png}, {csi_breakdown_png}, {perf_png}, {brier_png}")
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--modelname", required=True)
    a = p.parse_args()
    main(a.snapshotdate, a.modelname)
