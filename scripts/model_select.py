"""
Model selection script.

Reads both model artefacts from model_bank/ (XGBoost and Logistic Regression),
compares their OOT AUC, and writes the winner's filename to champion.txt.

This script is a separate DAG task that runs after both model_train_xgb and
model_train_logreg complete — making the selection step explicitly visible in
the Airflow DAG graph rather than hidden inside a combined training script.

Design choice: champion = argmax(OOT AUC).
- OOT is the only split never touched during tuning or calibration, making it
  the single most honest pre-deployment performance estimate.
- AUC summarises ranking quality across ALL thresholds, unlike F1/precision/
  recall which are threshold-dependent and would require an additional business
  decision about the operating point before they could be used for selection.
- Calibration is applied AFTER selection (inside model_inference.py), not
  before — it is a monotonic rescaling that cannot change ranking or AUC, so
  it correctly plays no role in the selection decision itself.

Only overwrites champion.txt for the standard 12-month window run. Comparison
runs (tt_months != 12) produce separately-named artefacts and must never
silently change which model the downstream inference/monitor tasks treat as
the champion in production.

    python3 model_select.py --snapshotdate 2024-09-01
"""

import argparse
import os
import pickle

MODEL_BANK = "model_bank/"


def main(snapshotdate, tt_months=12):
    print(f"\n--- model selection {snapshotdate} (tt_months={tt_months}) ---\n")

    window_suffix = "" if tt_months == 12 else f"_tt{tt_months}m"
    tag = snapshotdate.replace("-", "_") + window_suffix

    candidates = {
        "xgb":    f"credit_xgb_{tag}.pkl",
        "logreg": f"credit_logreg_{tag}.pkl",
    }

    results = {}
    for key, fname in candidates.items():
        path = os.path.join(MODEL_BANK, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected artefact not found: {path}\n"
                f"Ensure model_train_xgb and model_train_logreg both completed "
                f"successfully for snapshotdate={snapshotdate} before running "
                f"model_select."
            )
        with open(path, "rb") as f:
            art = pickle.load(f)
        oot_auc = art["results"]["auc_oot"]
        results[key] = {"fname": fname, "auc_oot": oot_auc}
        print(f"  {key:8s}  OOT AUC = {oot_auc:.4f}  ({fname})")

    champ_key = max(results, key=lambda k: results[k]["auc_oot"])
    champ_fname = results[champ_key]["fname"]
    champ_auc   = results[champ_key]["auc_oot"]
    runner_key  = [k for k in results if k != champ_key][0]
    runner_auc  = results[runner_key]["auc_oot"]

    print(f"\nCHAMPION: {champ_key} (OOT AUC {champ_auc:.4f})"
          f"  vs  {runner_key} (OOT AUC {runner_auc:.4f})"
          f"  margin: {champ_auc - runner_auc:.4f}")

    if tt_months == 12:
        champion_txt = os.path.join(MODEL_BANK, "champion.txt")
        with open(champion_txt, "w") as f:
            f.write(f"{champ_fname}\n")
        print(f"wrote champion.txt -> {champ_fname}")
    else:
        print(f"(tt_months={tt_months} is a comparison run — "
              f"champion.txt left untouched)")

    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--tt_months", type=int, default=12)
    a = p.parse_args()
    main(a.snapshotdate, a.tt_months)
