"""
Feature importance report (interpretability analysis).

Standalone, on-demand analysis script - not part of the orchestrated DAG.
Loads a trained model from the model bank, ranks its features by importance,
and saves both a chart and a data table so the ranking can be reviewed,
presented, or audited without re-running training.

Feature importance serves as an investigative approach, not direcetly plugged into
the pipeline, i.e. not a recurring operational step like inference or
monitoring - it doesn't need to regenerate every month.

    python3 feature_importance_report.py --modelname credit_xgb_2024_09_01.pkl
    python3 feature_importance_report.py --modelname credit_xgb_2024_09_01.pkl --top_n 15

Only tree-based models (XGBoost, Random Forest, etc.) expose
`feature_importances_`. For Logistic Regression, the script instead reports
the absolute standardized coefficients, which play an analogous role: since
all features were scaled by the same StandardScaler before fitting, the
coefficient magnitude reflects how much that feature moves the prediction
per unit of (standardized) change.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_BANK = "model_bank/"
REPORT_DIR = "reports/feature_importance/"


def _get_importances(artefact):
    """Return a (feature, importance, kind) ranking appropriate to the model type."""
    model = artefact["model"]
    feature_cols = artefact["feature_cols"]

    if hasattr(model, "feature_importances_"):
        # Tree-based models (XGBoost, RandomForest, etc.): built-in importances,
        # already non-negative and summing to 1.
        values = model.feature_importances_
        kind = "XGBoost importance (gain-based, normalised to sum to 1)"
    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression): use |coefficient|. Valid as
        # a comparable ranking specifically because every feature was scaled by
        # the same StandardScaler before fitting (see preprocessing_transformers
        # in the artefact), so coefficients are on the same standardized scale.
        values = np.abs(model.coef_).ravel()
        kind = "|standardized coefficient| (Logistic Regression)"
    else:
        raise ValueError(
            f"Model type {artefact.get('model_type')} exposes neither "
            f"feature_importances_ nor coef_; cannot rank features."
        )

    df = pd.DataFrame({"feature": feature_cols, "importance": values})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df, kind


def main(modelname, top_n):
    print(f"\n--- feature importance report: {modelname} ---\n")

    model_path = os.path.join(MODEL_BANK, modelname)
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    model_tag = artefact["model_version"]
    imp_df, kind = _get_importances(artefact)

    total = imp_df["importance"].sum()
    top = imp_df.head(top_n).copy()
    top_share = top["importance"].sum() / total if total > 0 else float("nan")

    print(f"model: {model_tag}  ({kind})")
    print(f"total features: {len(imp_df)}")
    print(f"top {top_n} features capture {top_share:.1%} of total importance")
    print()
    print(top.to_string(index=False))

    os.makedirs(REPORT_DIR, exist_ok=True)

    # ---- save full ranked table as CSV (data artefact) ----
    csv_path = os.path.join(REPORT_DIR, f"{model_tag}_feature_importance.csv")
    imp_df.to_csv(csv_path, index=False)
    print(f"\nsaved full ranking: {csv_path}")

    # ---- save horizontal bar chart of the top N (visual artefact for the deck) ----
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    plot_data = top.iloc[::-1]  # reverse so the #1 feature plots at the top
    ax.barh(plot_data["feature"], plot_data["importance"], color="#2563eb")
    ax.set_xlabel(kind)
    ax.set_title(f"Top {top_n} Feature Importances - {model_tag}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(REPORT_DIR, f"{model_tag}_feature_importance_top{top_n}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"saved chart: {png_path}")

    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelname", required=True,
                   help="filename of the model artefact in model_bank/, "
                        "e.g. credit_xgb_2024_09_01.pkl")
    p.add_argument("--top_n", type=int, default=20,
                   help="how many top features to chart and highlight (default 20)")
    a = p.parse_args()
    main(a.modelname, a.top_n)

