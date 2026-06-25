"""
Evidently AI drift report - complementary, presentation-layer drift report.

Standalone, on-demand script. Does not replace or feed into the existing
PSI/CSI drift monitoring already built into model_monitor.py, which remains
the system actually driving the DAG's automated retrain trigger (see
dags/dag.py's _should_train). This script produces an additional, polished
report combining:

  1. PSI  (model output drift)  - identical formula/baseline logic to
       model_monitor.py's _psi(): first scored month is the fixed reference,
       compared against the requested CURRENT month's predicted scores.
  2. CSI  (per-feature drift)   - identical formula/baseline logic to
       model_monitor.py's _compute_csi_for_snapshot(), scoped to the same
       top-N important features (per feature_importance_report.py's saved
       ranking). Unlike model_monitor.py, which only keeps the WORST
       feature's CSI for its table, this report shows the FULL per-feature
       breakdown (interactive dropdown + bar chart).
  3. Data summary (Evidently)   - DataQualityPreset only. Evidently's own
       DataDriftPreset is deliberately NOT used here, to avoid presenting a
       second, differently-computed "feature drift" number alongside our
       own CSI - having two disagreeing drift metrics for the same features
       would be confusing rather than additive. Evidently is used for what
       it's genuinely complementary at: a polished data-quality/profile
       summary, not a competing drift metric.

PSI and CSI are computed here (not read from model_monitor.py's saved
table) so this script can show the full per-feature CSI breakdown, which
model_monitor.py currently discards (it only persists the single worst
feature). Both use the exact same helper functions, copied verbatim from
model_monitor.py, so the numbers shown here are guaranteed consistent with
what's already driving the DAG's retrain logic - never a second, silently
different calculation of "the same" metric.

    python3 evidently_drift_report.py --baseline 2024-09-01 --current 2024-12-01 \
        --modelname credit_xgb_2024_09_01.pkl
"""

import argparse
import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataSummaryPreset

FEATURE_STORE_DIR = "datamart/gold/feature_store"
PRED_DIR_TMPL = "datamart/gold/model_predictions/{model_tag}/"
OUT_DIR = "datamart/gold/evidently_reports"
TOP_N_CSI_FEATURES = 15
DRIFT_THRESHOLD = 0.25

# Same exclusion list as model_train.py's feature_cols construction, kept
# consistent so "what counts as a feature" matches the rest of the pipeline.
NON_FEATURE_COLS = ("Customer_ID", "snapshot_date", "label", "loan_id", "label_def")


# ── PSI / CSI helpers - copied verbatim from model_monitor.py for consistency ──

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
    """Scope CSI to the top-N features by importance, reusing the ranking
    already saved by feature_importance_report.py. Falls back to the first
    top_n columns in feature_cols if that report isn't available."""
    csv_path = f"reports/feature_importance/{model_tag}_feature_importance.csv"
    if os.path.exists(csv_path):
        ranked = pd.read_csv(csv_path)
        col = "feature" if "feature" in ranked.columns else ranked.columns[0]
        ranked_feats = [f for f in ranked[col].tolist() if f in feature_cols]
        if ranked_feats:
            return ranked_feats[:top_n]
    return feature_cols[:top_n]


def _compute_csi_for_snapshot(feature_baseline_df, feature_actual_df, csi_features):
    """CSI per feature for one snapshot vs the baseline snapshot. Returns
    (csi_max, csi_max_feature, per_feature_dict) - identical to
    model_monitor.py, except THIS script keeps the per_feature dict rather
    than discarding it, since the report needs the full breakdown."""
    per_feature = {}
    for feat in csi_features:
        if feat not in feature_baseline_df.columns or feat not in feature_actual_df.columns:
            continue
        per_feature[feat] = _psi(
            feature_baseline_df[feat].dropna().values,
            feature_actual_df[feat].dropna().values,
        )
    valid = {k: v for k, v in per_feature.items() if v == v}
    if not valid:
        return np.nan, None, per_feature
    worst_feat = max(valid, key=valid.get)
    return valid[worst_feat], worst_feat, per_feature


# ── data loaders ────────────────────────────────────────────────────────────

def _load_feature_snapshot(date_str):
    path = os.path.join(FEATURE_STORE_DIR,
                         f"gold_feature_store_{date_str.replace('-', '_')}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no feature_store partition for {date_str} at {path}")
    return pd.read_parquet(path)


def _earliest_scored_month(model_tag):
    """The first (earliest) prediction partition for this model - this is
    the fixed PSI/CSI baseline, exactly matching model_monitor.py's logic
    of using the first scored month as the reference, never the requested
    --baseline argument directly (PSI/CSI baseline is a property of the
    model's deployment history, not something to pick per-report)."""
    pred_dir = PRED_DIR_TMPL.format(model_tag=model_tag)
    files = sorted(glob.glob(os.path.join(pred_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no predictions found for {model_tag} in {pred_dir}")
    first_file = files[0]
    first_df = pd.read_parquet(first_file)
    first_date = pd.to_datetime(first_df["snapshot_date"]).iloc[0].strftime("%Y-%m-%d")
    return first_date, first_df


def _load_scores(model_tag, date_str):
    pred_dir = PRED_DIR_TMPL.format(model_tag=model_tag)
    path_glob = os.path.join(pred_dir, f"*{date_str.replace('-', '_')}*.parquet")
    matches = glob.glob(path_glob)
    if not matches:
        raise FileNotFoundError(f"no predictions found for {model_tag} at {date_str}")
    return pd.read_parquet(matches[0])


# ── HTML section builders ───────────────────────────────────────────────────

def _psi_section_html(psi_value, baseline_date, current_date):
    if psi_value is None or psi_value != psi_value:
        status_html = ('<p style="color:#e53e3e;font-weight:600;">'
                        'PSI not available for this comparison.</p>')
    else:
        if psi_value > DRIFT_THRESHOLD:
            status, color = "SIGNIFICANT DRIFT", "#ef4444"
        elif psi_value > 0.10:
            status, color = "MODERATE DRIFT", "#f59e0b"
        else:
            status, color = "STABLE", "#22c55e"
        status_html = (f'<p style="color:{color};font-weight:700;font-size:1.1rem;">'
                        f'{status} (PSI = {psi_value:.4f})</p>')

    return f"""
    <div style="margin:40px;padding:30px;border:1px solid #e2e8f0;border-radius:16px;
                background-color:#fcfdff;box-shadow:0 10px 30px rgba(0,0,0,0.03);
                font-family:'Inter',sans-serif;">
      <div style="display:flex;flex-direction:column;align-items:center;text-align:center;">
        <h2 style="margin-bottom:10px;color:#1e1e2e;font-size:1.6rem;font-weight:700;">
          PSI (Model Output Drift)</h2>
        <p style="color:#585b70;max-width:700px;margin-bottom:10px;font-size:0.95rem;line-height:1.5;">
          Population Stability Index on the model's predicted-probability scores.
          Baseline = {baseline_date} (first scored month) &nbsp;|&nbsp; Current = {current_date}
          <br><span style="font-size:0.85rem;color:#94a3b8;">
          Thresholds: &lt; 0.10 stable | 0.10-0.25 moderate | &gt; 0.25 significant</span>
        </p>
        {status_html}
        <div id="psiGauge" style="width:480px;height:300px;margin:10px auto;"></div>
      </div>
    </div>
    <script>
      Plotly.newPlot("psiGauge", [{{
        type: "indicator", mode: "gauge+number",
        value: {json.dumps(psi_value if psi_value == psi_value else 0)},
        gauge: {{
          axis: {{ range: [0, {max(0.4, (psi_value or 0) * 1.15)}], tickwidth: 1, tickcolor: "#585b70" }},
          bar: {{ color: "#313244" }}, bgcolor: "white", borderwidth: 2, bordercolor: "#e2e8f0",
          steps: [
            {{ range: [0, 0.10], color: "#e2f9e0" }},
            {{ range: [0.10, 0.25], color: "#fef3c7" }},
            {{ range: [0.25, {max(0.4, (psi_value or 0) * 1.15)}], color: "#fee2e2" }}
          ],
          threshold: {{ line: {{ color: "#ef4444", width: 4 }}, value: 0.25 }}
        }}
      }}], {{ margin: {{t:30,b:30,l:30,r:30}}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)" }});
    </script>
    """


def _csi_section_html(per_feature_csi, baseline_date, current_date):
    features = list(per_feature_csi.keys())
    values = [round(v, 4) if v == v else 0.0 for v in per_feature_csi.values()]

    return f"""
    <div style="margin:40px;padding:30px;border:1px solid #e2e8f0;border-radius:16px;
                background-color:#fcfdff;box-shadow:0 10px 30px rgba(0,0,0,0.03);
                font-family:'Inter',sans-serif;">
      <div style="display:flex;flex-direction:column;align-items:center;text-align:center;">
        <h2 style="margin-bottom:10px;color:#1e1e2e;font-size:1.6rem;font-weight:700;">
          CSI (Covariate Shift - Top {len(features)} Important Features)</h2>
        <p style="color:#585b70;max-width:700px;margin-bottom:15px;font-size:0.95rem;line-height:1.5;">
          Per-feature distribution shift, scoped to the model's top {len(features)} most
          important features (per feature_importance_report.py's ranking).
          Baseline = {baseline_date} (first scored month) &nbsp;|&nbsp; Current = {current_date}
          <br><span style="font-size:0.85rem;color:#94a3b8;">
          Thresholds: &lt; 0.10 stable | 0.10-0.25 moderate | &gt; 0.25 significant</span>
        </p>
        <div style="margin-bottom:15px;">
          <label for="csiFeatureSelect" style="font-weight:600;margin-right:8px;color:#313244;">
            Select Feature: </label>
          <select id="csiFeatureSelect" onchange="updateCSIGauge()"
                  style="padding:8px 16px;border-radius:6px;border:1px solid #cbd5e1;
                         font-size:0.9rem;font-family:inherit;cursor:pointer;outline:none;
                         background-color:white;">
            {''.join(f'<option value="{f}">{f}</option>' for f in features)}
          </select>
        </div>
        <div id="csiGauge" style="width:480px;height:300px;margin:0 auto;"></div>
        <div id="csiBarChart" style="width:900px;height:{len(features)*25+60}px;
                                       margin-top:25px;margin-left:auto;margin-right:auto;"></div>
      </div>
    </div>
    <script>
      const csiFeatures = {json.dumps(features)};
      const csiValues = {json.dumps(values)};
      const csiData = {json.dumps(dict(zip(features, values)))};

      function csiColor(v) {{
        if (v < 0.10) return "#a6e3a1";
        if (v < 0.25) return "#f9e2af";
        return "#f38ba8";
      }}

      function updateCSIGauge() {{
        const selected = document.getElementById("csiFeatureSelect").value;
        const val = csiData[selected];
        const maxVal = Math.max(0.40, val * 1.15);

        Plotly.newPlot("csiGauge", [{{
          type: "indicator", mode: "gauge+number", value: val,
          gauge: {{
            axis: {{ range: [0, maxVal], tickwidth: 1, tickcolor: "#585b70" }},
            bar: {{ color: "#313244" }}, bgcolor: "white", borderwidth: 2, bordercolor: "#e2e8f0",
            steps: [
              {{ range: [0, 0.10], color: "#e2f9e0" }},
              {{ range: [0.10, 0.25], color: "#fef3c7" }},
              {{ range: [0.25, maxVal], color: "#fee2e2" }}
            ],
            threshold: {{ line: {{ color: "#ef4444", width: 4 }}, value: 0.25 }}
          }}
        }}], {{ margin: {{t:30,b:30,l:30,r:30}}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)" }});

        const colors = csiFeatures.map(f => f === selected ? "#cba6f7" : csiColor(csiData[f]));
        Plotly.newPlot("csiBarChart", [{{
          type: "bar", orientation: "h", x: csiValues, y: csiFeatures,
          marker: {{ color: colors }},
          text: csiValues.map(v => v.toFixed(4)), textposition: "auto"
        }}], {{
          margin: {{t:20,b:40,l:220,r:40}}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
          xaxis: {{ title: "CSI value", gridcolor: "#f1f3f9" }},
          yaxis: {{ autorange: "reversed", gridcolor: "#f1f3f9" }},
          shapes: [
            {{ type:"line", x0:0.10, x1:0.10, y0:-1, y1:csiFeatures.length,
               line:{{dash:"dash",color:"#f59e0b",width:1.5}} }},
            {{ type:"line", x0:0.25, x1:0.25, y0:-1, y1:csiFeatures.length,
               line:{{dash:"dash",color:"#ef4444",width:1.5}} }}
          ]
        }});
      }}
      updateCSIGauge();
    </script>
    """


def _inject_after_body(html_path, sections_html):
    """Splice custom HTML/JS sections right after the <body> tag of an
    already-saved Evidently HTML report, so PSI/CSI render ABOVE Evidently's
    own content (which appears later in the same document).

    Plotly is embedded INLINE (read directly from the locally-installed
    plotly package's bundled plotly.min.js) rather than referenced via a
    CDN <script src="https://...">. A CDN reference would make this report
    silently fail to render its gauges/charts in any environment without
    outbound internet access at VIEW time (e.g. behind a firewall, or if
    the CDN is unreachable) - since this report is meant to be a reliable,
    standalone artefact, embedding the ~4MB library directly guarantees it
    renders correctly regardless of the viewer's network conditions.
    """
    import plotly
    plotly_js_path = os.path.join(os.path.dirname(plotly.__file__),
                                   "package_data", "plotly.min.js")
    with open(plotly_js_path, "r", encoding="utf-8") as f:
        plotly_js = f.read()

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    match = re.search(r"<body[^>]*>", html, re.IGNORECASE)
    if not match:
        print("[warning] could not find <body> tag - PSI/CSI sections not injected.")
        return
    tag = match.group(0)
    plotly_script = f"<script>{plotly_js}</script>\n"
    html = html.replace(tag, tag + "\n" + plotly_script + sections_html, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── main ─────────────────────────────────────────────────────────────────────

def main(baseline_date, current_date, modelname):
    print(f"\n--- Evidently drift report: current={current_date}, model={modelname} ---\n")

    model_path = os.path.join("model_bank", modelname)
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model_tag = artefact["model_version"]
    feature_cols = [c for c in artefact.get("feature_cols", []) if c not in NON_FEATURE_COLS]
    csi_features = _get_top_csi_features(model_tag, feature_cols)

    # PSI baseline is always the model's FIRST scored month (matches
    # model_monitor.py) - not necessarily the --baseline argument, which is
    # kept only for the feature-snapshot (data summary) comparison below.
    psi_baseline_date, psi_baseline_preds = _earliest_scored_month(model_tag)
    current_preds = _load_scores(model_tag, current_date)
    psi_value = _psi(psi_baseline_preds["model_predictions"].values,
                      current_preds["model_predictions"].values)
    print(f"PSI baseline month: {psi_baseline_date}  current: {current_date}  PSI={psi_value}")

    # CSI uses the same fixed first-scored-month baseline, on the feature
    # store partitions (not the predictions).
    csi_baseline_feat = _load_feature_snapshot(psi_baseline_date)
    csi_current_feat = _load_feature_snapshot(current_date)
    csi_max, csi_max_feature, per_feature_csi = _compute_csi_for_snapshot(
        csi_baseline_feat, csi_current_feat, csi_features
    )
    print(f"CSI max: {csi_max} (feature: {csi_max_feature})")

    # Data summary (Evidently) - uses the user-supplied --baseline/--current,
    # since this section is a general data-profile comparison, not tied to
    # the model's deployment-history PSI/CSI baseline.
    baseline_df = _load_feature_snapshot(baseline_date)
    current_df = _load_feature_snapshot(current_date)
    baseline_features = baseline_df[[c for c in baseline_df.columns if c not in NON_FEATURE_COLS]]
    current_features = current_df[[c for c in current_df.columns if c not in NON_FEATURE_COLS]]

    report = Report([DataSummaryPreset()])
    my_eval = report.run(current_features, baseline_features)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUT_DIR,
        f"drift_report_{baseline_date.replace('-', '_')}_vs_{current_date.replace('-', '_')}.html"
    )
    my_eval.save_html(out_path)

    sections = (_psi_section_html(psi_value, psi_baseline_date, current_date)
                + _csi_section_html(per_feature_csi, psi_baseline_date, current_date))
    _inject_after_body(out_path, sections)

    print(f"\nsaved combined report (PSI -> CSI -> data summary): {out_path}")
    print("open this file directly in a browser to view the interactive report.")
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                    help="baseline date for the Evidently data-summary section, YYYY-MM-DD")
    p.add_argument("--current", required=True,
                    help="current (comparison) date, YYYY-MM-DD")
    p.add_argument("--modelname", required=True,
                    help="model artefact filename in model_bank/, e.g. credit_xgb_2024_09_01.pkl")
    a = p.parse_args()
    main(a.baseline, a.current, a.modelname)
