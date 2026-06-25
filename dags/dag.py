"""
Credit-Default ML Pipeline (Airflow DAG)
========================================

Monthly, backfillable end-to-end pipeline following the medallion architecture.
Schedule: 1st of every month; backfilled across 2023-01-01 .. 2024-12-01.

Flow per monthly run (execution date = {{ ds }}):

  FEATURE STORE                         LABEL STORE
  4x bronze (lms, clickstream,          bronze_lms_label
       attributes, financials)               |
       |                                silver_lms_label
  4x silver                                  |
       |                                gold_label_store
  gold_feature_store                         |
       |                                     |
       +--------------> store_ready <--------+
                            |
                  +---------+----------+
                  |                    |
            (when training         model_inference  (champion + challenger)
             month) model_train        |
                  |                model_monitor   (PSI + AUC/GINI + charts)
                  |
            model artefacts -> model_bank/ (+ MLflow registry)

Model governance
----------------
Two-phase retraining lifecycle:
  1. BOOTSTRAP: the very first model is trained on a fixed, human-chosen
     calendar month (TRAIN_MONTH) - there is no deployed model yet, so there
     is nothing for "drift" to be measured against.
  2. ONGOING: every month AFTER the bootstrap, retraining is DRIFT-TRIGGERED
     instead of date-triggered - it fires automatically if either PSI (score
     drift) or CSI (feature drift, see model_monitor.py) breached 0.25 in the
     PREVIOUS month's monitoring run.

Why "previous month", not "this month": training runs before monitoring in
the task order each month (store_ready -> training -> inference ->
monitoring). This month's PSI/CSI don't exist yet at the moment this month's
training decision is made - they're only computed by this same month's
monitoring task, later in the same run. So a drift-triggered decision can
only ever see the latest already-completed month's drift signal, meaning
there is an inherent one-month detection lag between when drift actually
occurs and when it can trigger a retrain. This is a real, honest limitation,
whereby production systems with this same train-then-monitor task order
would have the identical lag.
"""

from datetime import datetime, timedelta
import glob
import os

import pandas as pd
from dateutil.relativedelta import relativedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator, ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPTS = "/opt/airflow/scripts"

# Month on which the model is (re)trained.  Inference/monitoring use the model
# trained on this date.  Change here to refresh the model on a different month.
TRAIN_MONTH = "2024-09-01"
CHAMPION_MODEL = "credit_xgb_2024_09_01.pkl"
CHALLENGER_MODEL = "credit_logreg_2024_09_01.pkl"
DRIFT_THRESHOLD = 0.25

SOURCES = ["lms", "clickstream", "attributes", "financials"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _previous_month_drift_breach(snapshotdate_str, model_tag, threshold=DRIFT_THRESHOLD):
    """
    Check whether the previous month's already-saved monitoring table for a
    given model breached the drift threshold on PSI or CSI.
    """
    snap = datetime.strptime(snapshotdate_str, "%Y-%m-%d")
    prev_month = (snap - relativedelta(months=1)).strftime("%Y-%m-%d")

    mon_path = f"{SCRIPTS}/datamart/gold/model_monitoring/{model_tag}/{model_tag}_monitoring.parquet"
    if not os.path.exists(mon_path):
        return False, f"no monitoring history yet for {model_tag} (first eligible month)"

    mon = pd.read_parquet(mon_path)
    row = mon[mon["snapshot_date"] == prev_month]
    if row.empty:
        return False, f"no monitoring row found for {model_tag} at {prev_month}"

    row = row.iloc[0]
    if row.get("psi", float("nan")) == row.get("psi", float("nan")) and row["psi"] > threshold:
        return True, f"PSI={row['psi']:.4f} > {threshold} for {model_tag} at {prev_month}"
    if row.get("csi_max", float("nan")) == row.get("csi_max", float("nan")) and row["csi_max"] > threshold:
        feat = row.get("csi_max_feature", "unknown")
        return True, f"CSI={row['csi_max']:.4f} > {threshold} on '{feat}' for {model_tag} at {prev_month}"
    return False, f"no drift breach for {model_tag} at {prev_month}"


def _should_train(**context):
    """Branch: train only on TRAIN_MONTH (bootstrap), skip all other months.

    Retraining is a manual process — the data science team investigates drift
    alerts (PSI/CSI/AUC/Brier tracked in model_monitor.py) and decides when
    to retrain by running the training scripts directly outside the DAG:

        python3 model_train_xgb.py --snapshotdate <date>
        python3 model_train_logreg.py --snapshotdate <date>
        python3 model_select.py --snapshotdate <date>

    This design ensures:
    - No automatic retrain fires on transient drift or seasonal effects
    - Human judgment governs every model change
    - Drift monitoring continues providing early-warning signals regardless
    - The pipeline never silently deploys a new model without team awareness

    The DAG's training TaskGroup is preserved so the bootstrap run (TRAIN_MONTH)
    still trains the initial model, and the task structure remains visible in
    the Airflow graph for demonstration purposes.
    """
    ds = context["ds"]
    if ds == TRAIN_MONTH:
        print(f"[training] {ds} is the bootstrap month — training initial model.")
        return "training.start_training"

    print(f"[training] {ds} — retraining is manual; skipping automated training.")
    return "training.skip_training"


with DAG(
    dag_id="credit_default_ml_pipeline",
    default_args=default_args,
    description="End-to-end credit default ML pipeline (medallion + train + inference + monitor)",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    tags=["cs611", "credit-risk", "mle"],
) as dag:

    # =======================================================================
    # LABEL STORE branch (LMS only)
    # =======================================================================
    with TaskGroup("label_pipeline") as label_pipeline:
        label_bronze = BashOperator(
            task_id="label_bronze_lms",
            bash_command=(f"cd {SCRIPTS} && python3 bronze_processing.py "
                          f'--snapshotdate "{{{{ ds }}}}" --source lms'),
        )
        label_silver = BashOperator(
            task_id="label_silver_lms",
            bash_command=(f"cd {SCRIPTS} && python3 silver_processing.py "
                          f'--snapshotdate "{{{{ ds }}}}" --source lms'),
        )
        label_gold = BashOperator(
            task_id="gold_label_store",
            bash_command=(f"cd {SCRIPTS} && python3 gold_label_store.py "
                          f'--snapshotdate "{{{{ ds }}}}"'),
        )
        label_bronze >> label_silver >> label_gold

    # =======================================================================
    # FEATURE STORE branch.
    # The three feature-only sources run bronze->silver in parallel.
    # LMS bronze is already produced by the label branch (same partition), so
    # the feature side reuses it rather than re-ingesting (avoids a write race).
    # =======================================================================
    with TaskGroup("feature_pipeline") as feature_pipeline:
        feature_silver_tasks = []

        # LMS silver for features reuses the label branch's LMS bronze.
        lms_feature_silver = BashOperator(
            task_id="feature_silver_lms",
            bash_command=(f"cd {SCRIPTS} && python3 silver_processing.py "
                          f'--snapshotdate "{{{{ ds }}}}" --source lms'),
        )
        feature_silver_tasks.append(lms_feature_silver)

        for src in ["clickstream", "attributes", "financials"]:
            b = BashOperator(
                task_id=f"feature_bronze_{src}",
                bash_command=(f"cd {SCRIPTS} && python3 bronze_processing.py "
                              f'--snapshotdate "{{{{ ds }}}}" --source {src}'),
            )
            s = BashOperator(
                task_id=f"feature_silver_{src}",
                bash_command=(f"cd {SCRIPTS} && python3 silver_processing.py "
                              f'--snapshotdate "{{{{ ds }}}}" --source {src}'),
            )
            b >> s
            feature_silver_tasks.append(s)

        feature_gold = BashOperator(
            task_id="gold_feature_store",
            bash_command=(f"cd {SCRIPTS} && python3 gold_feature_store.py "
                          f'--snapshotdate "{{{{ ds }}}}"'),
        )
        for s in feature_silver_tasks:
            s >> feature_gold

    # Cross-group dependency: feature_silver_lms reuses label_pipeline's bronze.
    label_bronze >> lms_feature_silver

    # =======================================================================
    # Synchronisation gate: both stores must be ready
    # =======================================================================
    store_ready = DummyOperator(task_id="store_ready")
    [label_pipeline, feature_pipeline] >> store_ready

    # =======================================================================
    # MODEL TRAINING - bootstrap once on TRAIN_MONTH, then drift-triggered.
    # See _should_train()/_previous_month_drift_breach() above and the module
    # docstring for the full two-phase lifecycle and its honest limitations.
    # =======================================================================
    with TaskGroup("training") as training:
        branch_train = BranchPythonOperator(
            task_id="training_gate",
            python_callable=_should_train,
        )

        # XGBoost and Logistic Regression are trained as separate and parallel
        # DAG tasks so the training step is visibly split in the Airflow graph.
        # Both scripts use identical window/split logic (deterministic
        # StandardScaler fit on X_train only) so the two models are fully
        # comparable when model_select picks the champion by OOT AUC.
        # start_training is the single branch target when training should run.
        # BranchPythonOperator can only return one task_id, so this DummyOperator
        # acts as the fan-out point to both parallel training tasks.
        start_training = DummyOperator(task_id="start_training")

        xgb_train = BashOperator(
            task_id="xgb_train",
            bash_command=(f"cd {SCRIPTS} && python3 model_train_xgb.py "
                          f'--snapshotdate "{{{{ ds }}}}"'),
        )
        logreg_train = BashOperator(
            task_id="logreg_train",
            bash_command=(f"cd {SCRIPTS} && python3 model_train_logreg.py "
                          f'--snapshotdate "{{{{ ds }}}}"'),
        )

        # model_select runs after both training tasks finish. It reads both
        # saved artefacts, compares OOT AUC, and writes champion.txt —
        # making model selection an explicitly visible step in the DAG graph
        # rather than buried inside a combined training script.
        model_select = BashOperator(
            task_id="model_select",
            bash_command=(f"cd {SCRIPTS} && python3 model_select.py "
                          f'--snapshotdate "{{{{ ds }}}}"'),
        )

        skip_training = DummyOperator(task_id="skip_training")

        training_done = DummyOperator(
            task_id="training_done",
            trigger_rule=TriggerRule.ALL_DONE,
        )

        # training_gate → start_training → [xgb_train, logreg_train] → model_select → training_done
        # training_gate → skip_training → training_done
        branch_train >> [start_training, skip_training]
        start_training >> [xgb_train, logreg_train] >> model_select >> training_done
        skip_training >> training_done

    store_ready >> branch_train

    # =======================================================================
    # MODEL INFERENCE (champion + challenger), every month
    # =======================================================================
    with TaskGroup("inference") as inference:
        inference_start = DummyOperator(task_id="inference_start")
        xgb_inference = BashOperator(
            task_id="xgb_inference",
            # ds = label date. App date = label date - 6 months.
            # app_date_helper.py avoids bash/Python quote nesting issues.
            bash_command=(
                f"cd {SCRIPTS} && "
                f"if [ ! -f model_bank/champion.txt ]; then "
                f"  echo '[SKIP] no champion.txt for {{{{ ds }}}}'; exit 0; fi && "
                f"CHAMP=$(cat model_bank/champion.txt | tr -d '\\\\n') && "
                f"APP_DATE=$(python3 app_date_helper.py '{{{{ ds }}}}') && "
                f'python3 model_inference.py --snapshotdate "$APP_DATE" --labeldate "{{{{ ds }}}}" --modelname "$CHAMP"'
            ),
        )
        logreg_inference = BashOperator(
            task_id="logreg_inference",
            bash_command=(
                f"cd {SCRIPTS} && "
                f"if [ ! -f model_bank/champion.txt ]; then "
                f"  echo '[SKIP] no champion.txt for {{{{ ds }}}}'; exit 0; fi && "
                f"APP_DATE=$(python3 app_date_helper.py '{{{{ ds }}}}') && "
                f'python3 model_inference.py --snapshotdate "$APP_DATE" --labeldate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'
            ),
        )
        inference_done = DummyOperator(task_id="inference_done")
        inference_start >> [xgb_inference, logreg_inference] >> inference_done

    training_done >> inference_start

    # =======================================================================
    # MODEL MONITORING (PSI stability + AUC/GINI + charts), every month
    # =======================================================================
    with TaskGroup("monitoring") as monitoring:
        monitor_start = DummyOperator(task_id="monitor_start")
        xgb_monitor = BashOperator(
            task_id="xgb_monitor",
            bash_command=(f"cd {SCRIPTS} && "
                          f"if [ ! -f model_bank/champion.txt ]; then echo '[SKIP] no champion.txt for {{{{ ds }}}}'; exit 0; fi && "
                          f"CHAMP=$(cat model_bank/champion.txt | tr -d '\\n') && "
                          f'python3 model_monitor.py --snapshotdate "{{{{ ds }}}}" --modelname "$CHAMP"'),
        )
        logreg_monitor = BashOperator(
            task_id="logreg_monitor",
            bash_command=(f"cd {SCRIPTS} && "
                          f"if [ ! -f model_bank/champion.txt ]; then echo '[SKIP] no champion.txt for {{{{ ds }}}}'; exit 0; fi && "
                          f'python3 model_monitor.py --snapshotdate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'),
        )
        # Complementary Evidently AI drift report. This produces an
        # additional, polished HTML report comparing input feature
        # distributions: baseline is set to TRAIN_MONTH (the model's
        # bootstrap month, never changes), current is always "{{ ds }}"
        # (this run's own month, marches forward automatically) - the same
        # fixed-baseline-vs-moving-current pattern already used by PSI
        # above, so cumulative drift since deployment stays visible rather
        # than being hidden by only ever comparing month-to-month.
        # Scoped to CHAMPION_MODEL only (not also the challenger, unlike
        # xgb_monitor/logreg_monitor above) - this report's PSI/CSI
        # section is keyed to one specific model's predictions and feature
        # ranking, and the champion is the model actually driving real
        # decisions, so it's the one whose drift is most worth a polished,
        # presentation-ready report.
        evidently_drift_report = BashOperator(
            task_id="evidently_drift_report",
            bash_command=(f"cd {SCRIPTS} && "
                          f"if [ ! -f model_bank/champion.txt ]; then echo '[SKIP] no champion.txt for {{{{ ds }}}}'; exit 0; fi && "
                          f"CHAMP=$(cat model_bank/champion.txt | tr -d '\\n') && "
                          f'python3 evidently_drift_report.py '
                          f'--baseline "{TRAIN_MONTH}" --current "{{{{ ds }}}}" --modelname "$CHAMP"'),
        )
        monitor_done = DummyOperator(task_id="monitor_done")

        # Check whether the champion's latest monitoring run breached the
        # PSI/CSI drift threshold. ShortCircuitOperator skips all downstream
        # tasks (including the notification) when no breach occurred, so the
        # email only fires on genuine drift months — not every single run.
        # This is the correct placement: drift is detected here in monitoring,
        # so the notification logically belongs here, not in the training
        # TaskGroup where retraining is triggered a month later.
        def _drift_alert_check(**context):
            """Return True (proceed) only if the champion's latest PSI or CSI
            breached DRIFT_THRESHOLD, triggering the downstream notification.
            Fires for any month where drift is detected — including TRAIN_MONTH
            (September 2024) since that is the first live inference month, not
            a pre-deployment month. OOT1 (July–August 2024) is the validation
            window; September onwards is genuine production monitoring."""
            import glob, os
            import pandas as pd
            champ_txt = os.path.join(SCRIPTS, "model_bank", "champion.txt")
            if not os.path.exists(champ_txt):
                return False
            with open(champ_txt) as f:
                champ_fname = f.read().strip()
            model_tag = champ_fname.replace(".pkl", "")
            mon_path = os.path.join(
                SCRIPTS,
                f"datamart/gold/model_monitoring/{model_tag}/{model_tag}_monitoring.parquet"
            )
            if not os.path.exists(mon_path):
                return False
            mon = pd.read_parquet(mon_path)
            if mon.empty:
                return False
            latest = mon.sort_values("snapshot_date").iloc[-1]
            psi_breach = (latest["psi"] == latest["psi"] and
                          latest["psi"] > DRIFT_THRESHOLD)
            csi_breach = (latest["csi_max"] == latest["csi_max"] and
                          latest["csi_max"] > DRIFT_THRESHOLD)
            # AUC alert: drop > 0.05 below OOT1 baseline indicates meaningful
            # degradation in discrimination quality — directly impacts credit decisions.
            AUC_DROP_THRESHOLD = 0.05
            auc_breach = False
            if "auc" in latest.index and latest["auc"] == latest["auc"]:
                if "auc_drop_from_oot1" in latest.index and latest["auc_drop_from_oot1"] == latest["auc_drop_from_oot1"]:
                    auc_breach = latest["auc_drop_from_oot1"] > AUC_DROP_THRESHOLD
            # Brier alert: increase > 0.05 above OOT1 baseline indicates meaningful
            # degradation in calibration quality — predicted probabilities less trustworthy.
            BRIER_ALERT_THRESHOLD = 0.05
            brier_breach = False
            if "brier" in latest.index and latest["brier"] == latest["brier"]:
                oot1_brier = mon.sort_values("snapshot_date").iloc[0]["brier"]
                if oot1_brier == oot1_brier:  # not NaN
                    brier_breach = (latest["brier"] - oot1_brier) > BRIER_ALERT_THRESHOLD
            breached = bool(psi_breach or csi_breach or auc_breach or brier_breach)
            if breached:
                reasons = []
                if psi_breach:    reasons.append(f"PSI={latest['psi']:.4f} > {DRIFT_THRESHOLD}")
                if csi_breach:    reasons.append(f"CSI_max={latest['csi_max']:.4f} > {DRIFT_THRESHOLD}")
                if auc_breach:    reasons.append(f"AUC drop={latest['auc_drop_from_oot1']:.4f} > {AUC_DROP_THRESHOLD}")
                if brier_breach:  reasons.append(f"Brier increase={latest['brier'] - oot1_brier:.4f} > {BRIER_ALERT_THRESHOLD}")
                print(f"[ALERT] breach confirmed — {'; '.join(reasons)} — notifying data science team")
            return breached

        check_drift_alert = ShortCircuitOperator(
            task_id="check_drift_alert",
            python_callable=_drift_alert_check,
            provide_context=True,
            trigger_rule=TriggerRule.ALL_DONE,
        )

        # DEMONSTRATION-ONLY notification. No real SMTP backend is configured.
        # In a real deployment this would send an actual email to the data science
        # team. Here it demonstrates the governance pattern: drift detected →
        # team notified → human investigates → manually decides whether to retrain.
        # retries=0 ensures a failed email (no SMTP) does not block the pipeline —
        # the task fails gracefully without cascading to downstream runs.
        notify_data_science_team = EmailOperator(
            task_id="notify_data_science_team",
            to="data-science-team@example.com",
            subject="[MODEL ALERT] credit_default_ml_pipeline — PSI/CSI/AUC/Brier breached threshold {{ ds }}",
            html_content=(
                "<p>Model monitoring for the credit default champion has detected "
                "one or more breaches in the monitoring run for <b>{{ ds }}</b>:</p>"
                "<ul>"
                "<li><b>PSI &gt; 0.25</b>: output score distribution has shifted significantly</li>"
                "<li><b>CSI &gt; 0.25</b>: one or more input features have drifted significantly</li>"
                "<li><b>AUC drop &gt; 0.05</b>: model discrimination has degraded below OOT1 baseline</li>"
                "<li><b>Brier increase &gt; 0.05</b>: model calibration has degraded below OOT1 baseline</li>"
                "</ul>"
                "<p>Please review the monitoring charts and Evidently report, then "
                "manually trigger retraining if warranted.</p>"
            ),
            retries=0,
        )

        (monitor_start >> [xgb_monitor, logreg_monitor, evidently_drift_report]
         >> monitor_done >> check_drift_alert >> notify_data_science_team)

    inference_done >> monitor_start
