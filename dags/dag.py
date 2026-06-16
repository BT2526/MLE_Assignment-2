"""
Credit-Default ML Pipeline (Airflow DAG)
========================================

Monthly, backfillable end-to-end pipeline following the medallion architecture.
Schedule: 1st of every month; backfilled across 2023-01-01 .. 2024-12-01.

Flow per monthly run (execution date = {{ ds }}):

  FEATURE STORE                         LABEL STORE
  4x bronze (lms, clickstream,          bronze_lms_label
       attributes, financials)               |
       |                                 silver_lms_label
  4x silver                                   |
       |                                 gold_label_store
  gold_feature_store                          |
       |                                       |
       +--------------> store_ready <----------+
                            |
                  +---------+----------+
                  |                    |
            (when training         model_inference  (champion + challenger)
             month) model_train         |
                  |                  model_monitor   (PSI + AUC/GINI + charts)
                  |
            model artefacts -> model_bank/ (+ MLflow registry)

Model governance
----------------
Training (AutoML) runs only on a designated refresh month (TRAIN_MONTH) so the
backfill stays fast and reproducible; inference/monitoring run every month.
In production this gate would fire on a schedule (e.g. quarterly) or on a drift
alert (PSI > 0.25) raised by the monitoring task.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
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

SOURCES = ["lms", "clickstream", "attributes", "financials"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _should_train(**context):
    """Branch: only run AutoML training on the configured refresh month."""
    ds = context["ds"]
    return "model_train" if ds == TRAIN_MONTH else "skip_training"


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
    # FEATURE STORE branch (4 sources in parallel through bronze + silver)
    # =======================================================================
    # =======================================================================
    # FEATURE STORE branch.
    # The three feature-only sources run bronze->silver in parallel.
    # LMS bronze is already produced by the label branch (same partition), so
    # the feature side reuses it rather than re-ingesting (avoids a write race).
    # =======================================================================
    feature_silver_tasks = []

    # LMS silver for features reuses the label branch's LMS bronze.
    lms_feature_silver = BashOperator(
        task_id="feature_silver_lms",
        bash_command=(f"cd {SCRIPTS} && python3 silver_processing.py "
                      f'--snapshotdate "{{{{ ds }}}}" --source lms'),
    )
    label_bronze >> lms_feature_silver
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

    # =======================================================================
    # Synchronisation gate: both stores must be ready
    # =======================================================================
    store_ready = DummyOperator(task_id="store_ready")
    [label_gold, feature_gold] >> store_ready

    # =======================================================================
    # MODEL TRAINING (AutoML) - gated to the refresh month only
    # =======================================================================
    branch_train = BranchPythonOperator(
        task_id="branch_should_train",
        python_callable=_should_train,
    )
    model_train = BashOperator(
        task_id="model_train",
        bash_command=(f"cd {SCRIPTS} && python3 model_train.py "
                      f'--snapshotdate "{{{{ ds }}}}"'),
    )
    skip_training = DummyOperator(task_id="skip_training")
    training_done = DummyOperator(
        task_id="training_done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    store_ready >> branch_train >> [model_train, skip_training] >> training_done

    # =======================================================================
    # MODEL INFERENCE (champion + challenger), every month
    # =======================================================================
    inference_start = DummyOperator(task_id="inference_start")
    champion_inference = BashOperator(
        task_id="champion_inference",
        bash_command=(f"cd {SCRIPTS} && python3 model_inference.py "
                      f'--snapshotdate "{{{{ ds }}}}" --modelname "{CHAMPION_MODEL}"'),
    )
    challenger_inference = BashOperator(
        task_id="challenger_inference",
        bash_command=(f"cd {SCRIPTS} && python3 model_inference.py "
                      f'--snapshotdate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'),
    )
    inference_done = DummyOperator(task_id="inference_done")
    training_done >> inference_start
    inference_start >> [champion_inference, challenger_inference] >> inference_done

    # =======================================================================
    # MODEL MONITORING (PSI stability + AUC/GINI + charts), every month
    # =======================================================================
    monitor_start = DummyOperator(task_id="monitor_start")
    champion_monitor = BashOperator(
        task_id="champion_monitor",
        bash_command=(f"cd {SCRIPTS} && python3 model_monitor.py "
                      f'--snapshotdate "{{{{ ds }}}}" --modelname "{CHAMPION_MODEL}"'),
    )
    challenger_monitor = BashOperator(
        task_id="challenger_monitor",
        bash_command=(f"cd {SCRIPTS} && python3 model_monitor.py "
                      f'--snapshotdate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'),
    )
    monitor_done = DummyOperator(task_id="monitor_done")
    inference_done >> monitor_start
    monitor_start >> [champion_monitor, challenger_monitor] >> monitor_done
