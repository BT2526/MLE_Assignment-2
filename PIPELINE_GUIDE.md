# End-to-End Pipeline Guide

## Overview

This is a monthly, backfillable credit-default ML pipeline built on Apache Airflow, following the **medallion architecture** (Bronze → Silver → Gold). It covers data ingestion, feature engineering, model training, inference, and monitoring — orchestrated as a single Airflow DAG that automatically backfills from January 2023 to December 2024.

### Architecture

```
FEATURE STORE                         LABEL STORE
4x bronze sources                     bronze_lms_label
(lms, clickstream,                         |
 attributes, financials)              silver_lms_label
       |                                   |
4x silver                            gold_label_store
       |                                   |
gold_feature_store                         |
       +-----------> store_ready <---------+
                         |
               +---------+---------+
               |                   |
         model_train           model_inference
         (bootstrap or         (champion + challenger)
          drift-triggered)          |
               |               model_monitor
         model_bank/           (PSI + CSI + AUC/Brier
         + MLflow registry      + Evidently report)
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **8 GB RAM** allocated to Docker (Airflow + Spark are memory-intensive)
- Ports **8080** (Airflow) and **5001** (MLflow) free

---

## Quick Start

### 1. Clone the repository and navigate to the project root

```bash
git clone https://github.com/BT2526/cs611-assignment2
cd <repo-folder>
```

### 2. Build and start all services

```bash
docker compose up --build
```

This will:
- Build the custom Airflow image (with Java 17 + all Python dependencies)
- Start PostgreSQL (Airflow metadata DB)
- Run `airflow-init` (DB migration + create admin user) — exits with code 0 when done
- Start the Airflow webserver, scheduler, and MLflow UI

> First build takes ~5–10 minutes depending on your machine. Subsequent starts are much faster.

### 3. Create required directories (to modify CLI commands accordingly based on parent folder)

```bash
mkdir -p scripts/mlruns
```

### 4. Verify services are healthy

```bash
docker ps
```

Expected status:

| Container | Status |
|---|---|
| `assignment2-postgres-1` | Up (healthy) |
| `assignment2-airflow-init-1` | Exited (0) |
| `assignment2-airflow-webserver-1` | Up (healthy) |
| `assignment2-airflow-scheduler-1` | Up |
| `assignment2-mlflow-ui-1` | Up |

---

## Accessing the UIs

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| MLflow | http://localhost:5001 | _(no login required)_ |

---

## Running the Pipeline

### Automatic backfill (default behaviour)

The DAG `credit_default_ml_pipeline` is configured with:
- `start_date`: 2023-01-01
- `end_date`: 2024-12-01
- `schedule_interval`: 1st of every month
- `catchup=True`, `max_active_runs=1`

Once the scheduler starts, it will **automatically backfill all 24 months** sequentially (one run at a time). No manual trigger is needed.

Monitor progress in the Airflow UI under **DAGs → credit_default_ml_pipeline → Grid view**.

### Manual trigger (single month)

To trigger a specific month manually from the Airflow UI:

1. Go to **DAGs → credit_default_ml_pipeline**
2. Click **Trigger DAG w/ config**
3. Set `logical_date` to the 1st of the desired month (e.g., `2024-09-01`)

Or via CLI inside the container (to modify CLI commands accordingly based on parent folder):

```bash
docker compose exec airflow-scheduler \
  airflow dags trigger credit_default_ml_pipeline \
  --exec-date 2024-09-01
```

---

## Pipeline Stages (per monthly run)

### Stage 1 — Label Store (`label_pipeline` task group)
Processes the LMS source through Bronze → Silver → Gold to produce the `gold_label_store` (ground-truth default labels).

### Stage 2 — Feature Store (`feature_pipeline` task group)
Processes all four sources (LMS, clickstream, attributes, financials) through Bronze → Silver in parallel, then joins into `gold_feature_store`. LMS bronze is shared with the label branch to avoid duplicate writes.

### Stage 3 — `store_ready`
A synchronisation gate — both stores must complete before training/inference begins.

### Stage 4 — Model Training (`training` task group)
- **Bootstrap month only (2024-09-01):** trains XGBoost and Logistic Regression in parallel, then runs `model_select.py` to pick the champion by OOT AUC and write `model_bank/champion.txt`.
- **All other months:** training is skipped (the `skip_training` branch). Retraining is a **manual process** — the data science team reviews drift alerts and re-runs the training scripts directly if warranted.

Model artefacts are saved to `model_bank/` and registered in MLflow (viewable at http://localhost:5001).

### Stage 5 — Model Inference (`inference` task group)
Runs both the champion and challenger models against the current month's feature store. Skipped automatically if `champion.txt` does not yet exist (i.e., pre-bootstrap months).

### Stage 6 — Model Monitoring (`monitoring` task group)
- Computes **PSI** (score distribution drift) and **CSI** (feature drift) against the bootstrap month baseline
- Tracks **AUC** and **Brier score** vs. OOT1 baseline
- Generates an **Evidently AI HTML drift report** for the champion model
- If any metric breaches its threshold (PSI/CSI > 0.25, AUC drop > 0.05, Brier increase > 0.05), fires a `notify_data_science_team` alert

> Drift alert emails are delivered via MailHog (configured as the SMTP backend). View them at http://localhost:8025 — no real email is sent externally.

---

## Model Governance

| Phase | Trigger | Action |
|---|---|---|
| Bootstrap | `ds == 2024-09-01` | Train initial XGBoost + LogReg, select champion |
| Ongoing | Every month | Inference + monitoring only; no auto-retrain |
| Drift detected | PSI/CSI/AUC/Brier breach | Alert fired; **human decides** whether to retrain |

To manually retrain outside the DAG:

```bash
docker compose exec airflow-scheduler bash -c "
  cd /opt/airflow/scripts &&
  python3 model_train_xgb.py --snapshotdate 2024-09-01 &&
  python3 model_train_logreg.py --snapshotdate 2024-09-01 &&
  python3 model_select.py --snapshotdate 2024-09-01
"
```

---

## Viewing Outputs

### MLflow experiment runs
Go to http://localhost:5001 — each training run logs parameters, metrics, and the model artefact.

### Monitoring reports (Evidently HTML)
Generated inside the container at:
```
/opt/airflow/scripts/reports/
```

To copy to your local machine (to modify CLI commands accordingly based on parent folder):
```bash
docker compose cp airflow-scheduler:/opt/airflow/scripts/reports ./reports
```

### Datamart files
The `datamart/` directory is mounted as a volume — Gold-layer parquet files are written directly to your local `datamart/` folder and are accessible without `docker cp`.

---

## Stopping the Stack

```bash
docker compose down
```

To also remove the PostgreSQL volume (resets Airflow DB and DAG run history):

```bash
docker compose down -v
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Webserver not healthy after 2 min | Slow first build | Wait; watch `docker compose logs airflow-webserver` |
| DAG not appearing in UI | Scheduler hasn't picked it up yet | Wait ~30s after webserver is healthy |
| `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'` | Snowflake connector import conflict | Harmless — Snowflake is not used in this pipeline |
| `OSError while attempting to symlink` in init logs | Log directory symlink race | Harmless — init exits 0 successfully |
| Task fails with `No such file or directory` | Bronze/silver file not yet written for that month | Check the upstream task's logs for the real error |

---

## Key Configuration

All tuneable constants live at the top of [dags/dag.py](dags/dag.py):

```python
TRAIN_MONTH      = "2024-09-01"   # Bootstrap training month
CHAMPION_MODEL   = "credit_xgb_2024_09_01.pkl"
CHALLENGER_MODEL = "credit_logreg_2024_09_01.pkl"
DRIFT_THRESHOLD  = 0.25           # PSI/CSI alert threshold
```
