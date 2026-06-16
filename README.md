# CS611 Assignment 2 — Credit Default ML Pipeline

End-to-end, backfillable machine-learning pipeline that predicts loan default at
application time. Built on the medallion architecture, orchestrated with Airflow,
containerised with Docker, and tracked with MLflow.

---

## Quick start

```bash
docker-compose build
docker-compose up
```

Then open:

- **Airflow** → http://localhost:8080  (login `admin` / `admin`)
- **MLflow**  → http://localhost:5000  (experiment runs / model metrics)

Trigger / unpause the `credit_default_ml_pipeline` DAG. With `catchup=True` it
backfills every month from 2023-01-01 to 2024-12-01.

---

## Pipeline architecture

```
                         ┌──────────────── FEATURE STORE ────────────────┐
  bronze_lms ──┬──────►  feature_silver_lms                              │
  (label)      │         feature_bronze_clickstream → feature_silver_*   │
               │         feature_bronze_attributes  → feature_silver_*   ├─► gold_feature_store ─┐
               │         feature_bronze_financials  → feature_silver_*   │                       │
               │                                                          │                       │
               ▼                                                                                  ▼
  label_silver_lms ──► gold_label_store ───────────────────────────────────────────────► store_ready
                                                                                                  │
                                                            branch_should_train ◄─────────────────┘
                                                            │            │
                                                    model_train     skip_training
                                                            └──► training_done
                                                                      │
                                                          inference_start
                                                          ├─ champion_inference  (XGBoost)
                                                          └─ challenger_inference (LogReg)
                                                                      │
                                                          monitor_start
                                                          ├─ champion_monitor    (PSI + AUC/GINI + charts)
                                                          └─ challenger_monitor
```

### Medallion layers

| Layer  | Content | Path |
|--------|---------|------|
| Bronze | Raw per-source, per-date snapshots (CSV) | `datamart/bronze/<source>/` |
| Silver | Cleaned, typed, lightly-enriched (Parquet) | `datamart/silver/<source>/` |
| Gold   | Label store, feature store, predictions, monitoring | `datamart/gold/...` |

---

## Data sources

Own Assignment-1 datamart **plus** features merged from a classmate's
Assignment-1 pipeline (Source 2):

- `lms_loan_daily.csv` — loan account snapshots → labels (`mob`, `dpd`)
- `feature_clickstream.csv` — behavioural features `fe_1..fe_20`
- `features_attributes.csv` — demographics (age, occupation)
- `features_financials.csv` — income, debt, credit profile

**Merged from Source 2:** financials cleaning (regex numeric extraction, outlier
clipping at the 97th percentile, credit-history-age parsing, payment-behaviour
splitting), the loan-type frequency table, one-hot encoding of categoricals, and
mean-aggregated clickstream. Combined with own engineered ratio features
(`debt_to_income`, `emi_to_salary`, `savings_rate`, `payment_stress`,
`loans_per_credit_card`, `monthly_surplus`).

---

## Label definition

`label = 1` if a customer is **≥ 30 days past due at month-on-book 6**, else 0.
Computed from the LMS silver table; written to `datamart/gold/label_store/`.

---

## Leakage controls

- **Target leakage:** every feature is taken **as-of the application month
  (mob = 0)** — the only information available when the lending decision is made.
  The label (observed at mob = 6) never enters the feature store.
- **Temporal leakage:** clickstream history is aggregated only over dates
  `<= application date`. The OOT window is strictly later than train/test.
- **Train-test contamination:** the `StandardScaler` is fit on the **training
  split only**, then applied to test/OOT.

Feature ↔ label alignment is done on `Customer_ID` with a **6-month offset**
(`feature_date + 6 months = label_date`), reflecting the mob-0 → mob-6 gap.

---

## Models & governance

Two models are trained and written to the model bank (`model_bank/`):

| Role | Model | Selection |
|------|-------|-----------|
| Champion | XGBoost (RandomizedSearchCV) | Higher OOT AUC |
| Challenger | Logistic Regression (balanced) | Interpretable baseline |

The champion is chosen by **OOT AUC** and recorded in `model_bank/champion.txt`.
All runs (params, AUC/GINI for train/test/OOT) are logged to MLflow.

**Refresh SOP:** training (AutoML) is gated to a single refresh month
(`TRAIN_MONTH` in `dags/dag.py`) via a `BranchPythonOperator`; inference and
monitoring run every month. In production this gate fires on a schedule
(e.g. quarterly) or on a **drift alert (PSI > 0.25)** from the monitoring task.

---

## Monitoring

`model_monitor.py` produces, per snapshot, across the full time period:

- **Stability** — Population Stability Index (PSI) of the score distribution vs
  the first scored month. Thresholds: <0.1 stable, 0.1–0.25 moderate, >0.25 shift.
- **Performance** — AUC / GINI where matured labels exist (mob-6 outcomes).

Outputs:

- Gold table → `datamart/gold/model_monitoring/<model>/`
- Charts → `datamart/gold/monitoring_plots/<model>/` (`psi_stability.png`,
  `auc_gini_performance.png`)

Recent months show AUC = NaN because their mob-6 labels have not yet matured —
this is exactly why PSI is monitored as a **leading** indicator when outcomes lag.

---

## Project structure

```
.
├── dags/dag.py                 # Airflow orchestration
├── scripts/
│   ├── pipeline_common.py      # shared SparkSession + path setup
│   ├── bronze_processing.py    # bronze entry point (per source)
│   ├── silver_processing.py    # silver entry point (per source)
│   ├── gold_label_store.py     # gold label entry point
│   ├── gold_feature_store.py   # gold feature entry point
│   ├── model_train.py          # AutoML: XGBoost + LogReg, MLflow
│   ├── model_inference.py      # batch scoring → gold predictions
│   └── model_monitor.py        # PSI + AUC/GINI + charts
├── utils/
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   └── data_processing_gold_table.py
├── data/                       # raw source CSVs
├── datamart/                   # bronze / silver / gold outputs
├── model_bank/                 # serialised model artefacts (.pkl)
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── Readme.txt                  # GitHub repo link (per submission spec)
```

---

## Notes for graders

- `docker-compose up` initialises the Airflow metadata DB (Postgres), creates the
  `admin` user, and starts the webserver + scheduler. The DAG is unpaused by
  default and begins backfilling immediately.
- PySpark requires Java; the image installs OpenJDK 17 and sets `JAVA_HOME`.
- Training is intentionally gated to one month so a full backfill is fast; for
  months before the refresh month, inference/monitoring skip gracefully (no model
  in the bank yet) so the backfill stays green.
```
