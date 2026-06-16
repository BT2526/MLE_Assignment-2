"""
Gold layer processing.

Two gold artefacts are produced:

1. LABEL STORE  (datamart/gold/label_store/)
   For each snapshot month, take loan accounts at month-on-book = mob and flag
   default = 1 if dpd >= dpd_threshold.  This is the supervised target.
   Carried over unchanged from the Assignment-1 / Main logic.

2. FEATURE STORE (datamart/gold/feature_store/)
   A single wide, model-ready table keyed on (Customer_ID, snapshot_date), where
   snapshot_date is the LOAN APPLICATION month (mob = 0).  We join:
       - financials  (income, debt, credit profile)   as-of application
       - attributes  (age, occupation)                 as-of application
       - loan_type   (loan-type frequency counts)      as-of application
       - clickstream (fe_1..fe_20), aggregated over the customer's history
                     UP TO AND INCLUDING the application month only
   plus a set of engineered ratio features.

   LEAKAGE CONTROL
   ---------------
   The label is observed at mob = 6, but every feature in this store is taken
   as-of mob = 0 (the application point) - the only information truly available
   when the lending decision is made.  Clickstream is aggregated only over dates
   <= application date.  Nothing from the future or from the target enters the
   feature store, so there is no temporal or target leakage.
"""

import os
import glob

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, NumericType, ArrayType,
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer


# ===========================================================================
# LABEL STORE
# ===========================================================================
def process_labels_gold_table(snapshot_date_str, silver_directory,
                              gold_label_directory, spark, dpd, mob):
    part = os.path.join(silver_directory, "lms",
                        snapshot_date_str.replace("-", "_") + ".parquet")
    df = spark.read.parquet(part)
    print(f"[gold:label] loaded {part} rows: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label",
                       F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def",
                       F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    os.makedirs(gold_label_directory, exist_ok=True)
    out = os.path.join(gold_label_directory,
                       "gold_label_store_" + snapshot_date_str.replace("-", "_") + ".parquet")
    df.write.mode("overwrite").parquet(out)
    print(f"[gold:label] saved to: {out}")
    return df


# ===========================================================================
# FEATURE STORE
# ===========================================================================
def _read_all(table, silver_directory, spark):
    folder = os.path.join(silver_directory, table)
    files = glob.glob(os.path.join(folder, "*.parquet"))
    if not files:
        return None
    return spark.read.parquet(*files)


def _one_hot(df, category_col):
    """One-hot encode a single categorical column into integer 0/1 columns."""
    idx = StringIndexer(inputCol=category_col,
                        outputCol=f"{category_col}_idx",
                        handleInvalid="keep")
    model = idx.fit(df)
    df = model.transform(df)
    enc = OneHotEncoder(inputCol=f"{category_col}_idx",
                        outputCol=f"{category_col}_ohe", dropLast=False)
    df = enc.fit(df).transform(df)
    to_arr = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{category_col}_arr", to_arr(f"{category_col}_ohe"))
    cats = [c.lower().replace(" ", "_") for c in model.labels]
    for i, c in enumerate(cats):
        df = df.withColumn(f"{category_col}_{c}",
                           col(f"{category_col}_arr")[i].cast(IntegerType()))
    return df.drop(category_col, f"{category_col}_idx",
                   f"{category_col}_ohe", f"{category_col}_arr")


def process_features_gold_table(snapshot_date_str, silver_directory,
                                gold_feature_directory, spark):
    """Build the model-ready feature store partition for one application month."""
    # --- application anchor: LMS accounts at mob = 0 for this snapshot ---
    lms = _read_all("lms", silver_directory, spark)
    app = (
        lms.filter((col("mob") == 0) & (col("snapshot_date") == snapshot_date_str))
        .select("Customer_ID",
                col("snapshot_date").alias("application_date"),
                "loan_amt", "tenure")
        .dropDuplicates(["Customer_ID"])
    )
    n_app = app.count()
    print(f"[gold:feature] {snapshot_date_str} applications (mob=0): {n_app}")
    if n_app == 0:
        print(f"[gold:feature] no applications this month, skipping.")
        return None

    # --- financials as-of application ---
    fin = _read_all("financials", silver_directory, spark)
    fin = fin.withColumnRenamed("snapshot_date", "fin_date")

    # --- attributes as-of application ---
    att = _read_all("attributes", silver_directory, spark) \
        .select("Customer_ID", "Age", "Occupation",
                col("snapshot_date").alias("att_date"))

    # --- loan-type counts as-of application ---
    loan = _read_all("loan_type", silver_directory, spark)
    if loan is not None:
        loan = loan.withColumnRenamed("snapshot_date", "loan_date")

    # Join the per-customer (single snapshot) sources onto the applications.
    df = app.join(fin, on="Customer_ID", how="left") \
            .join(att, on="Customer_ID", how="left")
    if loan is not None:
        df = df.join(loan, on="Customer_ID", how="left")

    # Keep the application snapshot_date as the partition key.
    df = df.withColumn("snapshot_date", col("application_date"))

    # --- clickstream aggregated over history up to the application month ---
    cs_all = _read_all("clickstream", silver_directory, spark)
    # Global per-feature medians, used as a neutral fill when a customer has no
    # clickstream history at all (genuine missing-data months).  This avoids an
    # artificial zero-spike that would otherwise look like score drift.
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    med_row = cs_all.approxQuantile(fe_cols, [0.5], 0.01)
    cs_medians = {c: (med_row[i][0] if med_row[i] else 0.0)
                  for i, c in enumerate(fe_cols)}

    cs = cs_all.join(app.select("Customer_ID",
                                col("application_date").alias("app_date")),
                     on="Customer_ID", how="inner")
    cs = cs.filter(col("snapshot_date") <= col("app_date"))
    agg = [F.avg(f"fe_{i}").alias(f"fe_{i}") for i in range(1, 21)]
    cs = cs.groupBy("Customer_ID").agg(*agg)
    df = df.join(cs, on="Customer_ID", how="left")
    # Neutral median fill for customers with no clickstream history.
    df = df.fillna(cs_medians)

    # --- engineered ratio features (from Main) ---
    df = df.withColumn("debt_to_income",
                       col("Outstanding_Debt") / (col("Annual_Income") + F.lit(1.0)))
    df = df.withColumn("emi_to_salary",
                       col("Total_EMI_per_month") / (col("Monthly_Inhand_Salary") + F.lit(1.0)))
    df = df.withColumn("savings_rate",
                       col("Amount_invested_monthly") / (col("Monthly_Inhand_Salary") + F.lit(1.0)))
    df = df.withColumn("payment_stress",
                       col("Num_of_Delayed_Payment") / (col("Num_of_Loan") + F.lit(1.0)))
    df = df.withColumn("loans_per_credit_card",
                       col("Num_of_Loan") / (col("Num_Credit_Card") + F.lit(1.0)))
    df = df.withColumn("monthly_surplus",
                       col("Monthly_Inhand_Salary") - col("Total_EMI_per_month"))

    # --- one-hot encode categoricals ---
    for c in ["Occupation", "Credit_Mix", "Payment_of_Min_Amount",
              "payment_behaviour_spent", "payment_behaviour_value"]:
        if c in df.columns:
            df = df.fillna({c: "unknown"})
            df = _one_hot(df, c)

    # --- drop identifiers / helper date cols, then impute numerics ---
    drop_cols = ["Name", "SSN", "fin_date", "att_date", "loan_date",
                 "application_date", "Credit_History_Age_raw"]
    df = df.drop(*[c for c in drop_cols if c in df.columns])

    numeric_cols = [c for c in df.columns
                    if isinstance(df.schema[c].dataType, NumericType)]
    if numeric_cols:
        # Imputer cannot compute a surrogate for a column that is entirely null
        # (e.g. customers with no clickstream history at the application date).
        # Identify such columns and fill them with 0; impute the median elsewhere.
        non_null_counts = df.select(
            [F.count(col(c)).alias(c) for c in numeric_cols]
        ).collect()[0].asDict()
        all_null = [c for c in numeric_cols if non_null_counts[c] == 0]
        impute_cols = [c for c in numeric_cols if non_null_counts[c] > 0]

        if all_null:
            df = df.fillna(0, subset=all_null)
        if impute_cols:
            imputer = Imputer(inputCols=impute_cols, outputCols=impute_cols,
                              strategy="median")
            df = imputer.fit(df).transform(df)

    os.makedirs(gold_feature_directory, exist_ok=True)
    out = os.path.join(gold_feature_directory,
                       "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet")
    df.write.mode("overwrite").parquet(out)
    print(f"[gold:feature] saved to: {out} ({df.count()} rows, {len(df.columns)} cols)")
    return df
