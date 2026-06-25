"""
Silver layer processing.

The silver layer takes raw bronze partitions and produces cleaned, correctly
typed, lightly-enriched records.  Each source has its own cleaning routine:

    lms          - cast types; derive mob, installments_missed, dpd (Main logic)
    attributes   - parse age, validate SSN, null sentinel occupations (Source-2 logic)
    financials   - strip non-numeric junk, clip outliers, split credit-history age
                   and payment-behaviour, build a loan-type frequency table
                   (Source-2 logic, merged in per design decision)
    clickstream  - cast fe_1..fe_20 to int

No business labelling or cross-source joins happen here - that is the gold layer.
"""

import os
import shutil
from collections import Counter
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType, MapType,
)

NUMERIC_REGEX = r"([-+]?\d*\.?\d+)"


# ---------------------------------------------------------------------------
# LMS  (carried over from the Assignment-1 / Main label-store logic)
# ---------------------------------------------------------------------------
def _process_lms(df):
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # days past due
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()),
    ).fillna(0, subset=["installments_missed"])
    df = df.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.add_months(col("snapshot_date"), -1 * col("installments_missed")),
        ).cast(DateType()),
    )
    df = df.withColumn(
        "dpd",
        F.when(
            col("overdue_amt") > 0.0,
            F.datediff(col("snapshot_date"), col("first_missed_date")),
        ).otherwise(0).cast(IntegerType()),
    )
    return df


# ---------------------------------------------------------------------------
# Attributes  (merged from Source 2)
# ---------------------------------------------------------------------------
def _process_attributes(df):
    # Extract numeric part of Age (raw data contains junk like "28_")
    df = df.withColumn("Age", F.regexp_extract(col("Age"), NUMERIC_REGEX, 1))

    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # Valid human age range
    df = df.withColumn(
        "Age",
        F.when((col("Age") >= 0) & (col("Age") <= 120), col("Age")).otherwise(None),
    )
    # Validate SSN format
    df = df.withColumn("SSN", F.regexp_extract(col("SSN"), r"^(\d{3}-\d{2}-\d{4})$", 1))
    df = df.withColumn("SSN", F.when(col("SSN") == "", None).otherwise(col("SSN")))
    # Null sentinel occupation
    df = df.withColumn(
        "Occupation",
        F.when(col("Occupation") == "_______", None).otherwise(col("Occupation")),
    )
    return df


# ---------------------------------------------------------------------------
# Financials  (merged from Source 2; also writes a loan_type frequency table)
# ---------------------------------------------------------------------------
def _split_loan_type(loan_type):
    if not isinstance(loan_type, str):
        return {}
    items = loan_type.replace(" and ", ",").split(",")
    cleaned = [i.strip().replace(" ", "_").lower() for i in items if i.strip() != ""]
    return dict(Counter(cleaned))


def _process_financials(df, silver_directory, snapshot_date_str, spark):
    numeric_cols = {
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Monthly_Balance": FloatType(),
    }
    for c, t in numeric_cols.items():
        df = df.withColumn(c, F.regexp_extract(col(c), NUMERIC_REGEX, 1))
        df = df.withColumn(c, col(c).cast(t))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Split credit-history age "10 Years and 9 Months" -> total months
    df = df.withColumn(
        "credit_history_age_year",
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Year", 1).cast(IntegerType()),
    )
    df = df.withColumn(
        "credit_history_age_month",
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Month", 1).cast(IntegerType()),
    )
    df = df.withColumn(
        "Credit_History_Age",
        (F.coalesce(col("credit_history_age_year"), F.lit(0)) * 12
         + F.coalesce(col("credit_history_age_month"), F.lit(0))).cast(IntegerType()),
    ).drop("credit_history_age_year", "credit_history_age_month")

    # Drop impossible negatives
    for c in ["Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment"]:
        df = df.withColumn(c, F.when(col(c) >= 0, col(c)).otherwise(None))

    # Clip extreme outliers to the 97th percentile
    for c in ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
              "Num_of_Loan", "Num_of_Delayed_Payment"]:
        q = df.approxQuantile(c, [0.97], 0.01)
        if q and q[0] is not None:
            df = df.withColumn(c, F.when(col(c) > q[0], q[0]).otherwise(col(c)))

    # Split payment behaviour into spend-level and value
    pb_regex = r"(Low|High)_spent_(Small|Medium|Large)_value"
    df = df.withColumn("payment_behaviour_spent",
                       F.regexp_extract(col("Payment_Behaviour"), pb_regex, 1))
    df = df.withColumn("payment_behaviour_spent",
                       F.when(col("payment_behaviour_spent") != "",
                              col("payment_behaviour_spent")).otherwise(None))
    df = df.withColumn("payment_behaviour_value",
                       F.regexp_extract(col("Payment_Behaviour"), pb_regex, 2))
    df = df.withColumn("payment_behaviour_value",
                       F.when(col("payment_behaviour_value") != "",
                              col("payment_behaviour_value")).otherwise(None))

    # Null sentinel credit-mix
    df = df.withColumn("Credit_Mix",
                       F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))

    # ---- loan-type frequency table (own silver sub-table) ----
    df_loan = df.select("Customer_ID", "snapshot_date", "Type_of_Loan")
    split_udf = F.udf(_split_loan_type, MapType(StringType(), IntegerType()))
    df_loan = df_loan.withColumn("loan_type_counts", split_udf(col("Type_of_Loan")))
    keys = (
        df_loan.select("loan_type_counts")
        .rdd.flatMap(lambda r: r["loan_type_counts"].keys() if r["loan_type_counts"] else [])
        .distinct().collect()
    )
    for k in keys:
        df_loan = df_loan.withColumn(
            "loan_" + k, F.coalesce(col("loan_type_counts").getItem(k), F.lit(0))
        )
    df_loan = df_loan.drop("loan_type_counts", "Type_of_Loan")

    loan_dir = os.path.join(silver_directory, "loan_type")
    os.makedirs(loan_dir, exist_ok=True)
    loan_path = os.path.join(loan_dir, snapshot_date_str.replace("-", "_") + ".parquet")
    df_loan.write.mode("overwrite").parquet(loan_path)
    print(f"[silver:loan_type] saved to: {loan_path}")

    return df.drop("Payment_Behaviour", "Type_of_Loan")


# ---------------------------------------------------------------------------
# Clickstream
# ---------------------------------------------------------------------------
def _process_clickstream(df):
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def process_silver_table(table_name, snapshot_date_str,
                         bronze_directory, silver_directory, spark):
    """Clean one bronze partition and write the corresponding silver partition."""
    bronze_part = os.path.join(
        bronze_directory, table_name,
        f"bronze_{table_name}_{snapshot_date_str.replace('-', '_')}.csv",
    )
    df = spark.read.csv(bronze_part, header=True, inferSchema=True)
    print(f"[silver:{table_name}] loaded {bronze_part} rows: {df.count()}")

    out_dir = os.path.join(silver_directory, table_name)
    os.makedirs(out_dir, exist_ok=True)

    if table_name == "lms":
        df = _process_lms(df)
    elif table_name == "attributes":
        df = _process_attributes(df)
    elif table_name == "financials":
        df = _process_financials(df, silver_directory, snapshot_date_str, spark)
    elif table_name == "clickstream":
        df = _process_clickstream(df)
    else:
        raise ValueError(f"Unknown source table: {table_name}")

    part = os.path.join(out_dir, snapshot_date_str.replace("-", "_") + ".parquet")
    # Remove any stale partial write from a previous failed attempt before
    # writing. Spark's mode("overwrite") does not reliably clean up a broken
    # _temporary directory left by a prior killed/failed write, causing a
    # ParentNotDirectoryException on retry. Removing the directory first
    # guarantees Spark starts with a clean slate every time.
    if os.path.exists(part):
        shutil.rmtree(part)
        print(f"[silver:{table_name}] cleared stale partition: {part}")
    # Coalesce to 1 partition before writing — avoids the Spark
    # FileNotFoundException: _temporary/0 does not exist error that
    # occurs in resource-constrained environments when Spark tries to
    # stage output across multiple task partitions. Silver partitions
    # are small (typically <10k rows) so single-partition write is fine.
    df.coalesce(1).write.mode("overwrite").parquet(part)
    print(f"[silver:{table_name}] saved to: {part}")
    return df
