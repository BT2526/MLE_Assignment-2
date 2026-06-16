"""
Bronze layer processing.

The bronze layer is the raw landing zone of the medallion architecture.  For
each source system we filter the raw CSV to a single snapshot_date and persist
it verbatim (no cleaning, no type-casting) as a dated CSV partition.  This gives
us a reproducible, replayable record of exactly what arrived from each source
on each date.

Four source systems are ingested:
    lms          - loan management system (daily loan account snapshots)
    clickstream  - behavioural features fe_1..fe_20 (monthly)
    attributes   - customer demographics (name, age, occupation, ...)
    financials   - customer financial profile (income, debt, credit mix, ...)
"""

import os
from datetime import datetime

from pyspark.sql.functions import col


# Maps a logical source name to its raw CSV file on the (simulated) source system.
SOURCE_FILES = {
    "lms": "data/lms_loan_daily.csv",
    "clickstream": "data/feature_clickstream.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
}


def process_bronze_table(table_name, snapshot_date_str, bronze_directory, spark):
    """Filter one source CSV to a single snapshot_date and write a bronze CSV partition.

    Parameters
    ----------
    table_name : str
        One of: lms, clickstream, attributes, financials.
    snapshot_date_str : str
        Snapshot date in 'YYYY-MM-DD' format (Airflow's {{ ds }}).
    bronze_directory : str
        Root of the bronze datamart (e.g. 'datamart/bronze/').
    spark : SparkSession
    """
    if table_name not in SOURCE_FILES:
        raise ValueError(f"Unknown source table: {table_name}")

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Per-source bronze sub-directory, e.g. datamart/bronze/lms/
    out_dir = os.path.join(bronze_directory, table_name)
    os.makedirs(out_dir, exist_ok=True)

    # IRL this would be a query against the source back-end system.
    src = SOURCE_FILES[table_name]
    df = (
        spark.read.csv(src, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date)
    )
    print(f"[bronze:{table_name}] {snapshot_date_str} row count: {df.count()}")

    # Persist raw partition. Pandas write keeps it as a simple single CSV file.
    partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(out_dir, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze:{table_name}] saved to: {filepath}")

    return df
