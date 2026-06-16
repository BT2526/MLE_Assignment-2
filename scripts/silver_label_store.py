import argparse
import os
import pyspark

import utils.data_processing_silver_table

# to call this script: python silver_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Input: bronze CSV written by bronze_label_store.py
    bronze_lms_directory = "datamart/bronze/lms/"

    # Output: silver Parquet with cleaned schema + engineered features (mob, dpd)
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    if not os.path.exists(silver_loan_daily_directory):
        os.makedirs(silver_loan_daily_directory)

    utils.data_processing_silver_table.process_silver_table(
        snapshotdate, bronze_lms_directory, silver_loan_daily_directory, spark
    )

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
