"""Bronze ingestion entry point. Ingests one source table for one snapshot date.

    python3 bronze_processing.py --snapshotdate 2023-01-01 --source lms
"""
import argparse
from pipeline_common import get_spark
import utils.data_processing_bronze_table as bronze


def main(snapshotdate, source):
    print(f"\n--- bronze {source} {snapshotdate} ---\n")
    spark = get_spark()
    bronze.process_bronze_table(source, snapshotdate, "datamart/bronze/", spark)
    spark.stop()
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--source", required=True,
                   choices=["lms", "clickstream", "attributes", "financials"])
    a = p.parse_args()
    main(a.snapshotdate, a.source)
