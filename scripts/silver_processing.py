"""Silver cleaning entry point.

    python3 silver_processing.py --snapshotdate 2023-01-01 --source financials
"""
import argparse
from pipeline_common import get_spark
import utils.data_processing_silver_table as silver


def main(snapshotdate, source):
    print(f"\n--- silver {source} {snapshotdate} ---\n")
    spark = get_spark()
    silver.process_silver_table(source, snapshotdate,
                                "datamart/bronze/", "datamart/silver/", spark)
    spark.stop()
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--source", required=True,
                   choices=["lms", "clickstream", "attributes", "financials"])
    a = p.parse_args()
    main(a.snapshotdate, a.source)
