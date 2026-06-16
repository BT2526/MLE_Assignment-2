"""Gold label-store entry point.

    python3 gold_label_store.py --snapshotdate 2023-07-01
"""
import argparse
from pipeline_common import get_spark
import utils.data_processing_gold_table as gold


def main(snapshotdate):
    print(f"\n--- gold label {snapshotdate} ---\n")
    spark = get_spark()
    gold.process_labels_gold_table(
        snapshotdate, "datamart/silver/",
        "datamart/gold/label_store/", spark, dpd=30, mob=6,
    )
    spark.stop()
    print("\n--- done ---\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.snapshotdate)
