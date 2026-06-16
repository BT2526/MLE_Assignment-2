"""Shared helpers used by every pipeline script."""

import sys
import os

import pyspark


def get_spark(app_name="credit_risk_pipeline"):
    spark = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# Allow `import utils.xxx` whether run from /opt/airflow/scripts or repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
