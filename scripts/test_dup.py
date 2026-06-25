from pipeline_common import get_spark
spark = get_spark('isolatedtest')
df = spark.read.parquet('datamart/silver/lms/2023_07_01.parquet')
print('raw silver rows:', df.count())
filtered = df.filter(df.mob == 6)
print('filtered mob==6 rows:', filtered.count())
print('num partitions:', filtered.rdd.getNumPartitions())
spark.stop()
