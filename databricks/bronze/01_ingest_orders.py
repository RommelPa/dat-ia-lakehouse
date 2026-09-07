# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze · olist_orders
# MAGIC
# MAGIC Primer paso funcional del Lakehouse. Lee el CSV original de Olist desde
# MAGIC un Volume administrado y persiste una tabla Delta en `dat_ia.bronze`.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
LANDING_SCHEMA = "landing"
BRONZE_SCHEMA = "bronze"
VOLUME = "olist_raw"
SOURCE_FILE = "olist_orders_dataset.csv"
SOURCE_PATH = f"/Volumes/{CATALOG}/{LANDING_SCHEMA}/{VOLUME}/{SOURCE_FILE}"
TARGET_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.olist_orders"

# COMMAND ----------
orders_raw = (
    spark.read.option("header", True)
    .option("inferSchema", False)
    .option("multiLine", False)
    .option("escape", '"')
    .csv(SOURCE_PATH)
)

# COMMAND ----------
orders_bronze = (
    orders_raw
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

# COMMAND ----------
(
    orders_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

# COMMAND ----------
result = spark.table(TARGET_TABLE)
print(f"Tabla creada: {TARGET_TABLE}")
print(f"Filas: {result.count():,}")
result.printSchema()
display(result.limit(20))
