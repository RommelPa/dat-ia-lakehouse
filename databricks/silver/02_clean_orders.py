# Databricks notebook source
# MAGIC %md
# MAGIC # Silver · olist_orders
# MAGIC
# MAGIC Convierte la tabla Bronze de órdenes en una tabla Silver limpia y tipada.
# MAGIC La transformación es deliberadamente conservadora: valida calidad antes de
# MAGIC persistir y falla si encuentra claves nulas, duplicados o fechas no parseables.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
SOURCE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.olist_orders"
TARGET_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.olist_orders"

TIMESTAMP_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# COMMAND ----------
orders_bronze = spark.table(SOURCE_TABLE)

source_count = orders_bronze.count()
print(f"Fuente: {SOURCE_TABLE}")
print(f"Filas Bronze: {source_count:,}")

# COMMAND ----------
# Normalización mínima de texto. Los IDs y estados vacíos pasan a NULL para que
# las validaciones los detecten explícitamente.
orders_silver = (
    orders_bronze
    .withColumn("order_id", F.when(F.trim("order_id") == "", None).otherwise(F.trim("order_id")))
    .withColumn("customer_id", F.when(F.trim("customer_id") == "", None).otherwise(F.trim("customer_id")))
    .withColumn("order_status", F.when(F.trim("order_status") == "", None).otherwise(F.lower(F.trim("order_status"))))
)

for column_name in TIMESTAMP_COLUMNS:
    orders_silver = orders_silver.withColumn(
        column_name,
        F.to_timestamp(F.col(column_name), "yyyy-MM-dd HH:mm:ss"),
    )

orders_silver = orders_silver.withColumn("_processed_at", F.current_timestamp())

# COMMAND ----------
# Controles de calidad antes de escribir Silver.
null_order_ids = orders_silver.filter(F.col("order_id").isNull()).count()
null_customer_ids = orders_silver.filter(F.col("customer_id").isNull()).count()

duplicate_order_ids = (
    orders_silver
    .groupBy("order_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

parse_failures = {}
for column_name in TIMESTAMP_COLUMNS:
    raw_non_null = orders_bronze.filter(
        F.col(column_name).isNotNull() & (F.trim(F.col(column_name)) != "")
    ).count()
    typed_non_null = orders_silver.filter(F.col(column_name).isNotNull()).count()
    parse_failures[column_name] = raw_non_null - typed_non_null

quality = {
    "rows": source_count,
    "null_order_ids": null_order_ids,
    "null_customer_ids": null_customer_ids,
    "duplicate_order_ids": duplicate_order_ids,
    "timestamp_parse_failures": parse_failures,
}

print("Controles de calidad:")
print(quality)

problems = []
if null_order_ids:
    problems.append(f"order_id nulos: {null_order_ids}")
if null_customer_ids:
    problems.append(f"customer_id nulos: {null_customer_ids}")
if duplicate_order_ids:
    problems.append(f"order_id duplicados: {duplicate_order_ids}")

for column_name, failure_count in parse_failures.items():
    if failure_count:
        problems.append(f"{column_name} no parseables: {failure_count}")

if problems:
    raise ValueError("Fallaron controles Silver: " + "; ".join(problems))

# COMMAND ----------
(
    orders_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

# COMMAND ----------
result = spark.table(TARGET_TABLE)
result_count = result.count()

if result_count != source_count:
    raise ValueError(
        f"Conteo inconsistente Bronze={source_count:,} Silver={result_count:,}"
    )

print(f"Tabla creada: {TARGET_TABLE}")
print(f"Filas Silver: {result_count:,}")
result.printSchema()
display(result.limit(20))
