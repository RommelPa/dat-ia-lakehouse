# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze · validación del baseline Olist
# MAGIC
# MAGIC Verifica que las 9 tablas Bronze existan, conserven el volumen esperado
# MAGIC del dataset Olist cargado y mantengan el contrato raw: columnas de negocio
# MAGIC como `string`, más `_ingested_at` y `_source_file` como metadata técnica.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, TimestampType

CATALOG = "dat_ia"
BRONZE_SCHEMA = "bronze"

EXPECTED_ROWS = {
    "olist_customers": 99_441,
    "olist_geolocation": 1_000_163,
    "olist_order_items": 112_650,
    "olist_order_payments": 103_886,
    "olist_order_reviews": 99_224,
    "olist_orders": 99_441,
    "olist_products": 32_951,
    "olist_sellers": 3_095,
    "product_category_name_translation": 71,
}

# COMMAND ----------
missing_tables = []
for table_name in EXPECTED_ROWS:
    full_name = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"
    if not spark.catalog.tableExists(full_name):
        missing_tables.append(full_name)

if missing_tables:
    raise ValueError(
        "Faltan tablas Bronze: " + ", ".join(sorted(missing_tables))
    )

print("Contrato de presencia OK: las 9 tablas Bronze existen.")

# COMMAND ----------
summary = []
problems = []

for table_name, expected_rows in EXPECTED_ROWS.items():
    full_name = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"
    df = spark.table(full_name)

    actual_rows = df.count()
    row_count_ok = actual_rows == expected_rows

    fields = {field.name: field.dataType for field in df.schema.fields}
    ingested_type_ok = isinstance(fields.get("_ingested_at"), TimestampType)
    source_type_ok = isinstance(fields.get("_source_file"), StringType)

    business_fields = {
        name: data_type
        for name, data_type in fields.items()
        if name not in {"_ingested_at", "_source_file"}
    }
    business_strings_ok = all(
        isinstance(data_type, StringType)
        for data_type in business_fields.values()
    )

    null_source_files = df.filter(F.col("_source_file").isNull()).count()
    null_ingested_at = df.filter(F.col("_ingested_at").isNull()).count()

    table_ok = all(
        [
            row_count_ok,
            ingested_type_ok,
            source_type_ok,
            business_strings_ok,
            null_source_files == 0,
            null_ingested_at == 0,
        ]
    )

    summary.append(
        {
            "table": full_name,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "row_count_ok": row_count_ok,
            "business_strings_ok": business_strings_ok,
            "metadata_types_ok": ingested_type_ok and source_type_ok,
            "null_source_files": null_source_files,
            "null_ingested_at": null_ingested_at,
            "status": "ok" if table_ok else "failed",
        }
    )

    if not row_count_ok:
        problems.append(
            f"{full_name}: filas esperadas={expected_rows:,}, reales={actual_rows:,}"
        )
    if not business_strings_ok:
        problems.append(f"{full_name}: hay columnas raw que no son string")
    if not ingested_type_ok or not source_type_ok:
        problems.append(f"{full_name}: tipos de metadata inválidos")
    if null_source_files:
        problems.append(f"{full_name}: _source_file nulos={null_source_files}")
    if null_ingested_at:
        problems.append(f"{full_name}: _ingested_at nulos={null_ingested_at}")

# COMMAND ----------
summary_df = spark.createDataFrame(summary)
display(summary_df.orderBy("table"))

if problems:
    raise ValueError("Falló validación Bronze: " + "; ".join(problems))

print("Baseline Bronze validado correctamente.")
print(f"Tablas verificadas: {len(summary)}")
print(f"Filas totales Bronze: {sum(item['actual_rows'] for item in summary):,}")
