# Databricks notebook source
# MAGIC %md
# MAGIC # Silver · fuentes Olist
# MAGIC
# MAGIC Transforma las 9 tablas Bronze originales de Olist hacia `dat_ia.silver`
# MAGIC mediante tipado explícito, normalización mínima y controles de calidad.
# MAGIC `olist_orders` se omite por defecto porque ya fue creada y validada en
# MAGIC `02_clean_orders.py`.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

CATALOG = "dat_ia"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
OVERWRITE_EXISTING = False

TABLES = {
    "olist_customers": {
        "casts": {"customer_zip_code_prefix": "int"},
        "required": ["customer_id", "customer_unique_id"],
        "unique_key": ["customer_id"],
        "lower": ["customer_city"],
        "upper": ["customer_state"],
    },
    "olist_geolocation": {
        "casts": {
            "geolocation_zip_code_prefix": "int",
            "geolocation_lat": "double",
            "geolocation_lng": "double",
        },
        "required": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
        ],
        "unique_key": None,
        "lower": ["geolocation_city"],
        "upper": ["geolocation_state"],
    },
    "olist_order_items": {
        "casts": {
            "order_item_id": "int",
            "shipping_limit_date": "timestamp",
            "price": "decimal(12,2)",
            "freight_value": "decimal(12,2)",
        },
        "required": ["order_id", "order_item_id", "product_id", "seller_id"],
        "unique_key": ["order_id", "order_item_id"],
        "lower": [],
        "upper": [],
    },
    "olist_order_payments": {
        "casts": {
            "payment_sequential": "int",
            "payment_installments": "int",
            "payment_value": "decimal(12,2)",
        },
        "required": ["order_id", "payment_sequential", "payment_type"],
        "unique_key": ["order_id", "payment_sequential"],
        "lower": ["payment_type"],
        "upper": [],
    },
    "olist_order_reviews": {
        "casts": {
            "review_score": "int",
            "review_creation_date": "timestamp",
            "review_answer_timestamp": "timestamp",
        },
        "required": ["review_id", "order_id", "review_score"],
        "unique_key": None,
        "lower": [],
        "upper": [],
    },
    "olist_orders": {
        "casts": {},
        "required": ["order_id", "customer_id"],
        "unique_key": ["order_id"],
        "lower": ["order_status"],
        "upper": [],
    },
    "olist_products": {
        "casts": {
            "product_name_lenght": "int",
            "product_description_lenght": "int",
            "product_photos_qty": "int",
            "product_weight_g": "int",
            "product_length_cm": "int",
            "product_height_cm": "int",
            "product_width_cm": "int",
        },
        "required": ["product_id"],
        "unique_key": ["product_id"],
        "lower": ["product_category_name"],
        "upper": [],
    },
    "olist_sellers": {
        "casts": {"seller_zip_code_prefix": "int"},
        "required": ["seller_id"],
        "unique_key": ["seller_id"],
        "lower": ["seller_city"],
        "upper": ["seller_state"],
    },
    "product_category_name_translation": {
        "casts": {},
        "required": ["product_category_name", "product_category_name_english"],
        "unique_key": ["product_category_name"],
        "lower": ["product_category_name", "product_category_name_english"],
        "upper": [],
    },
}

# COMMAND ----------
missing_bronze = []
for table_name in TABLES:
    full_name = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"
    if not spark.catalog.tableExists(full_name):
        missing_bronze.append(full_name)

if missing_bronze:
    raise ValueError(
        "Faltan tablas Bronze requeridas: " + ", ".join(sorted(missing_bronze))
    )

print("Preflight OK: las 9 tablas Bronze están disponibles.")

# COMMAND ----------
def normalize_strings(df):
    """Recorta strings y convierte cadenas vacías en NULL."""
    result = df
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType) and not field.name.startswith("_"):
            result = result.withColumn(
                field.name,
                F.when(F.trim(F.col(field.name)) == "", None).otherwise(
                    F.trim(F.col(field.name))
                ),
            )
    return result


def apply_casts(df, casts):
    """Aplica casts tolerantes para poder contabilizar parseos inválidos."""
    result = df
    for column_name, sql_type in casts.items():
        result = result.withColumn(
            column_name,
            F.expr(f"try_cast(`{column_name}` as {sql_type})"),
        )
    return result


def count_parse_failures(raw_df, typed_df, casts):
    """Cuenta valores fuente no vacíos que no pudieron convertirse."""
    failures = {}
    for column_name in casts:
        raw_non_null = raw_df.filter(
            F.col(column_name).isNotNull() & (F.trim(F.col(column_name)) != "")
        ).count()
        typed_non_null = typed_df.filter(F.col(column_name).isNotNull()).count()
        failures[column_name] = raw_non_null - typed_non_null
    return failures

# COMMAND ----------
summary = []

for table_name, contract in TABLES.items():
    source_table = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"
    target_table = f"{CATALOG}.{SILVER_SCHEMA}.{table_name}"

    if spark.catalog.tableExists(target_table) and not OVERWRITE_EXISTING:
        existing_count = spark.table(target_table).count()
        print(f"SKIP {target_table}: ya existe ({existing_count:,} filas).")
        summary.append(
            {
                "table": target_table,
                "status": "skipped_existing",
                "rows": existing_count,
                "parse_failures": 0,
                "null_required": 0,
                "duplicate_keys": 0,
            }
        )
        continue

    raw_df = spark.table(source_table)
    source_count = raw_df.count()

    silver_df = normalize_strings(raw_df)

    for column_name in contract["lower"]:
        silver_df = silver_df.withColumn(column_name, F.lower(F.col(column_name)))

    for column_name in contract["upper"]:
        silver_df = silver_df.withColumn(column_name, F.upper(F.col(column_name)))

    silver_df = apply_casts(silver_df, contract["casts"])
    silver_df = silver_df.withColumn("_processed_at", F.current_timestamp())

    parse_failures = count_parse_failures(raw_df, silver_df, contract["casts"])
    total_parse_failures = sum(parse_failures.values())

    null_required = 0
    for column_name in contract["required"]:
        null_required += silver_df.filter(F.col(column_name).isNull()).count()

    duplicate_keys = 0
    if contract["unique_key"]:
        duplicate_keys = (
            silver_df
            .groupBy(*contract["unique_key"])
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

    problems = []
    if total_parse_failures:
        details = ", ".join(
            f"{name}={count}"
            for name, count in parse_failures.items()
            if count
        )
        problems.append(f"parseos inválidos: {details}")
    if null_required:
        problems.append(f"campos requeridos nulos: {null_required}")
    if duplicate_keys:
        problems.append(f"claves duplicadas: {duplicate_keys}")

    if problems:
        raise ValueError(
            f"Falló Silver para {table_name}: " + "; ".join(problems)
        )

    writer = silver_df.write.format("delta")
    if OVERWRITE_EXISTING:
        writer = writer.mode("overwrite").option("overwriteSchema", "true")

    writer.saveAsTable(target_table)

    result_count = spark.table(target_table).count()
    if result_count != source_count:
        raise ValueError(
            f"Conteo inconsistente {table_name}: "
            f"Bronze={source_count:,}, Silver={result_count:,}"
        )

    print(f"OK   {target_table}: {result_count:,} filas.")
    summary.append(
        {
            "table": target_table,
            "status": "created",
            "rows": result_count,
            "parse_failures": total_parse_failures,
            "null_required": null_required,
            "duplicate_keys": duplicate_keys,
        }
    )

# COMMAND ----------
summary_df = spark.createDataFrame(summary)
display(summary_df.orderBy("table"))

created = sum(1 for item in summary if item["status"] == "created")
skipped = sum(1 for item in summary if item["status"] == "skipped_existing")

print(f"Tablas Silver creadas: {created}")
print(f"Tablas Silver omitidas por existir: {skipped}")
print(f"Total tablas procesadas: {len(summary)}")
