# Databricks notebook source
# MAGIC %md
# MAGIC # Silver · Dat-IA Extended
# MAGIC
# MAGIC Tipado y validación de las 9 tablas canónicas generadas por
# MAGIC `scripts/generar_tablas_sinteticas.py` (SEED=42).
# MAGIC
# MAGIC Las versiones extendidas de items y payments conservan el sufijo
# MAGIC `_extended` en Silver. Gold decidirá después qué tablas exponer con los
# MAGIC nombres canónicos del modelo Dat-IA de 16 tablas.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

CATALOG = "dat_ia"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
OVERWRITE_EXISTING = False

TABLES = {
    "carriers": {
        "expected_rows": 25,
        "casts": {
            "avg_delivery_days": "decimal(5,1)",
            "cost_per_kg": "decimal(8,2)",
            "on_time_rate": "decimal(5,3)",
        },
        "required": ["carrier_id", "carrier_name"],
        "unique_key": ["carrier_id"],
        "lower": ["carrier_type"],
        "upper": [],
    },
    "product_price_history": {
        "expected_rows": 8_832,
        "casts": {
            "old_price": "decimal(12,2)",
            "new_price": "decimal(12,2)",
            "change_at": "timestamp",
        },
        "required": ["price_event_id", "product_id", "seller_id"],
        "unique_key": ["price_event_id"],
        "lower": ["change_reason"],
        "upper": [],
    },
    "seller_promotions": {
        "expected_rows": 5_000,
        "casts": {
            "discount_pct": "decimal(4,2)",
            "start_date": "date",
            "end_date": "date",
            "units_sold_during": "int",
        },
        "required": ["promotion_id", "seller_id", "product_id"],
        "unique_key": ["promotion_id"],
        "lower": ["promo_type"],
        "upper": [],
    },
    "warehouse_inventory": {
        "expected_rows": 26_202,
        "casts": {
            "stock_qty": "int",
            "reorder_point": "int",
            "last_restocked_date": "date",
        },
        "required": ["warehouse_id", "product_id", "seller_id"],
        "unique_key": ["warehouse_id", "product_id", "seller_id"],
        "lower": [],
        "upper": [],
    },
    "delivery_incidents": {
        "expected_rows": 10_000,
        "casts": {
            "reported_date": "timestamp",
            "resolved_date": "timestamp",
            "compensation_value": "decimal(12,2)",
        },
        "required": ["incident_id", "order_id"],
        "unique_key": ["incident_id"],
        "lower": ["incident_type", "resolution_type"],
        "upper": [],
    },
    "customer_support_tickets": {
        "expected_rows": 13_000,
        "casts": {
            "created_at": "timestamp",
            "resolution_time_hr": "decimal(8,1)",
            "satisfaction_score": "int",
            "resolved": "boolean",
        },
        "required": ["ticket_id", "customer_id", "order_id"],
        "unique_key": ["ticket_id"],
        "lower": ["category", "priority"],
        "upper": [],
    },
    "product_returns": {
        "expected_rows": 7_000,
        "casts": {
            "return_date": "timestamp",
            "refund_amount": "decimal(12,2)",
            "reestocked": "boolean",
        },
        "required": ["return_id", "order_id", "product_id"],
        "unique_key": ["return_id"],
        "lower": ["return_reason", "refund_method"],
        "upper": [],
    },
    "olist_order_items_extended": {
        "expected_rows": 112_650,
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
    "olist_order_payments_extended": {
        "expected_rows": 103_886,
        "casts": {
            "payment_sequential": "int",
            "payment_installments": "int",
            "payment_value": "decimal(12,2)",
        },
        "required": ["order_id", "payment_sequential"],
        "unique_key": ["order_id", "payment_sequential"],
        "lower": ["payment_type"],
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
        "Faltan tablas Bronze Dat-IA Extended: "
        + ", ".join(sorted(missing_bronze))
    )

print("Preflight OK: las 9 tablas Bronze Dat-IA Extended existen.")

# COMMAND ----------
def normalize_strings(df):
    """Recorta strings de negocio y convierte vacíos en NULL."""
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
    """Aplica casts tolerantes para poder detectar parseos fallidos."""
    result = df
    for column_name, sql_type in casts.items():
        result = result.withColumn(
            column_name,
            F.expr(f"try_cast(`{column_name}` as {sql_type})"),
        )
    return result


def count_parse_failures(raw_df, typed_df, casts):
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
        if existing_count != contract["expected_rows"]:
            raise ValueError(
                f"{target_table} existe con {existing_count:,} filas; "
                f"se esperaban {contract['expected_rows']:,}."
            )
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

    if source_count != contract["expected_rows"]:
        raise ValueError(
            f"Conteo Bronze inesperado para {table_name}: "
            f"esperado={contract['expected_rows']:,}, real={source_count:,}"
        )

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
            f"Falló Silver Dat-IA Extended para {table_name}: "
            + "; ".join(problems)
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

print(f"Tablas Silver Extended creadas: {created}")
print(f"Tablas Silver Extended omitidas por existir: {skipped}")
print(f"Total tablas procesadas: {len(summary)}")
print(f"Filas Silver Extended: {sum(item['rows'] for item in summary):,}")
