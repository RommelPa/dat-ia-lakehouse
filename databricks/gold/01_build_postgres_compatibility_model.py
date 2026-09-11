# Databricks notebook source
# MAGIC %md
# MAGIC # Gold · modelo de compatibilidad PostgreSQL
# MAGIC
# MAGIC Expone en `dat_ia.gold` las 16 tablas canónicas que conoce Dat-IA,
# MAGIC conservando los mismos nombres y columnas del esquema PostgreSQL actual.
# MAGIC
# MAGIC Esta capa usa **views** sobre Silver para:
# MAGIC - evitar duplicar datos;
# MAGIC - ocultar metadata técnica (`_ingested_at`, `_source_file`, `_processed_at`);
# MAGIC - sustituir items/payments originales por sus versiones extendidas;
# MAGIC - mantener una superficie SQL compatible con el DDL y los prompts existentes.

# COMMAND ----------
CATALOG = "dat_ia"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

MODEL = {
    "olist_customers_dataset": {
        "source": "olist_customers",
        "columns": [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ],
    },
    "olist_geolocation_dataset": {
        "source": "olist_geolocation",
        "columns": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
    },
    "olist_order_items_dataset": {
        "source": "olist_order_items_extended",
        "columns": [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
            "price_event_id",
            "carrier_id",
        ],
    },
    "olist_order_payments_dataset": {
        "source": "olist_order_payments_extended",
        "columns": [
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
            "payment_value",
            "promotion_id",
        ],
    },
    "olist_order_reviews_dataset": {
        "source": "olist_order_reviews",
        "columns": [
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ],
    },
    "olist_orders_dataset": {
        "source": "olist_orders",
        "columns": [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    },
    "olist_products_dataset": {
        "source": "olist_products",
        "columns": [
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    },
    "olist_sellers_dataset": {
        "source": "olist_sellers",
        "columns": [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ],
    },
    "product_category_name_translation": {
        "source": "product_category_name_translation",
        "columns": [
            "product_category_name",
            "product_category_name_english",
        ],
    },
    "carriers": {
        "source": "carriers",
        "columns": [
            "carrier_id",
            "carrier_name",
            "carrier_type",
            "avg_delivery_days",
            "coverage_regions",
            "cost_per_kg",
            "on_time_rate",
        ],
    },
    "product_price_history": {
        "source": "product_price_history",
        "columns": [
            "price_event_id",
            "product_id",
            "seller_id",
            "old_price",
            "new_price",
            "change_at",
            "change_reason",
        ],
    },
    "seller_promotions": {
        "source": "seller_promotions",
        "columns": [
            "promotion_id",
            "seller_id",
            "product_id",
            "discount_pct",
            "start_date",
            "end_date",
            "promo_type",
            "units_sold_during",
        ],
    },
    "warehouse_inventory": {
        "source": "warehouse_inventory",
        "columns": [
            "warehouse_id",
            "product_id",
            "seller_id",
            "stock_qty",
            "reorder_point",
            "last_restocked_date",
        ],
    },
    "delivery_incidents": {
        "source": "delivery_incidents",
        "columns": [
            "incident_id",
            "order_id",
            "incident_type",
            "reported_date",
            "resolved_date",
            "resolution_type",
            "compensation_value",
        ],
    },
    "customer_support_tickets": {
        "source": "customer_support_tickets",
        "columns": [
            "ticket_id",
            "customer_id",
            "order_id",
            "created_at",
            "category",
            "priority",
            "resolution_time_hr",
            "satisfaction_score",
            "resolved",
        ],
    },
    "product_returns": {
        "source": "product_returns",
        "columns": [
            "return_id",
            "order_id",
            "product_id",
            "return_reason",
            "return_date",
            "refund_amount",
            "refund_method",
            "reestocked",
        ],
    },
}

# COMMAND ----------
# Preflight: todas las fuentes Silver deben existir y tener las columnas esperadas.
preflight_problems = []

for gold_name, contract in MODEL.items():
    source_table = f"{CATALOG}.{SILVER_SCHEMA}.{contract['source']}"

    if not spark.catalog.tableExists(source_table):
        preflight_problems.append(f"falta {source_table}")
        continue

    available_columns = set(spark.table(source_table).columns)
    missing_columns = [
        column_name
        for column_name in contract["columns"]
        if column_name not in available_columns
    ]

    if missing_columns:
        preflight_problems.append(
            f"{source_table} sin columnas: {', '.join(missing_columns)}"
        )

if preflight_problems:
    raise ValueError(
        "Falló preflight Gold: " + "; ".join(preflight_problems)
    )

print("Preflight OK: las 16 fuentes Gold son compatibles con el contrato.")

# COMMAND ----------
# Crear o reemplazar las 16 views canónicas.
for gold_name, contract in MODEL.items():
    source_table = f"{CATALOG}.{SILVER_SCHEMA}.{contract['source']}"
    target_view = f"{CATALOG}.{GOLD_SCHEMA}.{gold_name}"
    selected_columns = ",\n    ".join(f"`{name}`" for name in contract["columns"])

    spark.sql(
        f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT
            {selected_columns}
        FROM {source_table}
        """
    )

    print(f"OK   {target_view} ← {source_table}")

# COMMAND ----------
# Validar conteos y contrato exacto de columnas Gold.
summary = []
problems = []

for gold_name, contract in MODEL.items():
    source_table = f"{CATALOG}.{SILVER_SCHEMA}.{contract['source']}"
    target_view = f"{CATALOG}.{GOLD_SCHEMA}.{gold_name}"

    source_rows = spark.table(source_table).count()
    gold_df = spark.table(target_view)
    gold_rows = gold_df.count()
    gold_columns = gold_df.columns

    same_rows = source_rows == gold_rows
    same_columns = gold_columns == contract["columns"]

    summary.append(
        {
            "gold_view": target_view,
            "source_table": source_table,
            "source_rows": source_rows,
            "gold_rows": gold_rows,
            "same_rows": same_rows,
            "same_columns": same_columns,
        }
    )

    if not same_rows:
        problems.append(
            f"{gold_name}: Silver={source_rows:,}, Gold={gold_rows:,}"
        )
    if not same_columns:
        problems.append(
            f"{gold_name}: columnas Gold no coinciden con el contrato"
        )

summary_df = spark.createDataFrame(summary)
display(summary_df.orderBy("gold_view"))

if problems:
    raise ValueError("Falló validación Gold: " + "; ".join(problems))

# COMMAND ----------
print("Modelo Gold de compatibilidad PostgreSQL creado correctamente.")
print(f"Views Gold creadas: {len(MODEL)}")
print(f"Filas lógicas Gold: {sum(item['gold_rows'] for item in summary):,}")
print("Items y payments usan las versiones extendidas canónicas de Dat-IA.")
