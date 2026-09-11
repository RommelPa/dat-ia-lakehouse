# Databricks notebook source
# MAGIC %md
# MAGIC # Gold · validación formal PostgreSQL compatibility
# MAGIC
# MAGIC Valida el baseline de 16 views de `dat_ia.gold` antes del benchmark
# MAGIC PostgreSQL vs Databricks. Es de solo lectura.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
GOLD = "gold"
SILVER = "silver"
EXPECTED_TOTAL = 1_620_981

CONTRACT = {
    "olist_customers_dataset": (99_441, ["customer_id","customer_unique_id","customer_zip_code_prefix","customer_city","customer_state"], ["customer_id","customer_unique_id"], ["customer_id"]),
    "olist_geolocation_dataset": (1_000_163, ["geolocation_zip_code_prefix","geolocation_lat","geolocation_lng","geolocation_city","geolocation_state"], ["geolocation_zip_code_prefix","geolocation_lat","geolocation_lng"], None),
    "olist_order_items_dataset": (112_650, ["order_id","order_item_id","product_id","seller_id","shipping_limit_date","price","freight_value","price_event_id","carrier_id"], ["order_id","order_item_id","product_id","seller_id"], ["order_id","order_item_id"]),
    "olist_order_payments_dataset": (103_886, ["order_id","payment_sequential","payment_type","payment_installments","payment_value","promotion_id"], ["order_id","payment_sequential","payment_type"], ["order_id","payment_sequential"]),
    "olist_order_reviews_dataset": (99_224, ["review_id","order_id","review_score","review_comment_title","review_comment_message","review_creation_date","review_answer_timestamp"], ["review_id","order_id","review_score"], None),
    "olist_orders_dataset": (99_441, ["order_id","customer_id","order_status","order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"], ["order_id","customer_id"], ["order_id"]),
    "olist_products_dataset": (32_951, ["product_id","product_category_name","product_name_lenght","product_description_lenght","product_photos_qty","product_weight_g","product_length_cm","product_height_cm","product_width_cm"], ["product_id"], ["product_id"]),
    "olist_sellers_dataset": (3_095, ["seller_id","seller_zip_code_prefix","seller_city","seller_state"], ["seller_id"], ["seller_id"]),
    "product_category_name_translation": (71, ["product_category_name","product_category_name_english"], ["product_category_name","product_category_name_english"], ["product_category_name"]),
    "carriers": (25, ["carrier_id","carrier_name","carrier_type","avg_delivery_days","coverage_regions","cost_per_kg","on_time_rate"], ["carrier_id","carrier_name"], ["carrier_id"]),
    "product_price_history": (8_832, ["price_event_id","product_id","seller_id","old_price","new_price","change_at","change_reason"], ["price_event_id","product_id","seller_id"], ["price_event_id"]),
    "seller_promotions": (5_000, ["promotion_id","seller_id","product_id","discount_pct","start_date","end_date","promo_type","units_sold_during"], ["promotion_id","seller_id","product_id"], ["promotion_id"]),
    "warehouse_inventory": (26_202, ["warehouse_id","product_id","seller_id","stock_qty","reorder_point","last_restocked_date"], ["warehouse_id","product_id","seller_id"], ["warehouse_id","product_id","seller_id"]),
    "delivery_incidents": (10_000, ["incident_id","order_id","incident_type","reported_date","resolved_date","resolution_type","compensation_value"], ["incident_id","order_id"], ["incident_id"]),
    "customer_support_tickets": (13_000, ["ticket_id","customer_id","order_id","created_at","category","priority","resolution_time_hr","satisfaction_score","resolved"], ["ticket_id","customer_id","order_id"], ["ticket_id"]),
    "product_returns": (7_000, ["return_id","order_id","product_id","return_reason","return_date","refund_amount","refund_method","reestocked"], ["return_id","order_id","product_id"], ["return_id"]),
}

# COMMAND ----------
def full(name):
    return f"{CATALOG}.{GOLD}.{name}"


def orphan_count(child, child_col, parent, parent_col):
    c = child.select(F.col(child_col).alias("fk")).where(F.col("fk").isNotNull()).distinct()
    p = parent.select(F.col(parent_col).alias("pk")).where(F.col("pk").isNotNull()).distinct()
    return c.join(p, c.fk == p.pk, "left_anti").count()


def duplicate_groups(df, key):
    if not key:
        return 0
    return df.groupBy(*key).count().where(F.col("count") > 1).count()

# COMMAND ----------
summary = []
problems = []

tables = {}
for name, (expected_rows, columns, required, key) in CONTRACT.items():
    table_name = full(name)
    if not spark.catalog.tableExists(table_name):
        problems.append(f"falta {table_name}")
        continue

    df = spark.table(table_name)
    tables[name] = df
    rows = df.count()
    null_required = sum(
        df.where(F.col(column).isNull()).count()
        for column in required
    )
    duplicate_keys = duplicate_groups(df, key)
    columns_ok = df.columns == columns

    summary.append({
        "table": table_name,
        "expected_rows": expected_rows,
        "actual_rows": rows,
        "columns_ok": columns_ok,
        "null_required": null_required,
        "duplicate_keys": duplicate_keys,
    })

    if rows != expected_rows:
        problems.append(f"{name}: filas esperadas={expected_rows:,}, reales={rows:,}")
    if not columns_ok:
        problems.append(f"{name}: columnas u orden incorrectos")
    if null_required:
        problems.append(f"{name}: requeridos nulos={null_required}")
    if duplicate_keys:
        problems.append(f"{name}: claves duplicadas={duplicate_keys}")

display(spark.createDataFrame(summary).orderBy("table"))

if problems:
    raise ValueError("Falló contrato Gold: " + "; ".join(problems))

total_rows = sum(item["actual_rows"] for item in summary)
if total_rows != EXPECTED_TOTAL:
    raise ValueError(f"Total Gold inesperado: {total_rows:,} != {EXPECTED_TOTAL:,}")

print("Contrato estructural Gold OK.")

# COMMAND ----------
FK = [
    ("orders.customer", "olist_orders_dataset","customer_id","olist_customers_dataset","customer_id"),
    ("items.order", "olist_order_items_dataset","order_id","olist_orders_dataset","order_id"),
    ("items.product", "olist_order_items_dataset","product_id","olist_products_dataset","product_id"),
    ("items.seller", "olist_order_items_dataset","seller_id","olist_sellers_dataset","seller_id"),
    ("payments.order", "olist_order_payments_dataset","order_id","olist_orders_dataset","order_id"),
    ("reviews.order", "olist_order_reviews_dataset","order_id","olist_orders_dataset","order_id"),
    ("price.product", "product_price_history","product_id","olist_products_dataset","product_id"),
    ("price.seller", "product_price_history","seller_id","olist_sellers_dataset","seller_id"),
    ("promo.product", "seller_promotions","product_id","olist_products_dataset","product_id"),
    ("promo.seller", "seller_promotions","seller_id","olist_sellers_dataset","seller_id"),
    ("inventory.product", "warehouse_inventory","product_id","olist_products_dataset","product_id"),
    ("inventory.seller", "warehouse_inventory","seller_id","olist_sellers_dataset","seller_id"),
    ("incidents.order", "delivery_incidents","order_id","olist_orders_dataset","order_id"),
    ("tickets.customer", "customer_support_tickets","customer_id","olist_customers_dataset","customer_id"),
    ("tickets.order", "customer_support_tickets","order_id","olist_orders_dataset","order_id"),
    ("returns.order", "product_returns","order_id","olist_orders_dataset","order_id"),
    ("returns.product", "product_returns","product_id","olist_products_dataset","product_id"),
    ("items.carrier", "olist_order_items_dataset","carrier_id","carriers","carrier_id"),
    ("items.price_event", "olist_order_items_dataset","price_event_id","product_price_history","price_event_id"),
    ("payments.promotion", "olist_order_payments_dataset","promotion_id","seller_promotions","promotion_id"),
]

fk_summary = []
for label, child, child_col, parent, parent_col in FK:
    orphans = orphan_count(tables[child], child_col, tables[parent], parent_col)
    fk_summary.append({"relationship": label, "orphan_keys": orphans})
    if orphans:
        problems.append(f"{label}: huérfanas={orphans}")

display(spark.createDataFrame(fk_summary).orderBy("relationship"))

if problems:
    raise ValueError("Falló integridad referencial Gold: " + "; ".join(problems))

print(f"Integridad referencial Gold OK: {len(FK)} relaciones.")

# COMMAND ----------
DOMAIN_CHECKS = [
    ("review_score", tables["olist_order_reviews_dataset"].where(~F.col("review_score").between(1,5) & F.col("review_score").isNotNull()).count()),
    ("price negativo", tables["olist_order_items_dataset"].where(F.col("price") < 0).count()),
    ("freight negativo", tables["olist_order_items_dataset"].where(F.col("freight_value") < 0).count()),
    ("payment negativo", tables["olist_order_payments_dataset"].where(F.col("payment_value") < 0).count()),
    ("on_time_rate", tables["carriers"].where(~F.col("on_time_rate").between(0,1) & F.col("on_time_rate").isNotNull()).count()),
    ("discount_pct", tables["seller_promotions"].where(~F.col("discount_pct").between(0,1) & F.col("discount_pct").isNotNull()).count()),
    ("promotion dates", tables["seller_promotions"].where(F.col("start_date") > F.col("end_date")).count()),
    ("satisfaction", tables["customer_support_tickets"].where(~F.col("satisfaction_score").between(1,5) & F.col("satisfaction_score").isNotNull()).count()),
    ("refund negativo", tables["product_returns"].where(F.col("refund_amount") < 0).count()),
]

for label, violations in DOMAIN_CHECKS:
    if violations:
        problems.append(f"{label}: violaciones={violations}")

if problems:
    raise ValueError("Fallaron dominios Gold: " + "; ".join(problems))

print(f"Dominios Gold OK: {len(DOMAIN_CHECKS)} controles.")

# COMMAND ----------
COVERAGE = [
    ("carrier_id", "olist_order_items_dataset", "olist_order_items_extended"),
    ("price_event_id", "olist_order_items_dataset", "olist_order_items_extended"),
    ("promotion_id", "olist_order_payments_dataset", "olist_order_payments_extended"),
]

coverage_summary = []
for column, gold_name, silver_name in COVERAGE:
    silver = spark.table(f"{CATALOG}.{SILVER}.{silver_name}")
    gold = tables[gold_name]
    gold_non_null = gold.where(F.col(column).isNotNull()).count()
    silver_non_null = silver.where(F.col(column).isNotNull()).count()
    coverage_summary.append({
        "column": column,
        "gold_non_null": gold_non_null,
        "silver_non_null": silver_non_null,
    })
    if gold.count() != silver.count() or gold_non_null != silver_non_null:
        problems.append(f"{column}: cobertura Gold != Silver")

display(spark.createDataFrame(coverage_summary).orderBy("column"))

if problems:
    raise ValueError("Falló cobertura extendida: " + "; ".join(problems))

# COMMAND ----------
print("Validación formal Gold completada correctamente.")
print(f"Views verificadas: {len(CONTRACT)}")
print(f"Filas lógicas Gold: {total_rows:,}")
print(f"Relaciones FK verificadas: {len(FK)}")
print(f"Controles de dominio verificados: {len(DOMAIN_CHECKS)}")
print(f"FKs extendidas comparadas: {len(COVERAGE)}")
print("Baseline listo para benchmark PostgreSQL vs Databricks compatibility.")
