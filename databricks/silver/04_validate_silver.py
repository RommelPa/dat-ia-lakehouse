# Databricks notebook source
# MAGIC %md
# MAGIC # Silver · validación integral Olist
# MAGIC
# MAGIC Cierra el baseline Silver de las 9 fuentes originales de Olist mediante:
# MAGIC - presencia y conteo de filas;
# MAGIC - conservación 1:1 respecto a Bronze;
# MAGIC - validaciones básicas de claves;
# MAGIC - integridad referencial entre las tablas principales.
# MAGIC
# MAGIC No se exige cobertura total de traducciones de categoría ni de geolocalización,
# MAGIC porque esas relaciones no son claves foráneas estrictas del dataset original.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"

TABLES = [
    "olist_customers",
    "olist_geolocation",
    "olist_order_items",
    "olist_order_payments",
    "olist_order_reviews",
    "olist_orders",
    "olist_products",
    "olist_sellers",
    "product_category_name_translation",
]

# COMMAND ----------
missing_tables = []
for table_name in TABLES:
    full_name = f"{CATALOG}.{SILVER_SCHEMA}.{table_name}"
    if not spark.catalog.tableExists(full_name):
        missing_tables.append(full_name)

if missing_tables:
    raise ValueError(
        "Faltan tablas Silver: " + ", ".join(sorted(missing_tables))
    )

print("Contrato de presencia OK: las 9 tablas Silver existen.")

# COMMAND ----------
row_summary = []
row_problems = []

for table_name in TABLES:
    bronze_table = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"
    silver_table = f"{CATALOG}.{SILVER_SCHEMA}.{table_name}"

    bronze_rows = spark.table(bronze_table).count()
    silver_rows = spark.table(silver_table).count()
    same_rows = bronze_rows == silver_rows

    row_summary.append(
        {
            "table": silver_table,
            "bronze_rows": bronze_rows,
            "silver_rows": silver_rows,
            "same_rows": same_rows,
        }
    )

    if not same_rows:
        row_problems.append(
            f"{table_name}: Bronze={bronze_rows:,}, Silver={silver_rows:,}"
        )

row_summary_df = spark.createDataFrame(row_summary)
display(row_summary_df.orderBy("table"))

if row_problems:
    raise ValueError(
        "Falló conservación Bronze→Silver: " + "; ".join(row_problems)
    )

print("Conservación de filas OK para las 9 tablas.")

# COMMAND ----------
customers = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_customers")
orders = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_orders")
items = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_order_items")
payments = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_order_payments")
reviews = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_order_reviews")
products = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_products")
sellers = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.olist_sellers")

# COMMAND ----------
def orphan_count(child_df, child_column, parent_df, parent_column):
    """Cuenta claves no nulas del hijo sin correspondencia en la tabla padre."""
    child_keys = (
        child_df
        .select(F.col(child_column).alias("fk"))
        .filter(F.col("fk").isNotNull())
        .distinct()
    )
    parent_keys = (
        parent_df
        .select(F.col(parent_column).alias("pk"))
        .filter(F.col("pk").isNotNull())
        .distinct()
    )

    return (
        child_keys
        .join(parent_keys, child_keys.fk == parent_keys.pk, "left_anti")
        .count()
    )


fk_checks = [
    (
        "orders.customer_id → customers.customer_id",
        orphan_count(orders, "customer_id", customers, "customer_id"),
    ),
    (
        "items.order_id → orders.order_id",
        orphan_count(items, "order_id", orders, "order_id"),
    ),
    (
        "items.product_id → products.product_id",
        orphan_count(items, "product_id", products, "product_id"),
    ),
    (
        "items.seller_id → sellers.seller_id",
        orphan_count(items, "seller_id", sellers, "seller_id"),
    ),
    (
        "payments.order_id → orders.order_id",
        orphan_count(payments, "order_id", orders, "order_id"),
    ),
    (
        "reviews.order_id → orders.order_id",
        orphan_count(reviews, "order_id", orders, "order_id"),
    ),
]

fk_summary = [
    {
        "relationship": relationship,
        "orphan_keys": orphan_keys,
        "status": "ok" if orphan_keys == 0 else "failed",
    }
    for relationship, orphan_keys in fk_checks
]

display(spark.createDataFrame(fk_summary).orderBy("relationship"))

fk_problems = [
    f"{relationship}: huérfanas={orphan_keys}"
    for relationship, orphan_keys in fk_checks
    if orphan_keys
]

if fk_problems:
    raise ValueError(
        "Falló integridad referencial Silver: " + "; ".join(fk_problems)
    )

print("Integridad referencial principal OK.")

# COMMAND ----------
# Controles de dominio simples que sí deberían cumplirse en el dataset Olist.
domain_checks = [
    (
        "review_score fuera de 1..5",
        reviews.filter(
            F.col("review_score").isNotNull()
            & ~F.col("review_score").between(1, 5)
        ).count(),
    ),
    (
        "price negativo",
        items.filter(F.col("price") < 0).count(),
    ),
    (
        "freight_value negativo",
        items.filter(F.col("freight_value") < 0).count(),
    ),
    (
        "payment_value negativo",
        payments.filter(F.col("payment_value") < 0).count(),
    ),
]

domain_summary = [
    {
        "check": check_name,
        "violations": violations,
        "status": "ok" if violations == 0 else "failed",
    }
    for check_name, violations in domain_checks
]

display(spark.createDataFrame(domain_summary).orderBy("check"))

domain_problems = [
    f"{check_name}: violaciones={violations}"
    for check_name, violations in domain_checks
    if violations
]

if domain_problems:
    raise ValueError(
        "Fallaron controles de dominio Silver: " + "; ".join(domain_problems)
    )

# COMMAND ----------
print("Baseline Silver validado correctamente.")
print(f"Tablas verificadas: {len(TABLES)}")
print(f"Filas totales Silver: {sum(item['silver_rows'] for item in row_summary):,}")
print(f"Relaciones FK verificadas: {len(fk_checks)}")
print(f"Controles de dominio verificados: {len(domain_checks)}")
