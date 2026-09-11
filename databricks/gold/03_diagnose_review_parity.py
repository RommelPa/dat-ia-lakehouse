# Databricks notebook source
# MAGIC %md
# MAGIC # Gold · diagnóstico de paridad de reseñas
# MAGIC
# MAGIC Este notebook investiga la única familia de datos que rompió la paridad
# MAGIC determinista frente al baseline PostgreSQL del golden set: reseñas.
# MAGIC
# MAGIC Es de solo lectura. No corrige ni duplica filas.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"

LAYERS = {
    "bronze": f"{CATALOG}.bronze.olist_order_reviews",
    "silver": f"{CATALOG}.silver.olist_order_reviews",
    "gold": f"{CATALOG}.gold.olist_order_reviews_dataset",
}

POSTGRES_EXPECTED_BY_SCORE = {
    1: 14_891,
    2: 4_091,
    3: 10_607,
    4: 25_030,
    5: 74_756,
}
POSTGRES_EXPECTED_TOTAL = sum(POSTGRES_EXPECTED_BY_SCORE.values())

# COMMAND ----------
# 1) Existencia y conteos por capa.
layer_summary = []

for layer, table_name in LAYERS.items():
    exists = spark.catalog.tableExists(table_name)
    row_count = spark.table(table_name).count() if exists else None
    layer_summary.append(
        {
            "layer": layer,
            "table": table_name,
            "exists": exists,
            "row_count": row_count,
        }
    )

display(spark.createDataFrame(layer_summary).orderBy("layer"))

missing = [item["table"] for item in layer_summary if not item["exists"]]
if missing:
    raise ValueError("Faltan tablas de reseñas: " + ", ".join(missing))

# COMMAND ----------
# 2) Distribución de review_score en Bronze/Silver/Gold.
distribution_rows = []

for layer, table_name in LAYERS.items():
    grouped = (
        spark.table(table_name)
        .groupBy("review_score")
        .count()
        .orderBy("review_score")
        .collect()
    )
    for row in grouped:
        distribution_rows.append(
            {
                "layer": layer,
                "review_score": int(row["review_score"]),
                "rows": int(row["count"]),
            }
        )

display(
    spark.createDataFrame(distribution_rows)
    .orderBy("review_score", "layer")
)

# COMMAND ----------
# 3) Comparación explícita Gold vs baseline PostgreSQL del golden_017.
gold_distribution = {
    int(row["review_score"]): int(row["count"])
    for row in (
        spark.table(LAYERS["gold"])
        .groupBy("review_score")
        .count()
        .collect()
    )
}

parity_rows = []
for score, postgres_count in POSTGRES_EXPECTED_BY_SCORE.items():
    gold_count = gold_distribution.get(score, 0)
    parity_rows.append(
        {
            "review_score": score,
            "postgres_expected": postgres_count,
            "databricks_gold": gold_count,
            "delta": postgres_count - gold_count,
            "postgres_over_gold_ratio": round(postgres_count / gold_count, 6)
            if gold_count
            else None,
        }
    )

display(spark.createDataFrame(parity_rows).orderBy("review_score"))

actual_gold_total = sum(gold_distribution.values())
print(f"PostgreSQL baseline total: {POSTGRES_EXPECTED_TOTAL:,}")
print(f"Databricks Gold total:     {actual_gold_total:,}")
print(f"Delta:                     {POSTGRES_EXPECTED_TOTAL - actual_gold_total:,}")
print(
    "Incremento baseline PostgreSQL vs Gold: "
    f"{100 * (POSTGRES_EXPECTED_TOTAL / actual_gold_total - 1):.2f}%"
)

# COMMAND ----------
# 4) Unicidad y duplicación lógica en cada capa.
# El DDL PostgreSQL usa un id sustituto porque review_id no es necesariamente único.
identity_summary = []

for layer, table_name in LAYERS.items():
    df = spark.table(table_name)
    total = df.count()
    distinct_orders = df.select("order_id").distinct().count()
    distinct_reviews = df.select("review_id").distinct().count()

    duplicated_order_groups = (
        df.groupBy("order_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    duplicated_review_groups = (
        df.groupBy("review_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    identity_summary.append(
        {
            "layer": layer,
            "rows": total,
            "distinct_order_id": distinct_orders,
            "extra_rows_vs_order_id": total - distinct_orders,
            "order_ids_with_multiple_rows": duplicated_order_groups,
            "distinct_review_id": distinct_reviews,
            "extra_rows_vs_review_id": total - distinct_reviews,
            "review_ids_with_multiple_rows": duplicated_review_groups,
        }
    )

display(spark.createDataFrame(identity_summary).orderBy("layer"))

# COMMAND ----------
# 5) Duplicados exactos en la superficie Gold.
gold = spark.table(LAYERS["gold"])
exact_columns = gold.columns

exact_duplicate_groups = (
    gold.groupBy(*exact_columns)
    .count()
    .filter(F.col("count") > 1)
)

exact_duplicate_group_count = exact_duplicate_groups.count()
exact_duplicate_extra_rows = (
    exact_duplicate_groups
    .select(F.sum(F.col("count") - F.lit(1)).alias("extra"))
    .first()["extra"]
    or 0
)

print(f"Grupos duplicados exactos Gold: {exact_duplicate_group_count:,}")
print(f"Filas extra por duplicado exacto: {int(exact_duplicate_extra_rows):,}")

# COMMAND ----------
# 6) Reproducir golden_022 con conteo auxiliar para entender el impacto.
category_stats = spark.sql(
    f"""
    WITH order_categories AS (
        SELECT DISTINCT
            i.order_id,
            p.product_category_name AS category
        FROM {CATALOG}.gold.olist_order_items_dataset i
        JOIN {CATALOG}.gold.olist_products_dataset p
          ON p.product_id = i.product_id
        WHERE p.product_category_name IS NOT NULL
    )
    SELECT
        oc.category,
        COUNT(*) AS reviews,
        ROUND(AVG(r.review_score), 2) AS avg_review_score
    FROM order_categories oc
    JOIN {CATALOG}.gold.olist_order_reviews_dataset r
      ON r.order_id = oc.order_id
    GROUP BY oc.category
    HAVING COUNT(*) >= 100
    ORDER BY avg_review_score DESC, reviews DESC, oc.category
    """
)

display(category_stats.limit(15))

# COMMAND ----------
# 7) Comparar específicamente las categorías esperadas por PostgreSQL en golden_022.
POSTGRES_GOLDEN_022 = {
    "livros_interesse_geral": 4.48,
    "construcao_ferramentas_ferramentas": 4.44,
    "livros_tecnicos": 4.41,
    "alimentos_bebidas": 4.38,
    "malas_acessorios": 4.34,
}

expected_categories = list(POSTGRES_GOLDEN_022)
comparison_022 = (
    category_stats
    .filter(F.col("category").isin(*expected_categories))
    .select("category", "reviews", "avg_review_score")
    .collect()
)

comparison_rows = []
current_by_category = {row["category"]: row for row in comparison_022}
for category, postgres_avg in POSTGRES_GOLDEN_022.items():
    current = current_by_category.get(category)
    comparison_rows.append(
        {
            "category": category,
            "postgres_expected_avg": postgres_avg,
            "databricks_reviews": int(current["reviews"]) if current else None,
            "databricks_avg": float(current["avg_review_score"]) if current else None,
        }
    )

display(spark.createDataFrame(comparison_rows).orderBy("category"))

# COMMAND ----------
# 8) Diagnóstico final de la capa donde aparece el drift.
counts = {item["layer"]: item["row_count"] for item in layer_summary}

if counts["bronze"] == counts["silver"] == counts["gold"]:
    print(
        "DIAGNÓSTICO: Bronze, Silver y Gold conservan el mismo número de reseñas. "
        "La diferencia frente al baseline PostgreSQL ya existe en el dataset "
        "ingestado en Bronze; no fue introducida por las transformaciones Silver/Gold."
    )
else:
    print(
        "DIAGNÓSTICO: los conteos cambian entre Bronze/Silver/Gold. "
        "Revisar la primera transición donde aparezca la diferencia."
    )

print(
    "No modificar el golden set ni duplicar filas hasta identificar la procedencia "
    "del dataset PostgreSQL de 129,375 reseñas."
)
