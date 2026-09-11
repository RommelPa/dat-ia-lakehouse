# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze · fuentes Olist
# MAGIC
# MAGIC Ingesta parametrizada de las 9 fuentes originales de Olist desde el
# MAGIC Volume `dat_ia.landing.olist_raw` hacia tablas Delta en `dat_ia.bronze`.
# MAGIC
# MAGIC Principios:
# MAGIC - Bronze conserva los campos de origen como `string`.
# MAGIC - Se valida que los 9 CSV existan antes de escribir cualquier tabla.
# MAGIC - Las tablas existentes se omiten por defecto para evitar sobrescrituras.
# MAGIC - La sobrescritura requiere cambiar explícitamente `OVERWRITE_EXISTING`.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
LANDING_SCHEMA = "landing"
BRONZE_SCHEMA = "bronze"
VOLUME = "olist_raw"
VOLUME_PATH = f"/Volumes/{CATALOG}/{LANDING_SCHEMA}/{VOLUME}"

OVERWRITE_EXISTING = False

SOURCES = [
    {
        "table": "olist_customers",
        "file": "olist_customers_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_geolocation",
        "file": "olist_geolocation_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_order_items",
        "file": "olist_order_items_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_order_payments",
        "file": "olist_order_payments_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_order_reviews",
        "file": "olist_order_reviews_dataset.csv",
        "multi_line": True,
    },
    {
        "table": "olist_orders",
        "file": "olist_orders_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_products",
        "file": "olist_products_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "olist_sellers",
        "file": "olist_sellers_dataset.csv",
        "multi_line": False,
    },
    {
        "table": "product_category_name_translation",
        "file": "product_category_name_translation.csv",
        "multi_line": False,
    },
]

# COMMAND ----------
# Preflight: no se inicia ninguna escritura si falta una fuente esperada.
volume_entries = dbutils.fs.ls(VOLUME_PATH)
available_files = {
    entry.name.rstrip("/")
    for entry in volume_entries
    if not entry.path.endswith("/")
}

expected_files = {source["file"] for source in SOURCES}
missing_files = sorted(expected_files - available_files)

print(f"Volume: {VOLUME_PATH}")
print(f"Archivos detectados: {len(available_files)}")

if missing_files:
    raise ValueError(
        "Faltan archivos Olist en el Volume: " + ", ".join(missing_files)
    )

print("Preflight OK: las 9 fuentes Olist están disponibles.")

# COMMAND ----------
def read_source(source: dict):
    """Lee una fuente CSV Olist conservando sus columnas como string."""
    source_path = f"{VOLUME_PATH}/{source['file']}"

    return (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("multiLine", source["multi_line"])
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .option("mode", "FAILFAST")
        .csv(source_path)
    )


def to_bronze(raw_df):
    """Agrega solo metadata técnica; no aplica reglas de negocio."""
    return (
        raw_df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )

# COMMAND ----------
summary = []

for source in SOURCES:
    target_table = f"{CATALOG}.{BRONZE_SCHEMA}.{source['table']}"

    if spark.catalog.tableExists(target_table) and not OVERWRITE_EXISTING:
        existing_count = spark.table(target_table).count()
        print(f"SKIP {target_table}: ya existe ({existing_count:,} filas).")
        summary.append(
            {
                "table": target_table,
                "status": "skipped_existing",
                "rows": existing_count,
                "source_file": source["file"],
            }
        )
        continue

    raw_df = read_source(source)
    bronze_df = to_bronze(raw_df)
    row_count = bronze_df.count()

    if row_count == 0:
        raise ValueError(f"La fuente {source['file']} no contiene filas.")

    writer = bronze_df.write.format("delta")

    if OVERWRITE_EXISTING:
        writer = writer.mode("overwrite").option("overwriteSchema", "true")

    # Para tablas nuevas usamos el comportamiento por defecto de saveAsTable.
    # En Databricks Serverless/Spark Connect el modo explícito "errorifexists"
    # no está soportado, por eso evitamos configurarlo aquí.
    writer.saveAsTable(target_table)

    print(f"OK   {target_table}: {row_count:,} filas.")
    summary.append(
        {
            "table": target_table,
            "status": "created",
            "rows": row_count,
            "source_file": source["file"],
        }
    )

# COMMAND ----------
summary_df = spark.createDataFrame(summary)

display(summary_df.orderBy("table"))

created = sum(1 for item in summary if item["status"] == "created")
skipped = sum(1 for item in summary if item["status"] == "skipped_existing")

print(f"Tablas creadas: {created}")
print(f"Tablas omitidas por existir: {skipped}")
print(f"Total fuentes procesadas: {len(summary)}")
