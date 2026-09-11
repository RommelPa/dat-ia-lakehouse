# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze · Dat-IA Extended
# MAGIC
# MAGIC Ingesta de los 9 CSV canónicos generados por
# MAGIC `scripts/generar_tablas_sinteticas.py` con `SEED=42`.
# MAGIC
# MAGIC Se cargan desde `dat_ia.landing.dat_ia_extended_raw` y se persisten como
# MAGIC tablas Delta separadas en `dat_ia.bronze`. Los conteos esperados forman
# MAGIC parte del contrato para garantizar que Databricks recibe exactamente el
# MAGIC mismo dataset derivado que usa el proyecto Dat-IA actual.

# COMMAND ----------
from pyspark.sql import functions as F

CATALOG = "dat_ia"
LANDING_SCHEMA = "landing"
BRONZE_SCHEMA = "bronze"
VOLUME = "dat_ia_extended_raw"
VOLUME_PATH = f"/Volumes/{CATALOG}/{LANDING_SCHEMA}/{VOLUME}"
OVERWRITE_EXISTING = False

SOURCES = [
    {"table": "carriers", "file": "carriers.csv", "expected_rows": 25},
    {
        "table": "product_price_history",
        "file": "product_price_history.csv",
        "expected_rows": 8_832,
    },
    {
        "table": "seller_promotions",
        "file": "seller_promotions.csv",
        "expected_rows": 5_000,
    },
    {
        "table": "warehouse_inventory",
        "file": "warehouse_inventory.csv",
        "expected_rows": 26_202,
    },
    {
        "table": "delivery_incidents",
        "file": "delivery_incidents.csv",
        "expected_rows": 10_000,
    },
    {
        "table": "customer_support_tickets",
        "file": "customer_support_tickets.csv",
        "expected_rows": 13_000,
    },
    {
        "table": "product_returns",
        "file": "product_returns.csv",
        "expected_rows": 7_000,
    },
    {
        "table": "olist_order_items_extended",
        "file": "olist_order_items_extended.csv",
        "expected_rows": 112_650,
    },
    {
        "table": "olist_order_payments_extended",
        "file": "olist_order_payments_extended.csv",
        "expected_rows": 103_886,
    },
]

# COMMAND ----------
# Preflight: los 9 CSV deben estar presentes antes de iniciar escrituras.
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
        "Faltan archivos Dat-IA Extended en el Volume: "
        + ", ".join(missing_files)
    )

print("Preflight OK: los 9 CSV Dat-IA Extended están disponibles.")

# COMMAND ----------
def read_source(source):
    source_path = f"{VOLUME_PATH}/{source['file']}"
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("multiLine", False)
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .option("mode", "FAILFAST")
        .csv(source_path)
    )


def to_bronze(raw_df):
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
        if existing_count != source["expected_rows"]:
            raise ValueError(
                f"{target_table} ya existe pero tiene {existing_count:,} filas; "
                f"se esperaban {source['expected_rows']:,}."
            )

        print(f"SKIP {target_table}: ya existe ({existing_count:,} filas).")
        summary.append(
            {
                "table": target_table,
                "status": "skipped_existing",
                "rows": existing_count,
                "expected_rows": source["expected_rows"],
                "source_file": source["file"],
            }
        )
        continue

    raw_df = read_source(source)
    bronze_df = to_bronze(raw_df)
    row_count = bronze_df.count()

    if row_count != source["expected_rows"]:
        raise ValueError(
            f"Conteo inesperado en {source['file']}: "
            f"esperado={source['expected_rows']:,}, real={row_count:,}"
        )

    writer = bronze_df.write.format("delta")
    if OVERWRITE_EXISTING:
        writer = writer.mode("overwrite").option("overwriteSchema", "true")

    writer.saveAsTable(target_table)

    print(f"OK   {target_table}: {row_count:,} filas.")
    summary.append(
        {
            "table": target_table,
            "status": "created",
            "rows": row_count,
            "expected_rows": source["expected_rows"],
            "source_file": source["file"],
        }
    )

# COMMAND ----------
summary_df = spark.createDataFrame(summary)
display(summary_df.orderBy("table"))

created = sum(1 for item in summary if item["status"] == "created")
skipped = sum(1 for item in summary if item["status"] == "skipped_existing")

print(f"Tablas Extended creadas: {created}")
print(f"Tablas Extended omitidas por existir: {skipped}")
print(f"Total fuentes procesadas: {len(summary)}")
print(f"Filas canónicas esperadas: {sum(item['expected_rows'] for item in summary):,}")
