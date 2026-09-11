-- Dat-IA Lakehouse bootstrap
-- Ejecutar una sola vez en Databricks Free Edition.
-- Crea el namespace del proyecto y un volumen administrado para los CSV Olist.

CREATE CATALOG IF NOT EXISTS dat_ia;

CREATE SCHEMA IF NOT EXISTS dat_ia.landing
COMMENT 'Zona de archivos fuente cargados manualmente o por ingesta.';

CREATE SCHEMA IF NOT EXISTS dat_ia.bronze
COMMENT 'Tablas raw Delta con cambios mínimos sobre la fuente.';

CREATE SCHEMA IF NOT EXISTS dat_ia.silver
COMMENT 'Tablas limpias, tipadas y con reglas de calidad.';

CREATE SCHEMA IF NOT EXISTS dat_ia.gold
COMMENT 'Capa semántica y marts consumidos por analytics y Dat-IA.';

CREATE SCHEMA IF NOT EXISTS dat_ia.ml
COMMENT 'Experimentos, features y artefactos relacionados con ML/MLOps.';

CREATE VOLUME IF NOT EXISTS dat_ia.landing.olist_raw
COMMENT 'Archivos CSV originales del dataset Olist y extensiones sintéticas.';

SHOW SCHEMAS IN dat_ia;
SHOW VOLUMES IN dat_ia.landing;
