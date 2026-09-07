-- Dat-IA extended landing zone
-- Crea un Volume separado para los 9 CSV canónicos generados localmente
-- por scripts/generar_tablas_sinteticas.py (SEED=42).

CREATE VOLUME IF NOT EXISTS dat_ia.landing.dat_ia_extended_raw
COMMENT 'CSV canónicos generados por Dat-IA: 7 tablas sintéticas + 2 tablas Olist extendidas.';

SHOW VOLUMES IN dat_ia.landing;
