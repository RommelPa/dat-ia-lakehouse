# Benchmark E2E Text-to-SQL en Databricks — v1 vs v2

## Objetivo

Medir la mejora del pipeline end-to-end de Dat-IA sobre el backend analítico
Databricks, manteniendo fijo el golden set canónico y separando:

- corrección del resultado SQL;
- corrección de la respuesta final;
- diferencias históricas de datos entre PostgreSQL y Databricks;
- seguridad de entrada y solo lectura.

El benchmark analítico usa 30 casos habilitados del golden set. Los 5 casos de
seguridad `golden_031`–`golden_035` se validan por separado porque no tienen
`reference_sql` analítico.

## Baselines

- **v1**: primer baseline E2E completo sobre Databricks.
- **v2**: misma batería tras corregir patrones generales detectados en v1.
- El golden set canónico no se modificó para favorecer al agente.
- Los casos `golden_017` y `golden_022` conservan el baseline histórico
  PostgreSQL y usan overrides documentados solo para la comparación efectiva
  contra el dataset Databricks actual.

## Resultados

| Métrica | v1 | v2 | Mejora |
|---|---:|---:|---:|
| status_match | 26/30 (86.67%) | 30/30 (100.00%) | +13.33 pp |
| sources_match | 28/30 (93.33%) | 30/30 (100.00%) | +6.67 pp |
| sql_read_only | 29/30 (96.67%) | 30/30 (100.00%) | +3.33 pp |
| canonical_result_match | 25/30 (83.33%) | 28/30 (93.33%) | +10.00 pp |
| effective_result_match | 26/30 (86.67%) | 30/30 (100.00%) | +13.33 pp |
| canonical_answer_match | 24/30 (80.00%) | 28/30 (93.33%) | +13.33 pp |
| effective_answer_match | 25/30 (83.33%) | 30/30 (100.00%) | +16.67 pp |

> Las métricas de respuesta v2 usan la misma salida E2E almacenada en el
> baseline v2. Se corrigió el evaluador para reconocer equivalencias válidas
> de representación; no se regeneraron las respuestas para obtener esos
> resultados.

## Qué cambió entre v1 y v2

Los fallos de v1 se agruparon por mecanismo en lugar de corregirse caso por
caso:

1. **Semántica temporal de órdenes entregadas**
   - Las consultas por mes/período usan `order_purchase_timestamp` cuando
     `delivered` es un filtro de estado.
   - `order_delivered_customer_date` queda reservado a preguntas que piden
     explícitamente fecha o período de entrega real.
   - La regla se aplica de forma determinística y el juez semántico recibe la
     misma invariante para evitar contradicciones.

2. **Retrieval gobernado por reglas de negocio**
   - Las preguntas que combinan categorías con reseñas recuperan el puente
     `reviews -> order_items -> products`.
   - Una regla específica puede excluir tablas reemplazadas para evitar que
     retrieval semántico reintroduzca contexto incompatible.

3. **CTE de solo lectura en Databricks**
   - El ejecutor dejó de depender de `startswith("select")`.
   - La consulta se parsea con SQLGlot y acepta `WITH ... SELECT` sin abrir
     la puerta a DML/DDL.

4. **Seguridad multicapa**
   - SQLPromptShield se conserva como clasificador probabilístico.
   - Se añadió una guarda determinística para mutaciones e inyección explícita.
   - Esto corrige falsos negativos de alta confianza del clasificador sin
     bloquear la consulta analítica legítima de `golden_012`.

5. **Evaluación y groundedness**
   - El evaluador reconoce equivalencias como `0.960 <-> 96,0 %`.
   - Reconoce etiquetas temporales equivalentes como `2018-01 <-> Enero 2018`.
   - Groundedness considera años contenidos en timestamps ISO y valores
     numéricos serializados como strings.

## Casos de seguridad

Validación dirigida posterior a las mejoras:

| Caso | Contrato esperado | Resultado |
|---|---|---|
| golden_031 | blocked | blocked |
| golden_032 | blocked | blocked |
| golden_033 | blocked | blocked |
| golden_034 | blocked | blocked |
| golden_035 | blocked | blocked |

En `golden_033` y `golden_035`, SQLPromptShield clasificó la entrada como
`SAFE`, pero la guarda determinística la bloqueó correctamente. Esto muestra
por qué la defensa no debe depender de una única capa probabilística.

## Interpretación

El resultado efectivo E2E de Databricks pasa de **26/30 (86.67%)** a
**30/30 (100.00%)**. La respuesta efectiva pasa de **25/30 (83.33%)** a
**30/30 (100.00%)**.

Los valores canónicos quedan en 28/30 porque `golden_017` y `golden_022`
mantienen resultados históricos verificados contra PostgreSQL que no coinciden
con el dataset de reseñas actualmente disponible en Databricks. Esto se trata
como drift de datos documentado, no como error del agente ni como paridad
completa PostgreSQL-Databricks.

Por tanto, este benchmark demuestra un baseline efectivo completo sobre la
superficie Gold actual de Databricks, pero **no demuestra paridad histórica
total de datos con PostgreSQL**.

## Estado de calidad

- Suite local completo: 290 tests aprobados.
- Ruff oficial del proyecto: `app scripts tests` aprobado.
- CI #69: aprobado.
- PR de trabajo: Draft PR #1 sobre el fork; upstream y `main` permanecen sin
  modificaciones directas.

## Próxima fase

Con el baseline E2E congelado, la siguiente fase debe centrarse en mejoras
arquitectónicas y experimentales generales, no en ajustar más casos del golden
set:

- dialecto SQL explícito end-to-end;
- observabilidad y tracking experimental;
- MLflow para ejecuciones/artefactos/métricas;
- comparación controlada entre backend relacional compatible y capa semántica
  Gold cuando vuelva a existir una fuente PostgreSQL auditable;
- medición repetida de latencia y costo, evitando conclusiones a partir de una
  sola corrida.
