# Investigación de latencia Gemini y controles de resiliencia

## Objetivo

Separar los cuellos de botella del pipeline Text-to-SQL de Dat-IA y determinar
si la latencia observada proviene de Databricks, de llamadas LLM visibles en la
aplicación o de reintentos/esperas dentro del cliente Gemini.

La investigación se realizó sobre el backend Databricks, manteniendo el golden
set sin modificaciones y usando únicamente configuraciones locales/gratuitas.

## Estado de referencia

La conexión Databricks ya se reutiliza dentro del proceso de la API. En una
corrida secuencial de tres preguntas distintas, solo el primer caso pagó el
costo de conexión; los siguientes reutilizaron la sesión.

| Caso | Latencia total | SQL execution | Databricks connect |
|---|---:|---:|---:|
| golden_004 | 34.52 s | 17.21 s | 9.18 s |
| golden_005 | 15.80 s | 1.72 s | — |
| golden_007 | 17.45 s | 1.56 s | — |

Los tres casos conservaron resultado y respuesta efectivos correctos. En los
casos warm, el componente LLM pasó a dominar la latencia total.

## Instrumentación disponible

Dat-IA registra actualmente:

- latencia por etapa del pipeline;
- latencia agregada por componente;
- número de invocaciones LLM visibles desde la aplicación;
- subetapas de Databricks SQL;
- modo del optimizer;
- configuración Gemini de retries y timeout en /ready y MLflow.

Las llamadas visibles se cuentan en:

- optimizer;
- sql_generation;
- sql_judgement;
- answer_synthesis.

Un contador igual a 1 representa una invocación `.invoke()` desde Dat-IA. No
equivale necesariamente a un único intento HTTP dentro del SDK.

## Experimento del optimizer rule-based

Se evaluó el modo `rule_based` con `golden_004`, `golden_005` y
`golden_007`.

Resultados:

- 3/3 resultados efectivos correctos;
- 3/3 respuestas efectivas correctas;
- optimizer promedio: 7.724 ms;
- ninguna llamada LLM visible del optimizer.

La corrida completa presentó outliers severos en otras etapas LLM, por lo que
la latencia end-to-end no sirve para atribuir una mejora o regresión al
optimizer en esa observación. El hallazgo válido es que el optimizer
determinístico elimina una llamada LLM visible sin perder calidad en esta
muestra pequeña.

No se cambió el default: `hybrid` permanece como modo oficial hasta validar
una muestra más amplia.

## Configuración Gemini observada

Entorno inspeccionado:

- `langchain-google-genai = 4.2.7`;
- default de `ChatGoogleGenerativeAI.max_retries = 6`;
- default de `timeout = None`.

El código instalado de 4.2.7 transforma el timeout de segundos a milisegundos
y construye las opciones HTTP así:

```text
timeout_seconds -> int(timeout_seconds * 1000)
max_retries -> HttpRetryOptions(attempts=max_retries)
```

Por tanto, en esta versión el valor expuesto como `max_retries` se entrega al
SDK como número de `attempts`.

Dat-IA incorporó dos Settings configurables sin alterar los defaults:

```text
GEMINI_MAX_RETRIES=6
GEMINI_TIMEOUT_SECONDS=<sin valor>
```

## Experimento 2 attempts / 30 s

Configuración:

```text
QUERY_OPTIMIZER_MODE=hybrid
GEMINI_MAX_RETRIES=2
GEMINI_TIMEOUT_SECONDS=30
```

### Primera corrida

Casos: `golden_004`, `golden_005`, `golden_007`.

- calidad: 3/3 en resultado y respuesta;
- latencia promedio: 45.39 s;
- LLM promedio: 35.03 s;
- database promedio: 8.11 s.

Promedios LLM por etapa:

| Etapa | Latencia promedio |
|---|---:|
| optimizer | 13.49 s |
| sql_generation | 14.57 s |
| sql_judgement | 3.23 s |
| answer_synthesis | 3.74 s |

La primera pregunta volvió a pagar cold start de Databricks.

### Segunda corrida warm

Se repitieron los mismos tres casos sin reiniciar Uvicorn.

- calidad: 3/3;
- latencia promedio: 67.28 s;
- LLM promedio: 63.12 s;
- database promedio: 1.38 s.

Promedios LLM por etapa:

| Etapa | Latencia promedio |
|---|---:|
| optimizer | 6.82 s |
| sql_generation | 22.39 s |
| sql_judgement | 1.95 s |
| answer_synthesis | 31.97 s |

La sesión Databricks warm confirma que la base de datos no explica la mayor
parte de esta latencia. Los outliers se desplazaron entre distintas etapas LLM,
por lo que no hay evidencia de un único prompt o etapa como cuello de botella
estable.

## Experimento 1 attempt / 30 s

Configuración:

```text
QUERY_OPTIMIZER_MODE=hybrid
GEMINI_MAX_RETRIES=1
GEMINI_TIMEOUT_SECONDS=30
```

Resultados:

- `golden_004`: success;
- `golden_005`: HTTP 500;
- `golden_007`: HTTP 500;
- calidad total: 1/3.

La configuración queda descartada. Reducir a un único attempt deterioró la
resiliencia de la aplicación.

La corrida también reveló un hueco de observabilidad: los HTTP 500 perdían la
etapa que había fallado y los timings acumulados del request.

## Mejora de observabilidad derivada

Se añadió `StageExecutionError` y una respuesta segura estructurada para
errores de etapas instrumentadas.

Un fallo de proveedor puede conservar ahora:

```json
{
  "status": "provider_error",
  "error_stage": "sql_generation",
  "error_type": "TimeoutError",
  "timings_ms": {},
  "stage_call_counts": {}
}
```

El mensaje original de la excepción no se devuelve al cliente, para evitar
exponer detalles internos o sensibles.

Validación posterior:

- Ruff completo: aprobado;
- suite local: 334 tests aprobados;
- CI #150: aprobado.

## Conclusiones

1. La reutilización de conexión Databricks funciona y reduce el costo de SQL en
   requests warm a aproximadamente 1–2 s en los casos observados.
2. La mayor variabilidad actual proviene de las llamadas Gemini, no de
   Databricks.
3. `2/30` conserva calidad en la muestra pequeña, pero no elimina la cola larga
   de latencia.
4. `1/30` reduce demasiado la resiliencia y queda descartado.
5. No existe evidencia suficiente para cambiar los defaults de Gemini.
6. La configuración oficial permanece:
   - `QUERY_OPTIMIZER_MODE=hybrid`;
   - `GEMINI_MAX_RETRIES=6`;
   - `GEMINI_TIMEOUT_SECONDS=None`.
7. La latencia de una única corrida no debe usarse para afirmar mejoras de
   rendimiento cuando existen outliers externos del proveedor.

## Próximo experimento: ampliar la evaluación rule-based

Antes de considerar `rule_based` como default, debe probarse una muestra que
incluya operaciones simples y consultas donde el optimizer LLM puede aportar
semántica adicional.

Muestra propuesta de 12 casos:

| Caso | Cobertura principal |
|---|---|
| golden_004 | distinct / compradores únicos |
| golden_005 | agrupación simple |
| golden_011 | serie temporal + regla de entregadas |
| golden_012 | porcentaje + fechas |
| golden_014 | ranking + múltiples joins + categoría |
| golden_015 | métrica monetaria + ranking |
| golden_018 | agrupación + porcentaje |
| golden_021 | agrupación + HAVING/subconsulta |
| golden_022 | CTE + joins + umbral mínimo + data drift documentado |
| golden_023 | EXISTS + deduplicación lógica |
| golden_027 | joins + COUNT DISTINCT |
| golden_030 | serie temporal + ingreso |

Criterio de avance:

- resultado efectivo 12/12;
- respuesta efectiva 12/12;
- cero runner errors;
- revisar `optimized`, retrieval y SQL de cualquier discrepancia antes de
  modificar reglas;
- no cambiar el default con una sola corrida;
- las comparaciones de latencia deben tratarse por separado de la corrección y
  preferir observaciones warm/repetidas.
