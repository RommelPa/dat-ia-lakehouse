"""Runtime unificado para ejecutar consultas en PostgreSQL o Databricks.

Esta capa mantiene la compatibilidad con ``SQLDatabase`` mientras el monolito
``app.main`` se migra de forma incremental. PostgreSQL conserva su ruta actual;
Databricks usa ``DatabricksExecutor`` y omite por ahora el dry-run específico
del motor durante la validación determinística.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import sqlglot
from langchain_community.utilities import SQLDatabase
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.config import Settings
from app.database.backend import QueryBackend, create_query_backend
from app.database.databricks import DatabricksExecutor


@dataclass(slots=True)
class QueryRuntime:
    """Recursos de ejecución ya inicializados para el backend activo."""

    name: str
    executor: Any
    validation_db: SQLDatabase | None = None

    @property
    def sql_dialect(self) -> str:
        """Dialecto SQL canónico usado por generación y validación."""
        return "postgres" if self.name == "postgres" else "databricks"

    @property
    def dialect(self) -> str:
        """Nombre de motor mostrado en health/ready y observabilidad."""
        if self.name == "postgres" and self.validation_db is not None:
            return str(self.validation_db.dialect)
        return "databricks"

    def execute(
        self,
        sql_text: str,
        row_limit: int = 200,
        *,
        stage_timings_ms: MutableMapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Ejecuta SQL de solo lectura con un contrato común de salida."""
        if self.name == "databricks":
            return self.executor.execute(
                sql_text,
                row_limit=row_limit,
                timings_ms=stage_timings_ms,
            )

        return _execute_postgres(self.validation_db, sql_text, row_limit=row_limit)


def _execute_postgres(
    db: SQLDatabase | None,
    sql_text: str,
    *,
    row_limit: int,
) -> dict[str, Any]:
    """Conserva la semántica histórica de ``app.main.execute_sql``."""
    if db is None:
        return {"error": "El backend PostgreSQL no está inicializado."}

    stripped = str(sql_text or "").strip().rstrip(";")

    if ";" in stripped:
        return {"error": "Solo se permite una sentencia SQL por consulta."}

    try:
        expression = sqlglot.parse_one(stripped, read="postgres")
    except ParseError:
        return {"error": "Solo se permiten sentencias SELECT."}

    if not isinstance(expression, exp.Select):
        return {"error": "Solo se permiten sentencias SELECT."}

    if row_limit < 1:
        return {"error": "row_limit debe ser mayor que cero."}

    result = db.run_no_throw(stripped, fetch="cursor")

    if isinstance(result, str):
        return {"error": result}

    rows = [dict(row) for row in result.mappings()]
    return {"rows": rows[:row_limit]}


def create_query_runtime(settings: Settings) -> QueryRuntime:
    """Inicializa el runtime correspondiente a ``QUERY_BACKEND``."""
    backend: QueryBackend = create_query_backend(settings)

    if backend.name == "postgres":
        db = SQLDatabase(backend.resource, lazy_table_reflection=True)
        return QueryRuntime(
            name="postgres",
            executor=db,
            validation_db=db,
        )

    executor = backend.resource
    if not isinstance(executor, DatabricksExecutor):
        raise TypeError("El backend Databricks no creó un DatabricksExecutor válido.")

    return QueryRuntime(
        name="databricks",
        executor=executor,
        validation_db=None,
    )
