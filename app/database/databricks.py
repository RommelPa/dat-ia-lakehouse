"""Acceso al backend analítico Databricks SQL.

El módulo mantiene una dependencia opcional sobre ``databricks-sql-connector``:
se importa únicamente cuando se crea una conexión real. Esto permite que el
resto de Dat-IA y sus tests sigan funcionando sin instalar el conector hasta
que el backend Databricks sea activado explícitamente.
"""

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.observability.timing import timed_call


ConnectCallable = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class DatabricksSqlConfig:
    """Destino lógico para consultas SQL contra Databricks."""

    server_hostname: str
    http_path: str
    catalog: str = "dat_ia"
    schema: str = "gold"
    auth_type: str = "databricks-oauth"

    def __post_init__(self) -> None:
        """Rechaza configuraciones incompletas antes de crear un cliente."""
        required = {
            "server_hostname": self.server_hostname,
            "http_path": self.http_path,
            "catalog": self.catalog,
            "schema": self.schema,
            "auth_type": self.auth_type,
        }

        missing = [
            name
            for name, value in required.items()
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(
                "DatabricksSqlConfig requiere valores no vacíos para: "
                + ", ".join(missing)
            )

    @property
    def namespace(self) -> str:
        """Namespace ``catalog.schema`` usado por el backend analítico."""
        return f"{self.catalog}.{self.schema}"


def _default_connect(**kwargs: Any):
    """Carga el SQL Connector solo cuando Databricks se usa realmente."""
    try:
        from databricks import sql
    except ImportError as exc:  # pragma: no cover - depende del entorno local
        raise RuntimeError(
            "El backend Databricks requiere 'databricks-sql-connector'. "
            "Instálalo antes de activar este backend."
        ) from exc

    return sql.connect(**kwargs)


class DatabricksExecutor:
    """Ejecutor SQL de solo lectura sobre un Databricks SQL Warehouse.

    Devuelve el mismo contrato que usa actualmente Dat-IA para PostgreSQL:
    ``{"rows": [...]}`` en éxito o ``{"error": "..."}`` ante cualquier
    rechazo o error de ejecución.
    """

    def __init__(
        self,
        config: DatabricksSqlConfig,
        *,
        connect: ConnectCallable | None = None,
    ) -> None:
        self.config = config
        self._connect = connect or _default_connect
        self._connection: Any | None = None
        self._connection_lock = Lock()

    def _get_connection(
        self,
        timings_ms: MutableMapping[str, float] | None,
    ) -> Any:
        """Abre la conexión una sola vez y la reutiliza mientras siga válida."""
        if self._connection is None:
            self._connection = timed_call(
                timings_ms,
                "databricks_connect",
                self._connect,
                server_hostname=self.config.server_hostname,
                http_path=self.config.http_path,
                auth_type=self.config.auth_type,
                catalog=self.config.catalog,
                schema=self.config.schema,
            )
        return self._connection

    def _close_connection_unlocked(
        self,
        timings_ms: MutableMapping[str, float] | None = None,
    ) -> None:
        connection = self._connection
        self._connection = None

        if connection is None:
            return

        try:
            timed_call(
                timings_ms,
                "databricks_connection_close",
                connection.close,
            )
        except Exception:
            pass

    def close(self) -> None:
        """Cierra explícitamente la sesión compartida del executor."""
        with self._connection_lock:
            self._close_connection_unlocked()

    def execute(
        self,
        sql_text: str,
        row_limit: int = 200,
        *,
        timings_ms: MutableMapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Ejecuta una consulta y mide opcionalmente subetapas Databricks."""
        stripped = str(sql_text or "").strip().rstrip(";")

        if ";" in stripped:
            return {"error": "Solo se permite una sentencia SQL por consulta."}

        try:
            expression = sqlglot.parse_one(stripped, read="databricks")
        except ParseError:
            return {"error": "Solo se permiten sentencias SELECT."}

        # Un SELECT con CTE sigue siendo exp.Select en sqlglot; esto
        # permite WITH ... SELECT sin abrir la puerta a DML/DDL.
        if not isinstance(expression, exp.Select):
            return {"error": "Solo se permiten sentencias SELECT."}

        if row_limit < 1:
            return {"error": "row_limit debe ser mayor que cero."}

        cursor = None
        invalidate_connection = False

        with self._connection_lock:
            try:
                connection = self._get_connection(timings_ms)
                cursor = timed_call(
                    timings_ms,
                    "databricks_cursor",
                    connection.cursor,
                )
                timed_call(
                    timings_ms,
                    "databricks_execute",
                    cursor.execute,
                    stripped,
                )

                description = cursor.description or []
                columns = [str(column[0]) for column in description]
                raw_rows = timed_call(
                    timings_ms,
                    "databricks_fetch",
                    cursor.fetchmany,
                    size=row_limit,
                )

                rows = [
                    {
                        column_name: value
                        for column_name, value in zip(columns, row, strict=False)
                    }
                    for row in raw_rows
                ]

                return {"rows": rows}
            except Exception as exc:
                invalidate_connection = True
                return {"error": str(exc)}
            finally:
                if cursor is not None:
                    try:
                        timed_call(
                            timings_ms,
                            "databricks_cursor_close",
                            cursor.close,
                        )
                    except Exception:
                        pass

                if invalidate_connection:
                    self._close_connection_unlocked(timings_ms)
