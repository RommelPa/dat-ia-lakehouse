"""Diagnóstico aislado de latencia del Databricks SQL Connector.

Abre una sola conexión y ejecuta la misma consulta SELECT varias veces para
separar costo de conexión frente a ejecuciones sucesivas sobre una conexión ya
abierta. No modifica el runtime de Dat-IA ni reutiliza conexiones en producción.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mide connect/cursor/execute/fetch/close de Databricks usando "
            "una sola conexión y una consulta SELECT repetida."
        )
    )
    parser.add_argument(
        "--sql",
        default="SELECT 1 AS value",
        help="Consulta SELECT de diagnóstico.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Número de ejecuciones sobre la misma conexión.",
    )
    return parser.parse_args()


def _validate_read_only_select(sql_text: str) -> str:
    stripped = str(sql_text or "").strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Solo se permite una sentencia SQL.")

    try:
        expression = sqlglot.parse_one(stripped, read="databricks")
    except ParseError as exc:
        raise ValueError("La consulta debe ser un SELECT válido.") from exc

    if not isinstance(expression, exp.Select):
        raise ValueError("Solo se permiten sentencias SELECT.")

    return stripped


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def diagnose_connection(
    *,
    connect: Callable[..., Any],
    settings: Settings,
    sql_text: str,
    runs: int,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("--runs debe ser mayor que cero.")

    sql = _validate_read_only_select(sql_text)
    if not settings.databricks_server_hostname:
        raise ValueError("Falta DATABRICKS_SERVER_HOSTNAME.")
    if not settings.databricks_http_path:
        raise ValueError("Falta DATABRICKS_HTTP_PATH.")

    connection = None
    cursor = None
    report: dict[str, Any] = {
        "runs": runs,
        "catalog": settings.databricks_catalog,
        "schema": settings.databricks_schema,
        "connect_ms": None,
        "cursor_ms": None,
        "executions": [],
        "close_ms": None,
    }

    try:
        started = time.perf_counter()
        connection = connect(
            server_hostname=settings.databricks_server_hostname,
            http_path=settings.databricks_http_path,
            auth_type=settings.databricks_auth_type,
            catalog=settings.databricks_catalog,
            schema=settings.databricks_schema,
        )
        report["connect_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        cursor = connection.cursor()
        report["cursor_ms"] = _elapsed_ms(started)

        for index in range(1, runs + 1):
            started = time.perf_counter()
            cursor.execute(sql)
            execute_ms = _elapsed_ms(started)

            started = time.perf_counter()
            rows = cursor.fetchmany(size=1)
            fetch_ms = _elapsed_ms(started)

            report["executions"].append(
                {
                    "run": index,
                    "execute_ms": execute_ms,
                    "fetch_ms": fetch_ms,
                    "rows_fetched": len(rows),
                }
            )

        return report
    finally:
        started = time.perf_counter()
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        report["close_ms"] = _elapsed_ms(started)


def main() -> None:
    args = parse_args()
    settings = Settings()

    try:
        from databricks import sql as databricks_sql
    except ImportError as exc:
        raise SystemExit(
            "Falta databricks-sql-connector en el entorno."
        ) from exc

    report = diagnose_connection(
        connect=databricks_sql.connect,
        settings=settings,
        sql_text=args.sql,
        runs=args.runs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
