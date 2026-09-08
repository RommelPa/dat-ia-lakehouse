"""Compara el golden set canónico entre PostgreSQL y Databricks Gold.

Este benchmark ejecuta los SQL de referencia directamente, sin optimizer,
retrieval, Gemini ni judge. Su objetivo es aislar la paridad de datos y motor
antes del benchmark end-to-end del agente Text-to-SQL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlglot
from sqlalchemy import text

from app.db.connect_db import create_db_engine
from app.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    compare_result_facts,
    golden_set_content_hash,
    is_read_only_sql,
    load_golden_set,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "archive"
    / "query_backend_reference_benchmark.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el SQL canónico del golden set contra PostgreSQL y/o "
            "Databricks Gold y compara los resultados."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("both", "postgres", "databricks"),
        default="both",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Ejecuta solo el case_id indicado; puede repetirse.",
    )
    parser.add_argument(
        "--allow-mismatches",
        action="store_true",
        help="Genera el reporte pero no devuelve código de salida 1 ante diferencias.",
    )
    return parser.parse_args()


def transpile_reference_sql(sql: str) -> str:
    """Traduce SQL PostgreSQL de referencia al dialecto Databricks."""
    if not is_read_only_sql(sql):
        raise ValueError("El SQL de referencia no cumple la política de solo lectura.")

    statements = sqlglot.transpile(sql, read="postgres", write="databricks")
    if len(statements) != 1:
        raise ValueError("Se esperaba exactamente una sentencia SQL de referencia.")
    return statements[0]


def rows_have_parity(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> bool:
    """Compara dos resultados en ambas direcciones para exigir hechos equivalentes."""
    left_contract = {
        "row_count": len(left_rows),
        "rows": list(left_rows),
        "numeric_tolerance": tolerance,
    }
    right_contract = {
        "row_count": len(right_rows),
        "rows": list(right_rows),
        "numeric_tolerance": tolerance,
    }
    return compare_result_facts(right_rows, left_contract) and compare_result_facts(
        left_rows,
        right_contract,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return value.hex
    if isinstance(value, bytes):
        return value.hex()
    return value


def _json_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in rows
    ]


def _timed_postgres_query(connection: Any, sql: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with connection.begin_nested():
            result = connection.execute(text(sql))
            rows = [dict(row) for row in result.mappings()]
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "rows": rows,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _timed_databricks_query(connection: Any, sql: str) -> dict[str, Any]:
    started = time.perf_counter()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [str(column[0]) for column in (cursor.description or [])]
        rows = [
            {
                column_name: value
                for column_name, value in zip(columns, raw_row, strict=False)
            }
            for raw_row in cursor.fetchall()
        ]
        return {
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "rows": rows,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if cursor is not None:
            cursor.close()


def _open_postgres(stack: ExitStack, database_url: str) -> tuple[Any, float]:
    engine = create_db_engine(database_url)
    stack.callback(engine.dispose)
    started = time.perf_counter()
    connection = stack.enter_context(engine.connect())
    setup_ms = round((time.perf_counter() - started) * 1000, 3)
    transaction = stack.enter_context(connection.begin())
    del transaction
    if connection.dialect.name == "postgresql":
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
    return connection, setup_ms


def _open_databricks(stack: ExitStack) -> tuple[Any, float]:
    try:
        from databricks import sql as databricks_sql
    except ImportError as exc:
        raise RuntimeError(
            "Falta databricks-sql-connector; ejecuta `uv sync --frozen`."
        ) from exc

    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    if not server_hostname or not http_path:
        raise RuntimeError(
            "Databricks requiere DATABRICKS_SERVER_HOSTNAME y DATABRICKS_HTTP_PATH."
        )

    started = time.perf_counter()
    connection = databricks_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        auth_type=os.getenv("DATABRICKS_AUTH_TYPE", "databricks-oauth"),
        catalog=os.getenv("DATABRICKS_CATALOG", "dat_ia"),
        schema=os.getenv("DATABRICKS_SCHEMA", "gold"),
    )
    setup_ms = round((time.perf_counter() - started) * 1000, 3)
    stack.callback(connection.close)
    return connection, setup_ms


def _select_cases(
    cases: list[dict[str, Any]],
    requested_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not requested_case_ids:
        return cases
    requested = set(requested_case_ids)
    selected = [case for case in cases if case["case_id"] in requested]
    missing = sorted(requested - {case["case_id"] for case in selected})
    if missing:
        raise ValueError("case_id no encontrado: " + ", ".join(missing))
    return selected


def _backend_summary(cases: Sequence[Mapping[str, Any]], backend: str) -> dict[str, Any]:
    records = [case[backend] for case in cases if case.get(backend) is not None]
    latencies = [float(record["latency_ms"]) for record in records if record["ok"]]
    return {
        "executed": sum(bool(record["ok"]) for record in records),
        "expected_match": sum(bool(record.get("expected_match")) for record in records),
        "errors": sum(not bool(record["ok"]) for record in records),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "min_latency_ms": round(min(latencies), 3) if latencies else None,
        "max_latency_ms": round(max(latencies), 3) if latencies else None,
    }


def benchmark(
    *,
    backend: str,
    dataset_path: Path,
    report_path: Path,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    cases = _select_cases(load_golden_set(dataset_path), case_ids)
    run_postgres = backend in {"both", "postgres"}
    run_databricks = backend in {"both", "databricks"}

    database_url = os.getenv("DATABASE_URL")
    if run_postgres and not database_url:
        raise RuntimeError("PostgreSQL requiere DATABASE_URL.")

    case_results: list[dict[str, Any]] = []
    setup_latency_ms: dict[str, float] = {}

    with ExitStack() as stack:
        postgres_connection = None
        databricks_connection = None
        if run_postgres:
            postgres_connection, setup_latency_ms["postgres"] = _open_postgres(
                stack,
                str(database_url),
            )
        if run_databricks:
            databricks_connection, setup_latency_ms["databricks"] = _open_databricks(
                stack
            )

        for case in cases:
            reference = case["reference_outputs"]
            expected = reference["expected_result"]
            tolerance = float(expected.get("numeric_tolerance", 0.0))
            reference_sql = str(reference["reference_sql"])
            databricks_sql = transpile_reference_sql(reference_sql)

            record: dict[str, Any] = {
                "case_id": case["case_id"],
                "question": case["inputs"]["question"],
                "reference_sql": reference_sql,
                "databricks_sql": databricks_sql,
                "sql_changed_for_databricks": reference_sql.rstrip(";") != databricks_sql.rstrip(";"),
                "postgres": None,
                "databricks": None,
                "cross_backend_parity": None,
            }

            postgres_raw = None
            if postgres_connection is not None:
                postgres_raw = _timed_postgres_query(postgres_connection, reference_sql)
                postgres_raw["expected_match"] = postgres_raw["ok"] and compare_result_facts(
                    postgres_raw["rows"], expected
                )
                record["postgres"] = {
                    **postgres_raw,
                    "rows": _json_rows(postgres_raw["rows"]),
                }

            databricks_raw = None
            if databricks_connection is not None:
                databricks_raw = _timed_databricks_query(
                    databricks_connection,
                    databricks_sql,
                )
                databricks_raw["expected_match"] = databricks_raw["ok"] and compare_result_facts(
                    databricks_raw["rows"], expected
                )
                record["databricks"] = {
                    **databricks_raw,
                    "rows": _json_rows(databricks_raw["rows"]),
                }

            if postgres_raw is not None and databricks_raw is not None:
                record["cross_backend_parity"] = (
                    postgres_raw["ok"]
                    and databricks_raw["ok"]
                    and rows_have_parity(
                        postgres_raw["rows"],
                        databricks_raw["rows"],
                        tolerance=tolerance,
                    )
                )

            case_results.append(record)
            status_parts = []
            if record["postgres"] is not None:
                status_parts.append(
                    f"PG={'OK' if record['postgres']['expected_match'] else 'FAIL'}"
                )
            if record["databricks"] is not None:
                status_parts.append(
                    f"DBX={'OK' if record['databricks']['expected_match'] else 'FAIL'}"
                )
            if record["cross_backend_parity"] is not None:
                status_parts.append(
                    f"PARITY={'OK' if record['cross_backend_parity'] else 'FAIL'}"
                )
            print(f"{record['case_id']}: " + " | ".join(status_parts))

    summary: dict[str, Any] = {"total_cases": len(case_results)}
    if run_postgres:
        summary["postgres"] = _backend_summary(case_results, "postgres")
    if run_databricks:
        summary["databricks"] = _backend_summary(case_results, "databricks")
    if run_postgres and run_databricks:
        summary["cross_backend_parity"] = sum(
            case["cross_backend_parity"] is True for case in case_results
        )

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": cases[0]["metadata"]["dataset_version"],
        "dataset_content_sha256": golden_set_content_hash(cases),
        "dataset_path": str(dataset_path),
        "benchmark_type": "reference_sql_backend_parity",
        "backend_scope": backend,
        "setup_latency_ms": setup_latency_ms,
        "summary": summary,
        "cases": case_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def report_is_green(report: Mapping[str, Any]) -> bool:
    """Indica si todos los motores seleccionados y su paridad quedaron verdes."""
    summary = report["summary"]
    total = int(summary["total_cases"])
    for backend in ("postgres", "databricks"):
        if backend in summary and int(summary[backend]["expected_match"]) != total:
            return False
    if "cross_backend_parity" in summary:
        return int(summary["cross_backend_parity"]) == total
    return True


def main() -> None:
    args = parse_args()
    report = benchmark(
        backend=args.backend,
        dataset_path=args.dataset_path,
        report_path=args.report_path,
        case_ids=args.case_id,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Reporte: {args.report_path}")
    if not args.allow_mismatches and not report_is_green(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
