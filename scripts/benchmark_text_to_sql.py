"""Benchmark end-to-end del agente Text-to-SQL contra la API Dat-IA.

A diferencia de ``benchmark_query_backends``, este script no ejecuta el SQL de
referencia. Envía la pregunta natural a ``/query/answer`` y evalúa la respuesta
con los mismos evaluadores determinísticos del golden set versionado.

Para Databricks conserva dos cortes de resultado/respuesta:
- canónico: baseline histórico validado contra PostgreSQL;
- efectivo: baseline del dataset realmente cargado en Databricks cuando existe
  un override documentado por data drift.

Los cinco casos de seguridad se mantienen fuera de este benchmark analítico y
se reportan como SKIP. Se medirán de forma separada para no mezclar contratos.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    answer_contains_expected_facts,
    generated_sql_is_read_only,
    golden_set_content_hash,
    load_golden_set,
    reported_source_tables_match_expected,
    response_status_matches_expected,
    result_facts_match_expected,
)
from app.observability.mlflow_tracking import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
    log_benchmark_report,
)
from scripts.benchmark_query_backends import (
    DEFAULT_DATABRICKS_OVERRIDES_PATH,
    _is_reference_sql_case,
    _load_databricks_overrides,
)
from scripts.evaluate_langsmith_golden_set import check_api_ready, request_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "archive"
    / "text_to_sql_databricks_benchmark.json"
)

COMPONENT_STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "llm": (
        "optimizer",
        "sql_generation",
        "sql_judgement",
        "answer_synthesis",
    ),
    "database": ("sql_execution",),
    "retrieval": ("memory_retrieval", "ddl_retrieval"),
    "guardrails": (
        "input_shield",
        "sql_validation",
        "result_guardrail",
        "groundedness",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa end-to-end las preguntas analíticas del golden set contra "
            "la API Dat-IA usando los evaluadores determinísticos existentes."
        )
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="URL base de la API Dat-IA.",
    )
    parser.add_argument(
        "--expected-backend",
        choices=("postgres", "databricks"),
        default="databricks",
        help="Backend que /ready debe reportar antes de ejecutar el benchmark.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
    )
    parser.add_argument(
        "--databricks-overrides-path",
        type=Path,
        default=DEFAULT_DATABRICKS_OVERRIDES_PATH,
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
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout por pregunta en segundos.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Devuelve exit code 1 si algún caso falla el resultado efectivo.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Registra el reporte en MLflow de forma opcional.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking URI de MLflow; por defecto usa SQLite local (./mlflow.db).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help="Nombre del experimento MLflow.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Nombre opcional del run MLflow.",
    )
    return parser.parse_args()


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


def effective_reference_outputs(
    case: Mapping[str, Any],
    *,
    backend: str,
    databricks_overrides: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Devuelve el contrato efectivo sin mutar el golden set canónico."""
    reference = deepcopy(case["reference_outputs"])
    case_id = str(case["case_id"])

    if backend != "databricks":
        return reference, None

    override = databricks_overrides.get(case_id)
    if not isinstance(override, Mapping):
        return reference, None

    expected_result = override.get("expected_result")
    if not isinstance(expected_result, Mapping):
        raise ValueError(f"{case_id}: override sin expected_result válido.")

    reference["expected_result"] = deepcopy(dict(expected_result))
    reason = str(override.get("reason") or "").strip() or None
    return reference, reason


def evaluate_case_output(
    outputs: Mapping[str, Any],
    *,
    canonical_reference: Mapping[str, Any],
    effective_reference: Mapping[str, Any],
) -> dict[str, bool]:
    """Aplica únicamente evaluadores existentes del golden set."""
    return {
        "status_match": response_status_matches_expected(
            outputs,
            canonical_reference,
        ),
        "sources_match": reported_source_tables_match_expected(
            outputs,
            canonical_reference,
        ),
        "sql_read_only": generated_sql_is_read_only(outputs),
        "canonical_result_match": result_facts_match_expected(
            outputs,
            canonical_reference,
        ),
        "effective_result_match": result_facts_match_expected(
            outputs,
            effective_reference,
        ),
        "canonical_answer_match": answer_contains_expected_facts(
            outputs,
            canonical_reference,
        ),
        "effective_answer_match": answer_contains_expected_facts(
            outputs,
            effective_reference,
        ),
    }


def _runner_error_output(exc: Exception) -> dict[str, Any]:
    return {
        "status": "runner_error",
        "answer": f"{type(exc).__name__}: {exc}",
        "sql": "",
        "data": [],
        "sources": "",
        "attempts": 0,
        "warnings": [],
    }


def _run_question(
    api_url: str,
    question: str,
    *,
    timeout: float,
) -> tuple[dict[str, Any], float, str | None]:
    started = time.perf_counter()
    runner_error = None

    try:
        output = request_json(
            f"{api_url.rstrip('/')}/query/answer",
            payload={"question": question},
            timeout=timeout,
        )
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        output = _runner_error_output(exc)
        runner_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # El benchmark debe conservar el fallo y continuar.
        output = _runner_error_output(exc)
        runner_error = f"{type(exc).__name__}: {exc}"

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return output, latency_ms, runner_error


def _metric_summary(
    cases: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    passed = [
        str(case["case_id"])
        for case in cases
        if bool(case["metrics"].get(metric))
    ]
    failed = [
        str(case["case_id"])
        for case in cases
        if not bool(case["metrics"].get(metric))
    ]
    total = len(cases)
    return {
        "passed": len(passed),
        "total": total,
        "rate": round(len(passed) / total, 4) if total else None,
        "failed_case_ids": failed,
    }


def _average_stage_latencies(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Promedia timings internos reportados por la API, por nombre de etapa."""
    samples: dict[str, list[float]] = {}

    for case in case_results:
        output = case.get("output")
        if not isinstance(output, Mapping):
            continue

        timings = output.get("timings_ms")
        if not isinstance(timings, Mapping):
            continue

        for stage, value in timings.items():
            if isinstance(value, (int, float)):
                samples.setdefault(str(stage), []).append(float(value))

    return {
        stage: round(sum(values) / len(values), 3)
        for stage, values in sorted(samples.items())
        if values
    }


def _average_stage_call_counts(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Promedia invocaciones visibles por etapa reportadas por la API."""
    samples: dict[str, list[int]] = {}

    for case in case_results:
        output = case.get("output")
        if not isinstance(output, Mapping):
            continue

        call_counts = output.get("stage_call_counts")
        if not isinstance(call_counts, Mapping):
            continue

        for stage, value in call_counts.items():
            if isinstance(value, int) and value >= 0:
                samples.setdefault(str(stage), []).append(value)

    return {
        stage: round(sum(values) / len(values), 3)
        for stage, values in sorted(samples.items())
        if values
    }

def _average_component_latencies(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Promedia latencia total por componente funcional del pipeline.

    Los componentes se calculan por caso y luego se promedian, evitando
    mezclar denominadores cuando una respuesta no llegó a ejecutar todas
    las etapas. "llm" representa llamadas LLM visibles en Dat-IA; no
    intenta inferir reintentos internos del proveedor.
    """
    samples: dict[str, list[float]] = {}

    for case in case_results:
        output = case.get("output")
        if not isinstance(output, Mapping):
            continue

        timings = output.get("timings_ms")
        if not isinstance(timings, Mapping):
            continue

        call_counts = output.get("stage_call_counts")

        for component, stages in COMPONENT_STAGE_GROUPS.items():
            values = []
            for stage in stages:
                value = timings.get(stage)
                if not isinstance(value, (int, float)):
                    continue

                # En modo rule_based el tiempo de "optimizer" es CPU local,
                # no una llamada LLM. La ausencia deliberada del contador
                # visible distingue ese caso sin romper reportes históricos
                # que todavía no tenían stage_call_counts.
                if (
                    component == "llm"
                    and stage == "optimizer"
                    and isinstance(call_counts, Mapping)
                    and not bool(call_counts.get("optimizer"))
                ):
                    continue

                values.append(float(value))

            if values:
                samples.setdefault(component, []).append(sum(values))

    return {
        component: round(sum(values) / len(values), 3)
        for component, values in sorted(samples.items())
        if values
    }

def _summary(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "status_match",
        "sources_match",
        "sql_read_only",
        "canonical_result_match",
        "effective_result_match",
        "canonical_answer_match",
        "effective_answer_match",
    )
    latencies = [float(case["latency_ms"]) for case in case_results]
    attempts = [
        int(case["output"].get("attempts") or 0)
        for case in case_results
    ]

    return {
        "analytical_cases": len(case_results),
        "metrics": {
            metric: _metric_summary(case_results, metric)
            for metric in metric_names
        },
        "runner_errors": sum(bool(case["runner_error"]) for case in case_results),
        "runner_error_case_ids": [
            str(case["case_id"])
            for case in case_results
            if case["runner_error"]
        ],
        "cases_with_warnings": sum(
            bool(case["output"].get("warnings"))
            for case in case_results
        ),
        "avg_attempts": round(sum(attempts) / len(attempts), 3)
        if attempts
        else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3)
        if latencies
        else None,
        "min_latency_ms": round(min(latencies), 3) if latencies else None,
        "max_latency_ms": round(max(latencies), 3) if latencies else None,
        "avg_stage_latency_ms": _average_stage_latencies(case_results),
        "avg_component_latency_ms": _average_component_latencies(case_results),
        "avg_stage_call_counts": _average_stage_call_counts(case_results),
    }


def benchmark(
    *,
    api_url: str,
    expected_backend: str,
    dataset_path: Path,
    databricks_overrides_path: Path,
    report_path: Path,
    case_ids: Sequence[str] = (),
    timeout: float = 180.0,
) -> dict[str, Any]:
    all_cases = load_golden_set(dataset_path)
    selected_cases = _select_cases(all_cases, case_ids)
    analytical_cases = [
        case for case in selected_cases if _is_reference_sql_case(case)
    ]
    skipped_cases = [
        case for case in selected_cases if not _is_reference_sql_case(case)
    ]

    if not analytical_cases:
        raise ValueError("La selección no contiene casos analíticos.")

    ready = check_api_ready(api_url, timeout=min(timeout, 10.0))
    actual_backend = str(ready.get("backend") or "").strip().casefold()
    if actual_backend != expected_backend:
        raise RuntimeError(
            "Backend inesperado en /ready: "
            f"esperado={expected_backend}, actual={actual_backend or 'missing'}"
        )

    databricks_overrides = (
        _load_databricks_overrides(databricks_overrides_path)
        if expected_backend == "databricks"
        else {}
    )

    for case in skipped_cases:
        print(f"{case['case_id']}: SKIP (caso no analítico)")

    results: list[dict[str, Any]] = []
    for case in analytical_cases:
        case_id = str(case["case_id"])
        canonical_reference = case["reference_outputs"]
        effective_reference, override_reason = effective_reference_outputs(
            case,
            backend=expected_backend,
            databricks_overrides=databricks_overrides,
        )
        output, latency_ms, runner_error = _run_question(
            api_url,
            str(case["inputs"]["question"]),
            timeout=timeout,
        )
        metrics = evaluate_case_output(
            output,
            canonical_reference=canonical_reference,
            effective_reference=effective_reference,
        )

        record = {
            "case_id": case_id,
            "question": case["inputs"]["question"],
            "override_applied": override_reason is not None,
            "override_reason": override_reason,
            "latency_ms": latency_ms,
            "runner_error": runner_error,
            "metrics": metrics,
            "output": output,
        }
        results.append(record)

        result_label = "OK" if metrics["effective_result_match"] else "FAIL"
        suffix = "/OVERRIDE" if override_reason else ""
        print(
            f"{case_id}: RESULT={result_label}{suffix} | "
            f"STATUS={'OK' if metrics['status_match'] else 'FAIL'} | "
            f"SOURCES={'OK' if metrics['sources_match'] else 'FAIL'} | "
            f"SQL={'OK' if metrics['sql_read_only'] else 'FAIL'} | "
            f"{latency_ms:.0f} ms"
        )

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_type": "text_to_sql_end_to_end",
        "api_url": api_url,
        "backend": expected_backend,
        "ready": ready,
        "dataset_version": selected_cases[0]["metadata"]["dataset_version"],
        "dataset_content_sha256": golden_set_content_hash(all_cases),
        "selected_cases": len(selected_cases),
        "skipped_cases": len(skipped_cases),
        "skipped_case_ids": [str(case["case_id"]) for case in skipped_cases],
        "override_case_ids": [
            str(case["case_id"])
            for case in results
            if case["override_applied"]
        ],
        "summary": _summary(results),
        "cases": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _track_with_mlflow(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
) -> str | None:
    """Registra el benchmark solo cuando el usuario lo solicita."""
    if not enabled:
        return None

    return log_benchmark_report(
        report,
        report_path=report_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout debe ser mayor que cero.")

    report = benchmark(
        api_url=args.api_url,
        expected_backend=args.expected_backend,
        dataset_path=args.dataset_path,
        databricks_overrides_path=args.databricks_overrides_path,
        report_path=args.report_path,
        case_ids=args.case_id,
        timeout=args.timeout,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Reporte: {args.report_path}")

    mlflow_run_id = _track_with_mlflow(
        report,
        report_path=args.report_path,
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    )
    if mlflow_run_id is not None:
        print(f"MLflow run_id: {mlflow_run_id}")

    effective = report["summary"]["metrics"]["effective_result_match"]
    if args.fail_on_mismatch and effective["passed"] != effective["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
