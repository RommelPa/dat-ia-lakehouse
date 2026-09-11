"""Tracking opcional de benchmarks con MLflow.

La integración es deliberadamente perezosa: importar Dat-IA no requiere
`mlflow`. El paquete solo se carga cuando el usuario activa tracking en un
benchmark. Así el runtime base sigue liviano y el tracking local usa SQLite
sin servicios pagos ni credenciales.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MLFLOW_EXPERIMENT = "dat_ia_text_to_sql"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def _load_mlflow() -> Any:
    """Carga MLflow bajo demanda y devuelve un error accionable si falta."""
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow no está instalado. Instálalo solo si deseas tracking "
            "experimental; el benchmark funciona sin esta dependencia."
        ) from exc


def _summary_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Convierte el summary del benchmark en métricas numéricas de MLflow."""
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return {}

    metrics: dict[str, float] = {}
    metric_groups = summary.get("metrics")
    if isinstance(metric_groups, Mapping):
        for name, payload in metric_groups.items():
            if not isinstance(payload, Mapping):
                continue

            rate = payload.get("rate")
            passed = payload.get("passed")
            total = payload.get("total")

            if isinstance(rate, (int, float)):
                metrics[f"{name}.rate"] = float(rate)
            if isinstance(passed, (int, float)):
                metrics[f"{name}.passed"] = float(passed)
            if isinstance(total, (int, float)):
                metrics[f"{name}.total"] = float(total)

    scalar_names = (
        "runner_errors",
        "cases_with_warnings",
        "avg_attempts",
        "avg_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
    )
    for name in scalar_names:
        value = summary.get(name)
        if isinstance(value, (int, float)):
            metrics[name] = float(value)

    stage_latencies = summary.get("avg_stage_latency_ms")
    if isinstance(stage_latencies, Mapping):
        for stage, value in stage_latencies.items():
            if isinstance(value, (int, float)):
                metrics[f"stage_latency.{stage}.avg_ms"] = float(value)

    return metrics


def _report_params(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extrae parámetros de baja cardinalidad y sin secretos."""
    params: dict[str, Any] = {}
    keys = (
        "benchmark_type",
        "backend",
        "dataset_version",
        "dataset_content_sha256",
        "selected_cases",
        "skipped_cases",
    )

    for key in keys:
        value = report.get(key)
        if value is not None:
            params[key] = value

    override_case_ids = report.get("override_case_ids")
    if isinstance(override_case_ids, list):
        normalized_overrides = [
            str(case_id).strip()
            for case_id in override_case_ids
            if str(case_id).strip()
        ]
        params["override_count"] = len(normalized_overrides)
        if normalized_overrides:
            params["override_case_ids"] = ",".join(normalized_overrides)

    summary = report.get("summary")
    if isinstance(summary, Mapping):
        runner_error_case_ids = summary.get("runner_error_case_ids")
        if isinstance(runner_error_case_ids, list):
            normalized_runner_errors = [
                str(case_id).strip()
                for case_id in runner_error_case_ids
                if str(case_id).strip()
            ]
            if normalized_runner_errors:
                params["runner_error_case_ids"] = ",".join(
                    normalized_runner_errors
                )

    ready = report.get("ready")
    if isinstance(ready, Mapping):
        for key in ("backend", "database"):
            value = ready.get(key)
            if value is not None:
                params[f"ready_{key}"] = value

    return params


def log_benchmark_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    run_name: str | None = None,
    mlflow_module: Any | None = None,
) -> str:
    """Registra métricas, parámetros y el JSON del benchmark en MLflow.

    Returns:
        run_id de MLflow para poder enlazar el benchmark con documentación
        posterior.
    """
    mlflow = mlflow_module or _load_mlflow()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_report_params(report))
        mlflow.log_metrics(_summary_metrics(report))

        if report_path.is_file():
            mlflow.log_artifact(str(report_path), artifact_path="reports")

        run_id = str(run.info.run_id)

    return run_id
