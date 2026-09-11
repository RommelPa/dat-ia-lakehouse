from __future__ import annotations

from scripts.benchmark_query_backends import (
    DEFAULT_DATABRICKS_OVERRIDES_PATH,
    _load_databricks_overrides,
)
from scripts import benchmark_text_to_sql as benchmark_module
from scripts.benchmark_text_to_sql import (
    _average_component_latencies,
    _average_stage_call_counts,
    _average_stage_latencies,
    _track_with_mlflow,
    effective_reference_outputs,
    evaluate_case_output,
)


def test_effective_reference_uses_databricks_override_without_mutating_case() -> None:
    overrides = _load_databricks_overrides(DEFAULT_DATABRICKS_OVERRIDES_PATH)
    case = {
        "case_id": "golden_017",
        "reference_outputs": {
            "expected_status": "success",
            "expected_sources": ["olist_order_reviews_dataset"],
            "expected_result": {
                "row_count": 1,
                "rows": [{"reviews": 999}],
                "numeric_tolerance": 0.01,
            },
        },
    }

    effective, reason = effective_reference_outputs(
        case,
        backend="databricks",
        databricks_overrides=overrides,
    )

    assert reason
    assert effective["expected_result"]["row_count"] == 5
    assert case["reference_outputs"]["expected_result"]["row_count"] == 1


def test_effective_reference_keeps_canonical_contract_for_postgres() -> None:
    case = {
        "case_id": "golden_001",
        "reference_outputs": {
            "expected_status": "success",
            "expected_sources": ["olist_orders_dataset"],
            "expected_result": {
                "row_count": 1,
                "rows": [{"total": 99441}],
                "numeric_tolerance": 0.01,
            },
        },
    }

    effective, reason = effective_reference_outputs(
        case,
        backend="postgres",
        databricks_overrides={},
    )

    assert reason is None
    assert effective == case["reference_outputs"]
    assert effective is not case["reference_outputs"]


def test_evaluate_case_output_reports_canonical_and_effective_result_separately() -> None:
    outputs = {
        "status": "success",
        "sources": "olist_order_reviews_dataset",
        "sql": (
            "SELECT review_score, COUNT(*) AS reviews "
            "FROM olist_order_reviews_dataset GROUP BY review_score"
        ),
        "data": [
            {"review_score": 1, "reviews": 11424},
            {"review_score": 2, "reviews": 3151},
        ],
        "answer": "Las calificaciones 1 y 2 tienen 11.424 y 3.151 reseñas.",
    }
    canonical = {
        "expected_status": "success",
        "expected_sources": ["olist_order_reviews_dataset"],
        "expected_result": {
            "row_count": 2,
            "rows": [
                {"review_score": 1, "reviews": 14891},
                {"review_score": 2, "reviews": 4091},
            ],
            "numeric_tolerance": 0.01,
        },
    }
    effective = {
        **canonical,
        "expected_result": {
            "row_count": 2,
            "rows": [
                {"review_score": 1, "reviews": 11424},
                {"review_score": 2, "reviews": 3151},
            ],
            "numeric_tolerance": 0.01,
        },
    }

    metrics = evaluate_case_output(
        outputs,
        canonical_reference=canonical,
        effective_reference=effective,
    )

    assert metrics["status_match"] is True
    assert metrics["sources_match"] is True
    assert metrics["sql_read_only"] is True
    assert metrics["canonical_result_match"] is False
    assert metrics["effective_result_match"] is True


def test_track_with_mlflow_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    called = {"value": False}

    def fake_log(*args, **kwargs):
        called["value"] = True
        return "unexpected"

    monkeypatch.setattr(benchmark_module, "log_benchmark_report", fake_log)

    run_id = _track_with_mlflow(
        {"summary": {}},
        report_path=tmp_path / "report.json",
        enabled=False,
        tracking_uri="file:./mlruns",
        experiment_name="dat_ia_text_to_sql",
        run_name=None,
    )

    assert run_id is None
    assert called["value"] is False


def test_track_with_mlflow_forwards_tracking_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_log(
        report,
        *,
        report_path,
        tracking_uri,
        experiment_name,
        run_name,
    ):
        captured["report"] = report
        captured["report_path"] = report_path
        captured["tracking_uri"] = tracking_uri
        captured["experiment_name"] = experiment_name
        captured["run_name"] = run_name
        return "run-456"

    monkeypatch.setattr(benchmark_module, "log_benchmark_report", fake_log)
    report = {"summary": {}}
    report_path = tmp_path / "report.json"

    run_id = _track_with_mlflow(
        report,
        report_path=report_path,
        enabled=True,
        tracking_uri="file:./mlruns-test",
        experiment_name="experiment-test",
        run_name="run-test",
    )

    assert run_id == "run-456"
    assert captured == {
        "report": report,
        "report_path": report_path,
        "tracking_uri": "file:./mlruns-test",
        "experiment_name": "experiment-test",
        "run_name": "run-test",
    }


def test_average_stage_latencies_uses_api_timings() -> None:
    cases = [
        {
            "output": {
                "timings_ms": {
                    "optimizer": 100.0,
                    "sql_generation": 200.0,
                }
            }
        },
        {
            "output": {
                "timings_ms": {
                    "optimizer": 300.0,
                    "sql_generation": 400.0,
                    "sql_execution": 50.0,
                }
            }
        },
        {"output": {"status": "runner_error"}},
    ]

    result = _average_stage_latencies(cases)

    assert result == {
        "optimizer": 200.0,
        "sql_execution": 50.0,
        "sql_generation": 300.0,
    }


def test_average_component_latencies_groups_pipeline_stages() -> None:
    cases = [
        {
            "output": {
                "timings_ms": {
                    "optimizer": 100.0,
                    "sql_generation": 200.0,
                    "sql_judgement": 50.0,
                    "answer_synthesis": 25.0,
                    "sql_execution": 80.0,
                    "memory_retrieval": 10.0,
                    "ddl_retrieval": 20.0,
                    "input_shield": 5.0,
                    "sql_validation": 2.0,
                    "result_guardrail": 1.0,
                    "groundedness": 1.0,
                }
            }
        },
        {
            "output": {
                "timings_ms": {
                    "optimizer": 300.0,
                    "sql_generation": 100.0,
                    "sql_execution": 120.0,
                    "ddl_retrieval": 40.0,
                    "input_shield": 7.0,
                }
            }
        },
        {"output": {"status": "runner_error"}},
    ]

    result = _average_component_latencies(cases)

    assert result == {
        "database": 100.0,
        "guardrails": 8.0,
        "llm": 387.5,
        "retrieval": 35.0,
    }


def test_average_stage_call_counts_uses_api_counters() -> None:
    cases = [
        {"output": {"stage_call_counts": {"optimizer": 1, "sql_judgement": 1}}},
        {"output": {"stage_call_counts": {"optimizer": 1, "sql_judgement": 2}}},
        {"output": {"status": "runner_error"}},
    ]

    result = _average_stage_call_counts(cases)

    assert result == {
        "optimizer": 1.0,
        "sql_judgement": 1.5,
    }


def test_average_component_latencies_excludes_rule_based_optimizer() -> None:
    cases = [
        {
            "output": {
                "timings_ms": {
                    "optimizer": 25.0,
                    "sql_generation": 100.0,
                    "sql_judgement": 50.0,
                    "answer_synthesis": 30.0,
                },
                "stage_call_counts": {
                    "sql_generation": 1,
                    "sql_judgement": 1,
                    "answer_synthesis": 1,
                },
            }
        }
    ]

    result = _average_component_latencies(cases)

    assert result["llm"] == 180.0
