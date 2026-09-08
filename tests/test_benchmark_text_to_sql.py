from __future__ import annotations

from scripts.benchmark_query_backends import (
    DEFAULT_DATABRICKS_OVERRIDES_PATH,
    _load_databricks_overrides,
)
from scripts.benchmark_text_to_sql import (
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
