from __future__ import annotations

from scripts.benchmark_query_backends import (
    DEFAULT_DATABRICKS_OVERRIDES_PATH,
    _is_reference_sql_case,
    _load_databricks_overrides,
    rows_have_parity,
)


def test_databricks_overrides_are_minimal_and_documented() -> None:
    overrides = _load_databricks_overrides(DEFAULT_DATABRICKS_OVERRIDES_PATH)

    assert set(overrides) == {"golden_017", "golden_022"}
    assert all(overrides[case_id]["reason"] for case_id in overrides)
    assert overrides["golden_017"]["expected_result"]["row_count"] == 5
    assert overrides["golden_022"]["expected_result"]["row_count"] == 5


def test_reference_sql_benchmark_skips_security_cases() -> None:
    analytical = {
        "reference_outputs": {
            "expected_status": "success",
            "reference_sql": "SELECT 1",
        }
    }
    blocked = {
        "reference_outputs": {
            "expected_status": "blocked",
            "reference_sql": "",
        }
    }

    assert _is_reference_sql_case(analytical) is True
    assert _is_reference_sql_case(blocked) is False


def test_rows_have_parity_requires_bidirectional_equivalence() -> None:
    left = [{"category": "books", "score": 4.46}]
    right = [{"label": "books", "avg": 4.46}]

    assert rows_have_parity(left, right, tolerance=0.01) is True
    assert rows_have_parity(left, [{"label": "books", "avg": 4.40}], tolerance=0.01) is False
