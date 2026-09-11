from __future__ import annotations
from types import SimpleNamespace
from uuid import UUID

from app.evaluation import (
    GOLDEN_SET_DATASET_NAME,
    GOLDEN_SET_VERSION,
    answer_contains_expected_facts,
    compare_result_facts,
    generated_sql_is_read_only,
    is_read_only_sql,
    load_golden_set,
    reported_source_tables_match_expected,
    response_status_matches_expected,
    result_facts_match_expected,
    sync_golden_set,
)

def test_canonical_golden_set_has_thirty_complete_cases() -> None:
    cases = load_golden_set()
    monthly_revenue = cases[-6]["reference_outputs"]["expected_result"]

    assert len(cases) == 35
    assert len({case["case_id"] for case in cases}) == 35
    assert cases[0]["inputs"]["question"].startswith("¿Cuántas órdenes")
    assert cases[-6]["metadata"]["result_type"] == "time_series"
    assert all(
        "comparison_mode"
        not in case["reference_outputs"]["expected_result"]
        for case in cases
    )
    assert monthly_revenue["row_count"] == 8
    assert len(monthly_revenue["rows"]) == 8
    assert GOLDEN_SET_VERSION == "2.1.0"
    assert all(
        case["metadata"]["dataset_version"] == GOLDEN_SET_VERSION
        and case["metadata"]["quality_status"]
        == "verified_against_official_postgresql"
        for case in cases
    )


def test_compare_result_facts_ignores_aliases_and_normalizes_numbers() -> None:
    expected = {
        "row_count": 1,
        "rows": [{"avg_review_score": 4.09}],
        "numeric_tolerance": 0.01,
    }

    assert (
        compare_result_facts(
            [{"average_review_score": "4.0872966183574879"}],
            expected,
        )
        is True
    )


def test_compare_result_facts_normalizes_uuid_and_allows_extra_columns() -> None:
    expected = {
        "row_count": 1,
        "rows": [
            {
                "seller_id": "4869f7a5dfa277a7dca6462dcf3b52b2",
                "merchandise_revenue_brl": 226987.93,
            }
        ],
        "numeric_tolerance": 0.01,
    }
    actual = [
        {
            "seller_id": "4869f7a5-dfa2-77a7-dca6-462dcf3b52b2",
            "total_earnings": "226987.93",
            "debug_label": "fila principal",
        }
    ]

    assert compare_result_facts(actual, expected) is True


def test_compare_result_facts_ignores_database_row_order() -> None:
    expected = {
        "row_count": 3,
        "rows": [
            {"payment_type": "credit_card", "payment_records": 10},
            {"payment_type": "boleto", "payment_records": 5},
            {"payment_type": "voucher", "payment_records": 2},
        ],
        "numeric_tolerance": 0.0,
    }
    actual = [
        {"payment_method": "voucher", "total": 2},
        {"payment_method": "credit_card", "total": 10},
        {"payment_method": "boleto", "total": 5},
    ]

    assert compare_result_facts(actual, expected) is True


def test_answer_contains_expected_facts_handles_formatted_numbers() -> None:
    outputs = {
        "answer": (
            "Los estados con más órdenes son SP (40,501), "
            "RJ (12,350), MG (11,354), RS (5,345) y PR (4,923)."
        )
    }
    reference = {
        "expected_result": {
            "row_count": 5,
            "rows": [
                {"customer_state": "SP", "delivered_orders": 40501},
                {"customer_state": "RJ", "delivered_orders": 12350},
                {"customer_state": "MG", "delivered_orders": 11354},
                {"customer_state": "RS", "delivered_orders": 5345},
                {"customer_state": "PR", "delivered_orders": 4923},
            ],
            "numeric_tolerance": 0.01,
        }
    }

    assert answer_contains_expected_facts(outputs, reference) is True


def test_refreshed_golden_set_requires_only_question_relevant_columns() -> None:
    cases = {case["case_id"]: case for case in load_golden_set()}

    expected_columns = {
        "golden_015": {"seller_id", "merchandise_revenue_brl"},
        "golden_022": {"category", "avg_review_score"},
        "golden_025": {"change_reason", "avg_price_change_brl"},
        "golden_029": {"seller_id", "delivered_items"},
    }
    omitted_columns = {
        "golden_015": ["seller_city", "seller_state"],
        "golden_022": ["reviews"],
        "golden_025": ["price_changes"],
        "golden_029": ["seller_city", "seller_state"],
    }

    for case_id, columns in expected_columns.items():
        expected_rows = cases[case_id]["reference_outputs"]["expected_result"][
            "rows"
        ]

        assert all(set(row) == columns for row in expected_rows)
        assert (
            cases[case_id]["metadata"]["omitted_optional_columns"]
            == omitted_columns[case_id]
        )


def test_deterministic_evaluators_score_api_output() -> None:
    outputs = {
        "status": "success",
        "sources": "orders, customers",
        "sql": "SELECT COUNT(*) AS total FROM orders;",
        "data": [{"total": 3}],
        "answer": "El total es 3.",
    }
    reference = {
        "expected_status": "success",
        "expected_sources": ["customers", "orders"],
        "expected_result": {
            "row_count": 1,
            "rows": [{"total": 3}],
            "numeric_tolerance": 0.0,
        },
    }

    assert response_status_matches_expected(outputs, reference) is True
    assert reported_source_tables_match_expected(outputs, reference) is True
    assert result_facts_match_expected(outputs, reference) is True
    assert answer_contains_expected_facts(outputs, reference) is True
    assert generated_sql_is_read_only(outputs) is True


def test_sql_read_only_rejects_mutations_and_accepts_ctes() -> None:
    assert is_read_only_sql("SELECT * FROM orders;") is True
    assert (
        is_read_only_sql(
            "WITH totals AS (SELECT COUNT(*) AS n FROM orders) SELECT * FROM totals;"
        )
        is True
    )
    assert is_read_only_sql("DROP TABLE orders;") is False
    assert (
        is_read_only_sql(
            "WITH removed AS (DELETE FROM orders RETURNING *) SELECT * FROM removed;"
        )
        is False
    )


class _FakeLangSmithClient:
    def __init__(self) -> None:
        self.dataset = SimpleNamespace(
            id=UUID("a3c5ab99-8ecf-4b2a-a29f-76cd5d611f44")
        )
        self.created_examples = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        assert dataset_name == GOLDEN_SET_DATASET_NAME
        return False

    def create_dataset(self, **kwargs):
        assert kwargs["dataset_name"] == GOLDEN_SET_DATASET_NAME
        return self.dataset

    def list_examples(self, *, dataset_id):
        assert dataset_id == self.dataset.id
        return []

    def create_examples(self, *, dataset_id, examples, max_concurrency):
        assert dataset_id == self.dataset.id
        assert max_concurrency == 1
        self.created_examples = list(examples)


def test_sync_uses_stable_ids_and_creates_all_examples() -> None:
    client = _FakeLangSmithClient()
    cases = load_golden_set()

    summary = sync_golden_set(client, cases=cases)

    assert summary["dataset_created"] is True
    assert summary["created"] == 35
    assert len(client.created_examples) == 35
    assert len({example["id"] for example in client.created_examples}) == 35
    assert all(
        isinstance(example["id"], UUID)
        for example in client.created_examples
    )


def test_answer_contains_expected_facts_accepts_percentage_equivalent_of_rate() -> None:
    outputs = {
        "answer": (
            "El transportista con la mayor tasa de puntualidad declarada es: "
            "InterEstadual Cargo: 96,0 %"
        )
    }
    reference = {
        "expected_result": {
            "row_count": 1,
            "rows": [
                {
                    "carrier": "InterEstadual Cargo",
                    "on_time_rate": 0.96,
                }
            ],
            "numeric_tolerance": 0.01,
        }
    }

    assert answer_contains_expected_facts(outputs, reference) is True


def test_answer_contains_expected_facts_accepts_spanish_month_labels() -> None:
    outputs = {
        "answer": (
            "Enero 2018: 7069 órdenes; "
            "Febrero 2018: 6555 órdenes."
        )
    }
    reference = {
        "expected_result": {
            "row_count": 2,
            "rows": [
                {"month": "2018-01", "order_count": 7069},
                {"month": "2018-02", "order_count": 6555},
            ],
            "numeric_tolerance": 0.01,
        }
    }

    assert answer_contains_expected_facts(outputs, reference) is True
