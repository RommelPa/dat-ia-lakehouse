from datetime import datetime
from decimal import Decimal

from app.optimizer.query_optimizer import OptimizedQuery, QueryFilter
from app.validation.result_guardrail import check_groundedness, check_result


def _optimized_query(**overrides) -> OptimizedQuery:
    defaults = dict(
        original_question="Que transportista tiene el mayor promedio de retraso?",
        normalized_question="promedio de retraso por transportista",
        intent="aggregation",
        operation="average",
        metrics=["delay_days"],
        filters=[QueryFilter(field="state", operator="=", value="SP")],
        date_range=None,
        group_by=["carrier_name"],
        context=["logistica"],
        suggested_tables=["carriers"],
        optimizer="rule_based",
    )
    defaults.update(overrides)
    return OptimizedQuery(**defaults)


def test_check_result_ok_when_rows_present_and_under_limit() -> None:
    rows = [{"carrier_name": "DHL", "delay_days": 1.2}]

    result = check_result(rows, _optimized_query(), row_limit=200)

    assert result.ok is True
    assert result.warnings == []


def test_check_result_warns_when_rows_empty() -> None:
    result = check_result([], _optimized_query(), row_limit=200)

    assert result.ok is False
    assert "no devolvió filas" in result.warnings[0]


def test_check_result_warns_when_metric_all_null() -> None:
    rows = [
        {"carrier_name": "DHL", "delay_days": None},
        {"carrier_name": "FedEx", "delay_days": None},
    ]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is False
    assert "delay_days" in result.warnings[0]


def test_check_result_ignores_partial_nulls_in_metric() -> None:
    rows = [
        {"carrier_name": "DHL", "delay_days": None},
        {"carrier_name": "FedEx", "delay_days": 2.0},
    ]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is True


def test_check_result_matches_metric_by_substring_alias() -> None:
    """El generador no siempre alía la columna con el nombre exacto del
    metric (ej. `avg_resolution_time_hr` en vez de `resolution_time_hr`);
    la coincidencia por substring debe encontrarla igual.
    """
    rows = [
        {"carrier_name": "DHL", "avg_delay_days": None},
        {"carrier_name": "FedEx", "avg_delay_days": None},
    ]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is False
    assert "delay_days" in result.warnings[0]


def test_check_result_does_not_warn_when_no_column_resembles_metric() -> None:
    """Si ninguna columna se parece al nombre del metric, no hay evidencia
    de que el dato venga vacío -- puede ser solo una diferencia de nombre
    irreconocible. No advertir evita el falso positivo sistemático.
    """
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.97}]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is True


def test_check_result_warns_when_truncated_to_row_limit() -> None:
    rows = [{"carrier_name": f"carrier_{i}", "delay_days": 1.0} for i in range(3)]

    result = check_result(rows, _optimized_query(), row_limit=3)

    assert result.ok is False
    assert "truncó a 3 filas" in result.warnings[0]


def test_check_groundedness_ok_when_numbers_match_rows() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.97}]
    answer = "El transportista con mejor cumplimiento es DHL con 0.97."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_flags_unsupported_number() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.97}]
    answer = "El transportista con mejor cumplimiento es DHL con 452 pedidos."

    result = check_groundedness(answer, rows)

    assert result.ok is False
    assert "452" in result.unsupported_numbers


def test_check_groundedness_accepts_decimal_values_from_postgres() -> None:
    """Columnas NUMERIC via psycopg2/SQLAlchemy llegan como Decimal, no float."""
    rows = [{"carrier_name": "InterEstadual Cargo", "on_time_rate": Decimal("0.960")}]
    answer = "El transportista con mejor cumplimiento es InterEstadual Cargo, con 0.960."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_accepts_percentage_equivalent_of_ratio() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": Decimal("0.960")}]
    answer = "La tasa de cumplimiento de DHL fue 96,0 %."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_rejects_unrelated_percentage() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": Decimal("0.960")}]
    answer = "La tasa de cumplimiento de DHL fue 87,6 %."

    result = check_groundedness(answer, rows)

    assert result.ok is False
    assert result.unsupported_numbers == ["87,6 %"]


def test_check_groundedness_respects_rounding_tolerance() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.9701}]
    answer = "La tasa de cumplimiento es 0.97."

    result = check_groundedness(answer, rows, tolerance=0.01)

    assert result.ok is True


def test_check_groundedness_accepts_dot_as_thousands_separator() -> None:
    rows = [{"order_count": 99_441}]
    answer = "El número total de órdenes registradas es 99.441."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_accepts_comma_as_thousands_separator() -> None:
    rows = [{"order_count": 99_441}]
    answer = "The total number of orders is 99,441."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_keeps_leading_zero_three_decimals_fractional() -> None:
    rows = [{"on_time_rate": Decimal("0.960")}]
    answer = "La tasa fue 0.960."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_does_not_accept_wrong_thousands_interpretation() -> None:
    rows = [{"order_count": 99_442}]
    answer = "El número total de órdenes registradas es 99.441."

    result = check_groundedness(answer, rows)

    assert result.ok is False
    assert result.unsupported_numbers == ["99.441"]


def test_check_groundedness_accepts_year_from_datetime_value() -> None:
    rows = [
        {
            "month": datetime(2018, 1, 1),
            "order_count": 7069,
        }
    ]
    answer = "Enero 2018: 7069 órdenes."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_accepts_year_from_iso_date_string() -> None:
    rows = [
        {
            "month": "2018-01-01T00:00:00Z",
            "order_count": 7069,
        }
    ]
    answer = "Enero 2018: 7069 órdenes."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_accepts_numeric_string_values() -> None:
    rows = [{"month": "2018-01-01T00:00:00Z", "revenue": "924645.00"}]
    answer = "Enero 2018: 924645.00."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []
