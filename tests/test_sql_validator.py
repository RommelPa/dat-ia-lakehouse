from langchain_community.utilities import SQLDatabase

from app.validation.sql_validator import validate_sql


def _sqlite_db_with_carriers() -> SQLDatabase:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    db.run("CREATE TABLE carriers (carrier_name TEXT, on_time_rate REAL);")
    return db


def test_validate_sql_accepts_query_with_allowed_tables() -> None:
    sql = (
        "SELECT carrier_name FROM carriers "
        "ORDER BY on_time_rate DESC LIMIT 1;"
    )

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert result.stage == "ok"
    assert result.error == ""
    assert "LIMIT 1" in result.sql


def test_validate_sql_accepts_join_with_multiple_allowed_tables() -> None:
    sql = (
        "SELECT o.order_id, oi.price "
        "FROM olist_orders_dataset o "
        "JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id;"
    )

    result = validate_sql(
        sql,
        allowed_tables=[
            "olist_orders_dataset",
            "olist_order_items_dataset",
        ],
    )

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_accepts_query_without_tables() -> None:
    result = validate_sql(
        "SELECT 1 AS prototype_result;",
        allowed_tables=[],
    )

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_rejects_invalid_syntax() -> None:
    result = validate_sql("SELECT FROM WHERE;", allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "syntax"
    assert result.error


def test_validate_sql_rejects_empty_sql() -> None:
    result = validate_sql("   ", allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "syntax"


def test_validate_sql_rejects_table_outside_retrieved_schema() -> None:
    sql = "SELECT * FROM customer_support_tickets;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "tables"
    assert "customer_support_tickets" in result.error


def test_validate_sql_does_not_flag_cte_alias_as_unknown_table() -> None:
    sql = """
    WITH ranked AS (
        SELECT carrier_name, on_time_rate FROM carriers
    )
    SELECT * FROM ranked;
    """

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_dry_run_passes_with_valid_columns() -> None:
    db = _sqlite_db_with_carriers()
    sql = "SELECT carrier_name FROM carriers ORDER BY on_time_rate DESC;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=db)

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_dry_run_rejects_unknown_column() -> None:
    db = _sqlite_db_with_carriers()
    sql = "SELECT on_time_percentage FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=db)

    assert result.is_valid is False
    assert result.stage == "dry_run"
    assert result.error


def test_validate_sql_skips_dry_run_when_db_is_none() -> None:
    sql = "SELECT on_time_percentage FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=None)

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_rejects_stacked_statements() -> None:
    sql = "SELECT * FROM carriers; DROP TABLE carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "statement"
    assert result.error == "Solo se permite una sentencia SQL por consulta."


def test_validate_sql_rejects_non_select_statement() -> None:
    sql = "DROP TABLE carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "statement"
    assert result.error == "Solo se permiten sentencias SELECT."


def test_validate_sql_does_not_add_or_change_limit() -> None:
    sql = "SELECT carrier_name FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert "LIMIT" not in result.sql
    assert result.sql == "SELECT carrier_name FROM carriers"


def test_validate_sql_preserves_sql_formatting_verbatim() -> None:
    """validate_sql ya no reserializa con sqlglot: el SQL "descuidado" del
    LLM (minúsculas, cast abreviado, alias implícito) debe pasar intacto.
    """
    sql = "select price::text as total from olist_order_items_dataset;"

    result = validate_sql(sql, allowed_tables=["olist_order_items_dataset"])

    assert result.is_valid is True
    assert result.sql == "select price::text as total from olist_order_items_dataset"


def test_validate_sql_uses_explicit_databricks_dialect() -> None:
    sql = "SELECT date_trunc('month', order_purchase_timestamp) AS month FROM orders;"

    result = validate_sql(
        sql,
        allowed_tables=["orders"],
        dialect="databricks",
    )

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_rejects_postgres_distinct_on_for_databricks() -> None:
    sql = "SELECT DISTINCT ON (customer_id) customer_id FROM customers;"

    result = validate_sql(
        sql,
        allowed_tables=["customers"],
        dialect="databricks",
    )

    assert result.is_valid is False
    assert result.stage == "dialect"
    assert "DISTINCT ON" in result.error


def test_validate_sql_rejects_postgres_generate_series_for_databricks() -> None:
    result = validate_sql(
        "SELECT * FROM generate_series(1, 10);",
        allowed_tables=[],
        dialect="databricks",
    )

    assert result.is_valid is False
    assert result.stage == "dialect"
    assert "generate_series" in result.error


def test_validate_sql_rejects_databricks_qualify_for_postgres() -> None:
    sql = (
        "SELECT customer_id, ROW_NUMBER() OVER (ORDER BY customer_id) AS rn "
        "FROM customers QUALIFY rn = 1;"
    )

    result = validate_sql(
        sql,
        allowed_tables=["customers"],
        dialect="postgres",
    )

    assert result.is_valid is False
    assert result.stage == "dialect"
    assert "QUALIFY" in result.error


def test_validate_sql_rejects_databricks_collect_list_for_postgres() -> None:
    sql = "SELECT collect_list(customer_id) FROM customers;"

    result = validate_sql(
        sql,
        allowed_tables=["customers"],
        dialect="postgres",
    )

    assert result.is_valid is False
    assert result.stage == "dialect"
    assert "collect_list" in result.error


def test_validate_sql_dialect_guard_ignores_literals_and_comments() -> None:
    sql = (
        "SELECT 'generate_series(1, 10)' AS note "
        "FROM customers -- DISTINCT ON (customer_id)"
    )

    result = validate_sql(
        sql,
        allowed_tables=["customers"],
        dialect="databricks",
    )

    assert result.is_valid is True
    assert result.stage == "ok"
