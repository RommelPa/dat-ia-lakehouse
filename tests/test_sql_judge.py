from app.optimizer.query_optimizer import OptimizedQuery, QueryFilter
from app.validation.sql_judge import (
    SqlVerdict,
    deterministic_business_verdict,
    judge_sql,
    normalize_business_sql,
)


class _BoundFakeJudgeLlm:
    """Simula el runnable devuelto por llm.with_structured_output(schema)."""

    def __init__(self, payload: dict, schema, captured_prompts: list[str]) -> None:
        self.payload = payload
        self.schema = schema
        self.captured_prompts = captured_prompts

    def invoke(self, prompt: str):
        self.captured_prompts.append(prompt)
        return self.schema(**self.payload)


class FakeJudgeLlm:
    """Simula ChatGoogleGenerativeAI() antes de aplicar with_structured_output."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.captured_prompts: list[str] = []

    def with_structured_output(self, schema):
        return _BoundFakeJudgeLlm(self.payload, schema, self.captured_prompts)


def _optimized_query(**overrides) -> OptimizedQuery:
    defaults = dict(
        original_question="Que transportista tiene el mayor promedio de retraso?",
        normalized_question="promedio de retraso por transportista",
        intent="aggregation",
        operation="average",
        metrics=["delay_days"],
        filters=[QueryFilter(field="state", operator="=", value="SP")],
        date_range={"start_date": "2018-01-01", "end_date": "2018-12-31"},
        group_by=["carrier_name"],
        context=["logistica"],
        suggested_tables=["carriers"],
        optimizer="rule_based",
    )
    defaults.update(overrides)
    return OptimizedQuery(**defaults)


def _delivered_monthly_query(question: str) -> OptimizedQuery:
    return _optimized_query(
        original_question=question,
        normalized_question="órdenes entregadas por mes durante 2018",
        intent="temporal_trend",
        operation="count",
        metrics=["order_count"],
        filters=[QueryFilter(field="order_status", operator="=", value="delivered")],
        date_range={"start_date": "2018-01-01", "end_date": "2018-12-31"},
        group_by=["month"],
        suggested_tables=["olist_orders_dataset"],
    )


def test_judge_sql_passes_llm_payload_through_to_verdict() -> None:
    payload = {
        "issues": [],
        "is_valid": True,
        "answers_question": True,
        "suggested_fix": "",
        "confidence": 0.95,
    }
    llm = FakeJudgeLlm(payload)

    verdict = judge_sql(
        _optimized_query(),
        "SELECT carrier_name, AVG(delay_days) FROM carriers GROUP BY carrier_name;",
        llm,
    )

    assert isinstance(verdict, SqlVerdict)
    assert verdict.is_valid is True
    assert verdict.answers_question is True
    assert verdict.confidence == 0.95


def test_judge_sql_can_mark_answers_question_false_independent_of_is_valid() -> None:
    payload = {
        "issues": ["Agrupa por dia (created_at::date) en vez de por mes."],
        "is_valid": True,
        "answers_question": False,
        "suggested_fix": "Cambia GROUP BY created_at::date por GROUP BY DATE_TRUNC('month', created_at).",
        "confidence": 0.8,
    }
    llm = FakeJudgeLlm(payload)

    verdict = judge_sql(
        _optimized_query(operation="temporal_trend", group_by=["month"]),
        "SELECT created_at::date, COUNT(*) FROM orders GROUP BY created_at::date;",
        llm,
    )

    assert verdict.is_valid is True
    assert verdict.answers_question is False
    assert verdict.issues


def test_judge_sql_includes_optimized_query_fields_in_prompt() -> None:
    payload = {
        "issues": [],
        "is_valid": True,
        "answers_question": True,
        "suggested_fix": "",
        "confidence": 1.0,
    }
    llm = FakeJudgeLlm(payload)
    sql = "SELECT carrier_name, AVG(delay_days) FROM carriers GROUP BY carrier_name;"

    judge_sql(_optimized_query(), sql, llm)

    prompt = llm.captured_prompts[0]
    assert "average" in prompt
    assert "delay_days" in prompt
    assert "carrier_name" in prompt
    assert "2018-01-01" in prompt
    assert sql in prompt
    assert "promedio de retraso por transportista" in prompt


def test_judge_sql_does_not_receive_ddl_or_memory_examples() -> None:
    import inspect

    signature = inspect.signature(judge_sql)

    assert "ddl" not in signature.parameters
    assert "memory_examples" not in signature.parameters


def test_delivered_order_monthly_query_rejects_delivery_timestamp() -> None:
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes entregadas hubo por mes durante 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', order_delivered_customer_date) AS month, "
        "COUNT(*) AS order_count FROM olist_orders_dataset "
        "WHERE order_status = 'delivered' "
        "AND order_delivered_customer_date >= '2018-01-01' "
        "GROUP BY 1"
    )

    verdict = deterministic_business_verdict(optimized, sql)

    assert verdict is not None
    assert verdict.is_valid is False
    assert verdict.answers_question is False
    assert "order_purchase_timestamp" in verdict.suggested_fix


def test_delivered_order_monthly_query_accepts_purchase_timestamp() -> None:
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes entregadas hubo por mes durante 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', order_purchase_timestamp) AS month, "
        "COUNT(*) AS order_count FROM olist_orders_dataset "
        "WHERE order_status = 'delivered' "
        "AND order_purchase_timestamp >= '2018-01-01' "
        "GROUP BY 1"
    )

    assert deterministic_business_verdict(optimized, sql) is None


def test_normalize_business_sql_rewrites_delivery_timestamp_for_period() -> None:
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes entregadas hubo por mes durante 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', o.order_delivered_customer_date) AS month, "
        "COUNT(*) AS order_count FROM olist_orders_dataset o "
        "WHERE o.order_status = 'delivered' "
        "AND o.order_delivered_customer_date >= '2018-01-01' "
        "AND o.order_delivered_customer_date <= '2018-12-31' GROUP BY 1"
    )

    normalized = normalize_business_sql(optimized, sql)

    assert "order_delivered_customer_date" not in normalized
    assert normalized.count("order_purchase_timestamp") == 3


def test_normalize_business_sql_preserves_explicit_delivery_date_query() -> None:
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes se entregaron por mes según fecha de entrega en 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', order_delivered_customer_date) AS month, "
        "COUNT(*) FROM olist_orders_dataset "
        "WHERE order_status = 'delivered' GROUP BY 1"
    )

    assert normalize_business_sql(optimized, sql) == sql


def test_explicit_delivery_date_question_allows_delivery_timestamp() -> None:
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes se entregaron por mes según fecha de entrega en 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', order_delivered_customer_date) AS month, "
        "COUNT(*) AS order_count FROM olist_orders_dataset "
        "WHERE order_status = 'delivered' "
        "GROUP BY 1"
    )

    assert deterministic_business_verdict(optimized, sql) is None


def test_judge_prompt_preserves_purchase_timestamp_business_invariant() -> None:
    payload = {
        "issues": [],
        "is_valid": True,
        "answers_question": True,
        "suggested_fix": "",
        "confidence": 1.0,
    }
    llm = FakeJudgeLlm(payload)
    optimized = _delivered_monthly_query(
        "¿Cuántas órdenes entregadas hubo por mes durante 2018?"
    )
    sql = (
        "SELECT DATE_TRUNC('month', order_purchase_timestamp) AS month, "
        "COUNT(*) AS order_count FROM olist_orders_dataset "
        "WHERE order_status = 'delivered' GROUP BY 1"
    )

    judge_sql(optimized, sql, llm)

    prompt = llm.captured_prompts[0]
    assert "Regla determinística autorizada" in prompt
    assert "order_purchase_timestamp" in prompt
    assert "No rechaces el SQL por usar la fecha de compra" in prompt


def test_judge_sql_includes_target_dialect_in_prompt() -> None:
    payload = {
        "issues": [],
        "is_valid": True,
        "answers_question": True,
        "suggested_fix": "",
        "confidence": 1.0,
    }
    llm = FakeJudgeLlm(payload)

    judge_sql(
        _optimized_query(),
        "SELECT carrier_name FROM carriers",
        llm,
        dialect="databricks",
    )

    prompt = llm.captured_prompts[0]
    assert "dialect: databricks" in prompt
