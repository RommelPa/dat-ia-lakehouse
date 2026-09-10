from app import main as main_module
from app.main import RAGResponse, generate_validated_sql
from app.optimizer.query_optimizer import OptimizedQuery, QueryFilter
from app.validation.sql_judge import SqlVerdict


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


def _rag_response(sql: str, sources: str = "carriers", status: str = "success") -> RAGResponse:
    return RAGResponse(
        sql=sql,
        sources=sources,
        confidence_note="",
        status=status,
    )


class _FakeBuildRagResponseSequence:
    """Devuelve una RAGResponse distinta en cada llamada; registra el feedback recibido."""

    def __init__(self, responses: list[RAGResponse]) -> None:
        self.responses = responses
        self.captured_feedback: list[SqlVerdict | None] = []

    def __call__(
        self,
        question,
        ddl,
        optimized_query=None,
        memory_examples=None,
        feedback=None,
        table_policies="",
        business_rules_text="",
    ):
        _ = question, ddl, optimized_query, memory_examples, table_policies, business_rules_text
        self.captured_feedback.append(feedback)
        return self.responses[len(self.captured_feedback) - 1]


class _FakeJudgeSqlSequence:
    """Devuelve un SqlVerdict distinto en cada llamada; cuenta cuántas veces se invocó."""

    def __init__(self, verdicts: list[SqlVerdict]) -> None:
        self.verdicts = verdicts
        self.calls = 0

    def __call__(self, optimized_query, sql, llm, source_schema):
        _ = optimized_query, sql, llm, source_schema
        verdict = self.verdicts[self.calls]
        self.calls += 1
        return verdict


_APPROVED = SqlVerdict(
    issues=[], is_valid=True, answers_question=True, suggested_fix="", confidence=1.0,
)


def test_generate_validated_sql_returns_on_first_approved_attempt(monkeypatch) -> None:
    build_fake = _FakeBuildRagResponseSequence(
        [_rag_response("SELECT carrier_name FROM carriers LIMIT 1;")],
    )
    judge_fake = _FakeJudgeSqlSequence([_APPROVED])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta", "ddl", _optimized_query(), ["carriers"], judge_llm=object(), db=None,
    )

    assert attempts == 1
    assert verdict is not None and verdict.is_valid and verdict.answers_question
    # Sin ";" final: validate_sql solo recorta espacios/";" — no hay
    # reescritura de sqlglot.
    assert rag_response.sql == "SELECT carrier_name FROM carriers LIMIT 1"
    assert build_fake.captured_feedback == [None]
    assert judge_fake.calls == 1


def test_generate_validated_sql_retries_with_judge_feedback_and_succeeds(monkeypatch) -> None:
    rejected = SqlVerdict(
        issues=["Agrupa por dia en vez de por transportista."],
        is_valid=True,
        answers_question=False,
        suggested_fix="Agrupa por carrier_name.",
        confidence=0.7,
    )
    build_fake = _FakeBuildRagResponseSequence(
        [
            _rag_response("SELECT created_at, AVG(delay_days) FROM carriers GROUP BY created_at;"),
            _rag_response("SELECT carrier_name, AVG(delay_days) FROM carriers GROUP BY carrier_name;"),
        ],
    )
    judge_fake = _FakeJudgeSqlSequence([rejected, _APPROVED])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta", "ddl", _optimized_query(), ["carriers"], judge_llm=object(), db=None,
    )

    assert attempts == 2
    assert verdict.is_valid and verdict.answers_question
    # Sin ";" final: validate_sql solo recorta espacios/";", no reescribe
    # el SQL (no se le agrega ningún LIMIT).
    assert rag_response.sql == (
        "SELECT carrier_name, AVG(delay_days) FROM carriers GROUP BY carrier_name"
    )
    assert build_fake.captured_feedback == [None, rejected]
    assert judge_fake.calls == 2


def test_generate_validated_sql_retries_after_validator_rejection(monkeypatch) -> None:
    build_fake = _FakeBuildRagResponseSequence(
        [
            _rag_response("SELECT * FROM other_table;"),
            _rag_response("SELECT carrier_name FROM carriers;"),
        ],
    )
    judge_fake = _FakeJudgeSqlSequence([_APPROVED])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta", "ddl", _optimized_query(), ["carriers"], judge_llm=object(), db=None,
    )

    assert attempts == 2
    assert verdict.is_valid and verdict.answers_question
    # Sin ";" final: validate_sql solo recorta espacios/";", no reescribe
    # el SQL (no se le agrega ningún LIMIT).
    assert rag_response.sql == "SELECT carrier_name FROM carriers"
    assert judge_fake.calls == 1  # el intento 1 nunca llega al juez

    first_feedback = build_fake.captured_feedback[1]
    assert first_feedback is not None
    assert "other_table" in first_feedback.issues[0]


def test_generate_validated_sql_exhausts_attempts_without_approval(monkeypatch) -> None:
    rejected_1 = SqlVerdict(
        issues=["issue 1"], is_valid=False, answers_question=False,
        suggested_fix="", confidence=0.2,
    )
    rejected_2 = SqlVerdict(
        issues=["issue 2"], is_valid=False, answers_question=False,
        suggested_fix="", confidence=0.2,
    )
    build_fake = _FakeBuildRagResponseSequence(
        [
            _rag_response("SELECT carrier_name FROM carriers;"),
            _rag_response("SELECT carrier_name FROM carriers;"),
        ],
    )
    judge_fake = _FakeJudgeSqlSequence([rejected_1, rejected_2])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta", "ddl", _optimized_query(), ["carriers"], judge_llm=object(), db=None,
        max_attempts=2,
    )

    assert attempts == 2
    assert verdict is rejected_2
    assert not (verdict.is_valid and verdict.answers_question)
    assert judge_fake.calls == 2


def test_generate_validated_sql_returns_none_verdict_when_llm_does_not_know(monkeypatch) -> None:
    build_fake = _FakeBuildRagResponseSequence(
        [_rag_response("I do not know", sources="", status="unknown")],
    )
    judge_fake = _FakeJudgeSqlSequence([])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta", "ddl", _optimized_query(), ["carriers"], judge_llm=object(), db=None,
    )

    assert attempts == 1
    assert verdict is None
    assert rag_response.sources == ""
    assert judge_fake.calls == 0


def test_generate_validated_sql_applies_business_sql_normalization(monkeypatch) -> None:
    optimized = _optimized_query(
        original_question="¿Cuántas órdenes entregadas hubo por mes durante 2018?",
        normalized_question="órdenes entregadas por mes durante 2018",
        intent="temporal_trend",
        operation="count",
        metrics=["order_count"],
        filters=[QueryFilter(field="order_status", operator="=", value="delivered")],
        date_range={"start_date": "2018-01-01", "end_date": "2018-12-31"},
        group_by=["month"],
        suggested_tables=["olist_orders_dataset"],
    )
    build_fake = _FakeBuildRagResponseSequence(
        [
            _rag_response(
                "SELECT DATE_TRUNC('month', order_delivered_customer_date) AS month, "
                "COUNT(*) AS order_count FROM olist_orders_dataset "
                "WHERE order_status = 'delivered' "
                "AND order_delivered_customer_date >= '2018-01-01' "
                "GROUP BY 1;",
                sources="olist_orders_dataset",
            )
        ],
    )
    judge_fake = _FakeJudgeSqlSequence([_APPROVED])
    monkeypatch.setattr(main_module, "build_rag_response", build_fake)
    monkeypatch.setattr(main_module, "judge_sql", judge_fake)

    rag_response, verdict, attempts = generate_validated_sql(
        "pregunta",
        "ddl",
        optimized,
        ["olist_orders_dataset"],
        judge_llm=object(),
        db=None,
    )

    assert attempts == 1
    assert verdict is not None and verdict.is_valid
    assert "order_delivered_customer_date" not in rag_response.sql
    assert rag_response.sql.count("order_purchase_timestamp") == 2
