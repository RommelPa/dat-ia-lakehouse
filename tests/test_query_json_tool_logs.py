import asyncio
from types import SimpleNamespace

from app import main as main_module


class _DummyCollection:
    def __init__(self) -> None:
        self._collection = SimpleNamespace(count=lambda: 1)


def test_query_json_returns_tool_logs(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "text_collection", _DummyCollection())
    monkeypatch.setattr(main_module, "optimizer_llm", object())

    def fake_optimize_query(
        question,
        *,
        llm=None,
        use_llm=True,
    ):
        _ = llm
        assert use_llm is True
        return SimpleNamespace(
            normalized_question=question,
            suggested_tables=["ventas"],
            original_question=question,
            intent="aggregation",
            operation="sum",
            metrics=["revenue"],
            filters=[],
            date_range=None,
            group_by=[],
            context=[],
            to_dict=lambda: {"normalized_question": question},
        )

    monkeypatch.setattr(main_module, "optimize_query", fake_optimize_query)

    def fake_retrieve_ddl_context(
        collection,
        query,
        suggested_tables=None,
        distance_threshold=0.7,
        tool_logs=None,
    ):
        _ = (collection, query, suggested_tables, distance_threshold)
        return SimpleNamespace(
            ddl="create table ventas(id int);",
            tabla=["ventas"],
            descripcion=["ventas"],
            distance=[0.1],
            politicas=[[]],
        )

    monkeypatch.setattr(main_module, "retrieve_ddl_context", fake_retrieve_ddl_context)

    def fake_build_rag_response(
        question,
        ddl,
        optimized_query=None,
        memory_examples=None,
        tool_logs=None,
        table_policies="",
        business_rules_text="",
    ):
        _ = (question, ddl, optimized_query, memory_examples, table_policies, business_rules_text)
        return main_module.RAGResponse(
            sql="SELECT 1 AS result;",
            sources="ventas",
            confidence_note="ok",
            status="success",
        )

    monkeypatch.setattr(main_module, "build_rag_response", fake_build_rag_response)
    monkeypatch.setattr(main_module, "_save_query_memory_v2", lambda *args, **kwargs: None)

    response = asyncio.run(
        main_module.query_json(
            main_module.QueryRequest(question="¿Qué ventas hubo?"),
        )
    )

    assert response.status == "success"
    assert response.tool_logs is not None
    assert any(log.get("name") == "optimize_query" for log in response.tool_logs)
