from types import SimpleNamespace

from app.core.config import Settings
from app.database import runtime as runtime_module
from app.database.backend import QueryBackend
from app.database.databricks import DatabricksExecutor, DatabricksSqlConfig
from app.database.runtime import QueryRuntime, create_query_runtime


class FakeMappings:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class FakePostgresResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return FakeMappings(self._rows)


class FakeSQLDatabase:
    dialect = "postgresql"

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def run_no_throw(self, sql, fetch=None):
        self.calls.append((sql, fetch))
        return self.result


def test_query_runtime_executes_postgres_with_legacy_contract() -> None:
    db = FakeSQLDatabase(
        FakePostgresResult(
            [
                {"value": 1},
                {"value": 2},
                {"value": 3},
            ]
        )
    )
    runtime = QueryRuntime(name="postgres", executor=db, validation_db=db)

    result = runtime.execute("SELECT value FROM demo;", row_limit=2)

    assert result == {"rows": [{"value": 1}, {"value": 2}]}
    assert db.calls == [("SELECT value FROM demo", "cursor")]
    assert runtime.dialect == "postgresql"
    assert runtime.sql_dialect == "postgres"


def test_query_runtime_rejects_non_select_before_postgres_execution() -> None:
    db = FakeSQLDatabase(FakePostgresResult([]))
    runtime = QueryRuntime(name="postgres", executor=db, validation_db=db)

    result = runtime.execute("DELETE FROM demo")

    assert result == {"error": "Solo se permiten sentencias SELECT."}
    assert db.calls == []


def test_query_runtime_delegates_to_databricks_executor(monkeypatch) -> None:
    config = DatabricksSqlConfig(
        server_hostname="workspace.example.databricks.com",
        http_path="/sql/1.0/warehouses/example",
    )
    executor = DatabricksExecutor(config, connect=lambda **kwargs: None)
    monkeypatch.setattr(
        executor,
        "execute",
        lambda sql_text, row_limit=200: {
            "rows": [{"backend": "databricks", "limit": row_limit}]
        },
    )
    runtime = QueryRuntime(name="databricks", executor=executor)

    result = runtime.execute("SELECT 1", row_limit=25)

    assert result == {"rows": [{"backend": "databricks", "limit": 25}]}
    assert runtime.validation_db is None
    assert runtime.dialect == "databricks"
    assert runtime.sql_dialect == "databricks"


def test_create_query_runtime_builds_postgres_sql_database(monkeypatch) -> None:
    fake_engine = SimpleNamespace()
    fake_db = FakeSQLDatabase()

    monkeypatch.setattr(
        runtime_module,
        "create_query_backend",
        lambda settings: QueryBackend(name="postgres", resource=fake_engine),
    )
    monkeypatch.setattr(
        runtime_module,
        "SQLDatabase",
        lambda engine, lazy_table_reflection=True: fake_db,
    )

    runtime = create_query_runtime(Settings(database_url="postgresql://example"))

    assert runtime.name == "postgres"
    assert runtime.executor is fake_db
    assert runtime.validation_db is fake_db


def test_create_query_runtime_builds_databricks_without_sql_database(monkeypatch) -> None:
    executor = DatabricksExecutor(
        DatabricksSqlConfig(
            server_hostname="workspace.example.databricks.com",
            http_path="/sql/1.0/warehouses/example",
        ),
        connect=lambda **kwargs: None,
    )

    monkeypatch.setattr(
        runtime_module,
        "create_query_backend",
        lambda settings: QueryBackend(name="databricks", resource=executor),
    )

    runtime = create_query_runtime(
        Settings(
            query_backend="databricks",
            databricks_server_hostname="workspace.example.databricks.com",
            databricks_http_path="/sql/1.0/warehouses/example",
        )
    )

    assert runtime.name == "databricks"
    assert runtime.executor is executor
    assert runtime.validation_db is None


def test_query_runtime_accepts_postgres_read_only_cte() -> None:
    db = FakeSQLDatabase(
        FakePostgresResult([{"value": 1}])
    )
    runtime = QueryRuntime(name="postgres", executor=db, validation_db=db)

    sql = "WITH demo AS (SELECT 1 AS value) SELECT value FROM demo"
    result = runtime.execute(sql)

    assert result == {"rows": [{"value": 1}]}
    assert db.calls == [(sql, "cursor")]
