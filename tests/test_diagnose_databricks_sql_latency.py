from __future__ import annotations

from app.core.config import Settings
from scripts import diagnose_databricks_sql_latency as diagnostic


class FakeCursor:
    def __init__(self) -> None:
        self.execute_calls = []
        self.fetch_calls = []
        self.closed = False

    def execute(self, sql_text: str) -> None:
        self.execute_calls.append(sql_text)

    def fetchmany(self, size: int):
        self.fetch_calls.append(size)
        return [(1,)]

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(
        query_backend="databricks",
        databricks_server_hostname="workspace.example.databricks.com",
        databricks_http_path="/sql/1.0/warehouses/example",
    )


def test_diagnose_connection_reuses_one_connection(monkeypatch) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    connect_calls = []

    def fake_connect(**kwargs):
        connect_calls.append(kwargs)
        return connection

    ticks = iter([
        1.0, 1.010,
        2.0, 2.020,
        3.0, 3.030,
        4.0, 4.040,
        5.0, 5.050,
        6.0, 6.060,
        7.0, 7.070,
    ])
    monkeypatch.setattr(
        diagnostic.time,
        "perf_counter",
        lambda: next(ticks),
    )

    report = diagnostic.diagnose_connection(
        connect=fake_connect,
        settings=_settings(),
        sql_text="SELECT 1 AS value",
        runs=2,
    )

    assert len(connect_calls) == 1
    assert cursor.execute_calls == [
        "SELECT 1 AS value",
        "SELECT 1 AS value",
    ]
    assert cursor.fetch_calls == [1, 1]
    assert cursor.closed is True
    assert connection.closed is True
    assert report["connect_ms"] == 10.0
    assert report["cursor_ms"] == 20.0
    assert report["executions"] == [
        {
            "run": 1,
            "execute_ms": 30.0,
            "fetch_ms": 40.0,
            "rows_fetched": 1,
        },
        {
            "run": 2,
            "execute_ms": 50.0,
            "fetch_ms": 60.0,
            "rows_fetched": 1,
        },
    ]
    assert report["close_ms"] == 70.0


def test_diagnose_connection_rejects_non_select() -> None:
    try:
        diagnostic.diagnose_connection(
            connect=lambda **kwargs: None,
            settings=_settings(),
            sql_text="DELETE FROM demo",
            runs=2,
        )
    except ValueError as exc:
        assert "SELECT" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError")


def test_diagnose_connection_rejects_invalid_runs() -> None:
    try:
        diagnostic.diagnose_connection(
            connect=lambda **kwargs: None,
            settings=_settings(),
            sql_text="SELECT 1",
            runs=0,
        )
    except ValueError as exc:
        assert "mayor que cero" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError")
