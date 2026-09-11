from app.database.databricks import DatabricksExecutor, DatabricksSqlConfig


class FakeCursor:
    def __init__(self, rows=None, description=None, error=None):
        self.rows = rows or []
        self.description = description or []
        self.error = error
        self.executed_sql = None
        self.closed = False
        self.fetchmany_size = None

    def execute(self, sql_text):
        self.executed_sql = sql_text
        if self.error:
            raise self.error

    def fetchmany(self, size):
        self.fetchmany_size = size
        return self.rows[:size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def build_executor(cursor):
    captured = {}
    connection = FakeConnection(cursor)

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return connection

    executor = DatabricksExecutor(
        DatabricksSqlConfig(
            server_hostname="workspace.example.databricks.com",
            http_path="/sql/1.0/warehouses/example",
        ),
        connect=fake_connect,
    )
    return executor, connection, captured


def test_execute_maps_rows_and_uses_gold_namespace():
    cursor = FakeCursor(
        rows=[(99_441, "delivered")],
        description=[("orders",), ("status",)],
    )
    executor, connection, captured = build_executor(cursor)

    result = executor.execute(
        "SELECT COUNT(*) AS orders, 'delivered' AS status",
        row_limit=10,
    )

    assert result == {"rows": [{"orders": 99_441, "status": "delivered"}]}
    assert captured == {
        "server_hostname": "workspace.example.databricks.com",
        "http_path": "/sql/1.0/warehouses/example",
        "auth_type": "databricks-oauth",
        "catalog": "dat_ia",
        "schema": "gold",
    }
    assert cursor.executed_sql == "SELECT COUNT(*) AS orders, 'delivered' AS status"
    assert cursor.fetchmany_size == 10
    assert cursor.closed is True
    assert connection.closed is False

    executor.close()

    assert connection.closed is True


def test_execute_rejects_non_select_without_connecting():
    calls = 0

    def fake_connect(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("No debe conectar para SQL rechazado")

    executor = DatabricksExecutor(
        DatabricksSqlConfig(
            server_hostname="workspace.example.databricks.com",
            http_path="/sql/1.0/warehouses/example",
        ),
        connect=fake_connect,
    )

    assert executor.execute("DELETE FROM olist_orders_dataset") == {
        "error": "Solo se permiten sentencias SELECT."
    }
    assert calls == 0


def test_execute_rejects_multiple_statements():
    cursor = FakeCursor()
    executor, _, _ = build_executor(cursor)

    assert executor.execute("SELECT 1; SELECT 2") == {
        "error": "Solo se permite una sentencia SQL por consulta."
    }


def test_execute_returns_connector_errors_and_closes_resources():
    cursor = FakeCursor(error=RuntimeError("warehouse unavailable"))
    executor, connection, _ = build_executor(cursor)

    assert executor.execute("SELECT 1") == {"error": "warehouse unavailable"}
    assert cursor.closed is True
    assert connection.closed is True


def test_execute_rejects_invalid_row_limit():
    cursor = FakeCursor()
    executor, _, _ = build_executor(cursor)

    assert executor.execute("SELECT 1", row_limit=0) == {
        "error": "row_limit debe ser mayor que cero."
    }


def test_execute_accepts_read_only_cte() -> None:
    cursor = FakeCursor(
        rows=[("books", 4.5)],
        description=[("category",), ("score",)],
    )
    executor, connection, _ = build_executor(cursor)

    sql = (
        "WITH ranked AS (SELECT 'books' AS category, 4.5 AS score) "
        "SELECT category, score FROM ranked"
    )
    result = executor.execute(sql)

    assert result == {"rows": [{"category": "books", "score": 4.5}]}
    assert cursor.executed_sql == sql
    assert cursor.closed is True
    assert connection.closed is False

    executor.close()

    assert connection.closed is True


def test_execute_records_databricks_substep_timings(monkeypatch) -> None:
    from app.observability import timing

    cursor = FakeCursor(
        rows=[(1,)],
        description=[("value",)],
    )
    executor, _, _ = build_executor(cursor)
    ticks = iter([
        1.0, 1.010,
        2.0, 2.020,
        3.0, 3.030,
        4.0, 4.040,
        5.0, 5.005,
        6.0, 6.006,
    ])
    monkeypatch.setattr(
        timing.time,
        "perf_counter",
        lambda: next(ticks),
    )
    timings = {}

    result = executor.execute(
        "SELECT 1 AS value",
        timings_ms=timings,
    )

    assert result == {"rows": [{"value": 1}]}
    assert timings == {
        "databricks_connect": 10.0,
        "databricks_cursor": 20.0,
        "databricks_execute": 30.0,
        "databricks_fetch": 40.0,
        "databricks_cursor_close": 5.0,
    }


def test_execute_reuses_connection_across_queries() -> None:
    first_cursor = FakeCursor(
        rows=[(1,)],
        description=[("value",)],
    )
    second_cursor = FakeCursor(
        rows=[(2,)],
        description=[("value",)],
    )
    cursors = iter([first_cursor, second_cursor])
    connect_calls = []

    class MultiCursorConnection:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return next(cursors)

        def close(self):
            self.closed = True

    connection = MultiCursorConnection()

    def fake_connect(**kwargs):
        connect_calls.append(kwargs)
        return connection

    executor = DatabricksExecutor(
        DatabricksSqlConfig(
            server_hostname="workspace.example.databricks.com",
            http_path="/sql/1.0/warehouses/example",
        ),
        connect=fake_connect,
    )

    first = executor.execute("SELECT 1 AS value")
    second = executor.execute("SELECT 2 AS value")

    assert first == {"rows": [{"value": 1}]}
    assert second == {"rows": [{"value": 2}]}
    assert len(connect_calls) == 1
    assert connection.closed is False
    assert first_cursor.closed is True
    assert second_cursor.closed is True

    executor.close()

    assert connection.closed is True


def test_execute_invalidates_connection_after_error() -> None:
    failing_cursor = FakeCursor(error=RuntimeError("session expired"))
    healthy_cursor = FakeCursor(
        rows=[(1,)],
        description=[("value",)],
    )
    connections = [
        FakeConnection(failing_cursor),
        FakeConnection(healthy_cursor),
    ]
    connect_calls = []

    def fake_connect(**kwargs):
        connection = connections[len(connect_calls)]
        connect_calls.append(kwargs)
        return connection

    executor = DatabricksExecutor(
        DatabricksSqlConfig(
            server_hostname="workspace.example.databricks.com",
            http_path="/sql/1.0/warehouses/example",
        ),
        connect=fake_connect,
    )

    failed = executor.execute("SELECT 1")
    recovered = executor.execute("SELECT 1")

    assert failed == {"error": "session expired"}
    assert recovered == {"rows": [{"value": 1}]}
    assert len(connect_calls) == 2
    assert connections[0].closed is True
    assert connections[1].closed is False

    executor.close()

    assert connections[1].closed is True
