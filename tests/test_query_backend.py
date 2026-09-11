from app.core.config import Settings
from app.database.backend import create_query_backend
from app.database.databricks import DatabricksExecutor


def test_create_query_backend_defaults_to_postgres(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    settings = Settings()

    backend = create_query_backend(settings)

    assert backend.name == "postgres"
    assert backend.resource.url.drivername.startswith("postgresql")


def test_create_query_backend_builds_databricks(monkeypatch) -> None:
    monkeypatch.setenv("QUERY_BACKEND", "databricks")
    monkeypatch.setenv(
        "DATABRICKS_SERVER_HOSTNAME",
        "workspace.example.databricks.com",
    )
    monkeypatch.setenv(
        "DATABRICKS_HTTP_PATH",
        "/sql/1.0/warehouses/example",
    )

    backend = create_query_backend(Settings())

    assert backend.name == "databricks"
    assert isinstance(backend.resource, DatabricksExecutor)
    assert backend.resource.config.namespace == "dat_ia.gold"
    assert backend.resource.config.auth_type == "databricks-oauth"


def test_create_query_backend_requires_database_url(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("QUERY_BACKEND", "postgres")

    try:
        create_query_backend(Settings())
    except ValueError as exc:
        assert "DATABASE_URL" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError sin DATABASE_URL")


def test_create_query_backend_requires_databricks_host(monkeypatch) -> None:
    monkeypatch.setenv("QUERY_BACKEND", "databricks")
    monkeypatch.delenv("DATABRICKS_SERVER_HOSTNAME", raising=False)
    monkeypatch.setenv(
        "DATABRICKS_HTTP_PATH",
        "/sql/1.0/warehouses/example",
    )

    try:
        create_query_backend(Settings())
    except ValueError as exc:
        assert "DATABRICKS_SERVER_HOSTNAME" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError sin hostname")


def test_create_query_backend_requires_databricks_http_path(monkeypatch) -> None:
    monkeypatch.setenv("QUERY_BACKEND", "databricks")
    monkeypatch.setenv(
        "DATABRICKS_SERVER_HOSTNAME",
        "workspace.example.databricks.com",
    )
    monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)

    try:
        create_query_backend(Settings())
    except ValueError as exc:
        assert "DATABRICKS_HTTP_PATH" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError sin HTTP path")
