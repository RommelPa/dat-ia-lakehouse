import pytest

from app.database.databricks import DatabricksSqlConfig


def test_databricks_sql_config_builds_namespace() -> None:
    config = DatabricksSqlConfig(
        server_hostname="workspace.example.databricks.com",
        http_path="/sql/1.0/warehouses/example",
    )

    assert config.catalog == "dat_ia"
    assert config.schema == "gold"
    assert config.namespace == "dat_ia.gold"


def test_databricks_sql_config_rejects_empty_required_values() -> None:
    with pytest.raises(ValueError, match="server_hostname"):
        DatabricksSqlConfig(
            server_hostname="",
            http_path="/sql/1.0/warehouses/example",
        )
