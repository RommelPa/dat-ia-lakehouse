"""Selección del backend de consultas de Dat-IA.

Esta capa traduce ``Settings`` en una implementación concreta sin acoplar el
resto de la aplicación a PostgreSQL o Databricks. Durante la migración,
PostgreSQL sigue siendo el backend predeterminado.
"""

from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings
from app.database.databricks import DatabricksExecutor, DatabricksSqlConfig
from app.database.postgres import create_postgres_engine


BackendName = Literal["postgres", "databricks"]


@dataclass(slots=True)
class QueryBackend:
    """Backend seleccionado y recurso asociado para ejecución SQL."""

    name: BackendName
    resource: Any


def create_query_backend(settings: Settings) -> QueryBackend:
    """Construye el backend configurado y valida sus parámetros requeridos."""
    if settings.query_backend == "postgres":
        if not settings.database_url:
            raise ValueError(
                "QUERY_BACKEND=postgres requiere DATABASE_URL."
            )

        return QueryBackend(
            name="postgres",
            resource=create_postgres_engine(settings.database_url),
        )

    if not settings.databricks_server_hostname:
        raise ValueError(
            "QUERY_BACKEND=databricks requiere DATABRICKS_SERVER_HOSTNAME."
        )

    if not settings.databricks_http_path:
        raise ValueError(
            "QUERY_BACKEND=databricks requiere DATABRICKS_HTTP_PATH."
        )

    config = DatabricksSqlConfig(
        server_hostname=settings.databricks_server_hostname,
        http_path=settings.databricks_http_path,
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
        auth_type=settings.databricks_auth_type,
    )

    return QueryBackend(
        name="databricks",
        resource=DatabricksExecutor(config),
    )
