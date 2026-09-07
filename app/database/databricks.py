"""Contrato de conexión al backend analítico Databricks.

Este módulo no abre conexiones ni agrega todavía el cliente de Databricks.
Define únicamente la configuración no sensible que necesitará el futuro
``DatabricksExecutor``. Las credenciales se resolverán fuera del código y nunca
se almacenarán en Git.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatabricksSqlConfig:
    """Destino lógico para consultas SQL contra Databricks."""

    server_hostname: str
    http_path: str
    catalog: str = "dat_ia"
    schema: str = "gold"

    def __post_init__(self) -> None:
        """Rechaza configuraciones incompletas antes de crear un cliente."""
        required = {
            "server_hostname": self.server_hostname,
            "http_path": self.http_path,
            "catalog": self.catalog,
            "schema": self.schema,
        }

        missing = [
            name
            for name, value in required.items()
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(
                "DatabricksSqlConfig requiere valores no vacíos para: "
                + ", ".join(missing)
            )

    @property
    def namespace(self) -> str:
        """Namespace ``catalog.schema`` usado por el backend analítico."""
        return f"{self.catalog}.{self.schema}"
