"""Configuración tipada de la aplicación.

Este módulo centraliza las variables de entorno sin cambiar todavía el
comportamiento de ``app.main``. La migración se hará de forma incremental para
mantener compatibilidad con los tests y despliegues actuales.

La clase ``Settings`` es la única fuente de configuración nueva que se irá
adoptando gradualmente durante el refactor estructural.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de Dat-IA cargada desde variables de entorno."""

    app_env: str = "test"
    app_version: str = "0.2.0"

    google_api_key: str | None = None
    database_url: str | None = None
    query_backend: Literal["postgres", "databricks"] = "postgres"

    databricks_server_hostname: str | None = None
    databricks_http_path: str | None = None
    databricks_catalog: str = "dat_ia"
    databricks_schema: str = "gold"
    databricks_auth_type: str = "databricks-oauth"

    model: str = "gemini-3.1-flash-lite-preview"
    embed_model: str = "gemini-embedding-2"

    chroma_path: str = "./chroma_db"
    chroma_host: str | None = None
    chroma_port: int = 8000

    use_cloudflare_llm: bool = False
    cloudflare_account_id: str = ""
    cloudflare_api_key: str = ""
    cloudflare_model: str = "@cf/qwen/qwen2.5-coder-32b-instruct"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cloudflare_base_url(self) -> str:
        """Endpoint OpenAI-compatible de Cloudflare Workers AI."""
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.cloudflare_account_id}/ai/v1"
        )

    @property
    def sql_generation_provider(self) -> str:
        """Proveedor activo para la generación SQL."""
        return "cloudflare" if self.use_cloudflare_llm else "google"

    @property
    def sql_generation_model(self) -> str:
        """Modelo activo para la generación SQL."""
        return self.cloudflare_model if self.use_cloudflare_llm else self.model
