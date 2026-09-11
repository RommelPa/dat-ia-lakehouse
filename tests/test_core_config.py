from app.core.config import Settings


def test_settings_defaults(monkeypatch) -> None:
    env_names = (
        "APP_ENV",
        "APP_VERSION",
        "GOOGLE_API_KEY",
        "DATABASE_URL",
        "QUERY_BACKEND",
        "MODEL",
        "EMBED_MODEL",
        "CHROMA_PATH",
        "CHROMA_HOST",
        "CHROMA_PORT",
        "USE_CLOUDFLARE_LLM",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_MODEL",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.app_env == "test"
    assert settings.app_version == "0.2.0"
    assert settings.google_api_key is None
    assert settings.database_url is None
    assert settings.model == "gemini-3.1-flash-lite-preview"
    assert settings.embed_model == "gemini-embedding-2"
    assert settings.chroma_path == "./chroma_db"
    assert settings.chroma_host is None
    assert settings.chroma_port == 8000
    assert settings.use_cloudflare_llm is False
    assert settings.query_backend == "postgres"
    assert settings.sql_dialect == "postgres"
    assert settings.sql_generation_provider == "google"
    assert settings.sql_generation_model == settings.model


def test_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_VERSION", "0.3.0")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("QUERY_BACKEND", "databricks")
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "9000")
    monkeypatch.setenv("USE_CLOUDFLARE_LLM", "true")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-123")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "test-cloudflare-key")

    settings = Settings()

    assert settings.app_env == "development"
    assert settings.app_version == "0.3.0"
    assert settings.google_api_key == "test-google-key"
    assert settings.database_url == "postgresql://example"
    assert settings.query_backend == "databricks"
    assert settings.sql_dialect == "databricks"
    assert settings.chroma_host == "chroma"
    assert settings.chroma_port == 9000
    assert settings.use_cloudflare_llm is True
    assert settings.cloudflare_account_id == "account-123"
    assert settings.cloudflare_api_key == "test-cloudflare-key"
    assert settings.sql_generation_provider == "cloudflare"
    assert settings.sql_generation_model == settings.cloudflare_model
    assert settings.cloudflare_base_url.endswith("/account-123/ai/v1")
