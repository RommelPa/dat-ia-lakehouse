"""Infraestructura PostgreSQL/Supabase para Dat-IA."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


POSTGRES_STATEMENT_TIMEOUT_MS = 60_000


def create_postgres_engine(database_url: str) -> Engine:
    """Crea un engine SQLAlchemy de solo conexión para PostgreSQL.

    La creación del engine no abre una conexión inmediatamente. El timeout de
    60 segundos limita consultas accidentales de larga duración y conserva el
    comportamiento histórico de ``app.db.connect_db.create_db_engine``.
    """
    return create_engine(
        database_url,
        connect_args={
            "options": (
                "-c statement_timeout="
                f"{POSTGRES_STATEMENT_TIMEOUT_MS}"
            )
        },
    )
