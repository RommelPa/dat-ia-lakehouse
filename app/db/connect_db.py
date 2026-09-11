"""Compatibilidad temporal para la conexión relacional de Dat-IA.

La implementación real vive en ``app.database.postgres``. Este módulo se
mantiene durante el refactor para no romper imports existentes en ``app.main``
y en scripts operativos.
"""

from sqlalchemy.engine import Engine

from app.database.postgres import create_postgres_engine


def create_db_engine(database_url: str) -> Engine:
    """Crea el engine PostgreSQL conservando la API histórica."""
    return create_postgres_engine(database_url)
