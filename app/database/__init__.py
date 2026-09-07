"""Acceso a los backends de datos de Dat-IA.

Este paquete concentra las implementaciones específicas de cada motor. La
aplicación mantiene temporalmente ``app.db`` como capa de compatibilidad para
no romper imports existentes durante el refactor.
"""

from app.database.postgres import create_postgres_engine

__all__ = ["create_postgres_engine"]
