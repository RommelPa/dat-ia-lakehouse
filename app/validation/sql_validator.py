"""Validación determinística del SQL generado, previa a su ejecución.

Compone capas en orden barato → caro: forma de la sentencia (solo SELECT,
sin apilar), parsear a un AST, comparar las tablas citadas contra el esquema
recuperado, y un dry-run con `EXPLAIN` contra la base de datos real (sin leer
filas) para validar columnas y tipos. No sustituye al juez LLM: solo captura
los niveles de error más baratos de detectar.

No acota el `LIMIT` ni reescribe el SQL de ninguna otra forma: el AST solo se
usa para verificar, nunca para reserializar. El tope de filas que sí existe
vive en `execute_sql` (`app/main.py`), que trunca en Python las filas ya
traídas de Postgres — es un tope de respuesta, no de ejecución en la BD.

Las guardas de solo-SELECT y anti-stacking vivían antes dentro de
`execute_sql` (`app/main.py`), donde se aplicaban justo antes de ejecutar
—después de gastar el juez LLM y el bucle de reintento en un SQL que iba a
ser rechazado de todos modos. Aquí son el primer filtro, antes de cualquier
llamada cara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlglot
from langchain_community.utilities import SQLDatabase
from sqlglot import exp

SqlValidationStage = Literal["syntax", "statement", "tables", "dry_run", "ok"]

DEFAULT_ROW_LIMIT = 200


@dataclass(frozen=True)
class SqlValidation:
    """Resultado de validar un SQL generado antes de ejecutarlo.

    `sql` trae el SQL del LLM, recortado únicamente de espacios y `;` final
    (`.strip().rstrip(";")`), cuando `is_valid=True`; queda vacío en
    cualquier rechazo. `validate_sql` nunca reescribe ni reserializa el SQL.
    """

    is_valid: bool
    stage: SqlValidationStage
    error: str
    sql: str = ""


def dry_run_explain(db: SQLDatabase, sql: str) -> str | None:
    """Corre `EXPLAIN` sobre el SQL sin ejecutarlo.

    Reutiliza `db.run_no_throw`, el mismo patrón que ya usa `execute_sql`
    (`app/main.py`): devuelve un `str` con el error en vez de lanzar una
    excepción. `EXPLAIN` (nunca `EXPLAIN ANALYZE`) construye el plan de
    ejecución sin leer una sola fila, lo que exige resolver tablas, columnas
    y tipos igual que lo haría la ejecución real.

    Returns:
        `None` si el plan se construyó sin error, o el mensaje de error si
        `EXPLAIN` falló (columna/tabla inexistente, tipo incompatible, etc.).
    """
    result = db.run_no_throw(f"EXPLAIN {sql}", fetch="cursor")
    return result if isinstance(result, str) else None


def validate_sql(
    sql: str,
    allowed_tables: list[str],
    db: SQLDatabase | None = None,
    dialect: Literal["postgres", "databricks"] = "postgres",
) -> SqlValidation:
    """Valida forma, sintaxis y tablas citadas, y hace dry-run.

    Args:
        sql: SQL generado por el LLM, aún sin ejecutar.
        allowed_tables: nombres de tabla recuperados para esta pregunta
            (ver `retrieve_ddl_context` en `app/main.py`).
        db: conexión para el dry-run con `EXPLAIN`. Si es `None` (por
            ejemplo, `DATABASE_URL` no configurada), esa etapa se omite sin
            marcarse como error.
        dialect: dialecto explícito usado por SQLGlot al parsear el SQL.

    Returns:
        `SqlValidation` con `is_valid=True`, `stage="ok"` y el SQL original
        (recortado) en `sql`, si pasa todas las etapas aplicables.
    """
    stripped = sql.strip().rstrip(";")

    if not stripped:
        return SqlValidation(is_valid=False, stage="syntax", error="El SQL está vacío.")

    if ";" in stripped:
        return SqlValidation(
            is_valid=False,
            stage="statement",
            error="Solo se permite una sentencia SQL por consulta.",
        )

    try:
        tree = sqlglot.parse_one(stripped, read=dialect)
    except sqlglot.errors.ParseError as exc:
        return SqlValidation(is_valid=False, stage="syntax", error=str(exc))

    if not isinstance(tree, exp.Select):
        return SqlValidation(
            is_valid=False,
            stage="statement",
            error="Solo se permiten sentencias SELECT.",
        )

    cte_aliases = {cte.alias for cte in tree.find_all(exp.CTE)}
    cited_tables = {table.name for table in tree.find_all(exp.Table)} - cte_aliases
    unknown_tables = cited_tables - set(allowed_tables)

    if unknown_tables:
        return SqlValidation(
            is_valid=False,
            stage="tables",
            error=f"Tablas no reconocidas en el esquema recuperado: {sorted(unknown_tables)}",
        )

    if db is not None:
        dry_run_error = dry_run_explain(db, stripped)
        if dry_run_error is not None:
            return SqlValidation(is_valid=False, stage="dry_run", error=dry_run_error)

    return SqlValidation(is_valid=True, stage="ok", error="", sql=stripped)
