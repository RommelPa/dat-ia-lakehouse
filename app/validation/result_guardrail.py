"""Guardrail ligero sobre el resultado ejecutado y la redacción final.

El SQL ya pasó el validador determinístico y el juez (Clases 2-3) antes de
ejecutarse. Lo que queda sin cubrir ocurre después: `execute_sql` puede
truncar filas en silencio o devolver una métrica en NULL, y el LLM que
redacta la respuesta en lenguaje natural puede inventar o redondear
números que no están en las filas.

Deliberadamente NO es un segundo juez semántico: eso duplicaría el trabajo
de `sql_judge` y añadiría otra llamada cara antes de responder. Son
chequeos deterministicos sobre las filas (`check_result`) más una
verificación de groundedness barata sobre el texto (`check_groundedness`),
sin LLM en la primera pasada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.optimizer.query_optimizer import OptimizedQuery
from app.validation.sql_validator import DEFAULT_ROW_LIMIT


_PERCENTAGE_HINTS = {
    "pct",
    "percent",
    "percentage",
    "porcentaje",
    "rate",
    "ratio",
    "tasa",
}


@dataclass(frozen=True)
class ResultCheck:
    """Señales de alerta detectadas mirando solo las filas, sin LLM."""

    ok: bool
    warnings: list[str]


@dataclass(frozen=True)
class GroundednessCheck:
    """Números del texto redactado que no aparecen entre los valores de las filas."""

    ok: bool
    unsupported_numbers: list[str]


def _looks_like_percentage_key(key: str) -> bool:
    """Indica si el nombre de una columna representa una tasa o porcentaje."""
    tokens = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", key.casefold())
    return bool(set(tokens) & _PERCENTAGE_HINTS)


def _parse_answer_number(raw: str) -> float:
    """Convierte números con separador decimal español o anglosajón."""
    normalized = raw.strip()

    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        decimal_part = normalized.rsplit(",", 1)[1]
        if normalized.startswith("0,") or len(decimal_part) <= 2:
            normalized = normalized.replace(",", ".")
        else:
            normalized = normalized.replace(",", "")

    return float(normalized)


def _parse_answer_number_candidates(raw: str) -> set[float]:
    """Devuelve interpretaciones plausibles de un número redactado.

    Un único separador seguido por tres dígitos es ambiguo en texto humano:
    ``99.441`` puede representar 99.441 o 99 441 según el locale. En vez de
    imponer una convención global, groundedness prueba ambas interpretaciones
    y acepta únicamente la que esté respaldada por las filas ejecutadas.

    Los valores que empiezan por cero, como ``0.960``, no generan una
    interpretación de miles porque representan de forma natural una fracción.
    """
    normalized = raw.strip()
    candidates = {_parse_answer_number(normalized)}

    separators = [separator for separator in (".", ",") if separator in normalized]
    if len(separators) != 1:
        return candidates

    separator = separators[0]
    if normalized.count(separator) != 1:
        return candidates

    integer_part, trailing_part = normalized.split(separator, 1)
    if (
        integer_part != "0"
        and integer_part.isdigit()
        and len(trailing_part) == 3
        and trailing_part.isdigit()
    ):
        candidates.add(float(integer_part + trailing_part))

    return candidates


def _numbers_match(candidate: float, value: float, tolerance: float) -> bool:
    """Compara números sin permitir tolerancia relativa sobre conteos enteros.

    Los conteos deben coincidir exactamente: una tolerancia relativa de 1 %
    haría que 99 441 pareciera compatible con muchos enteros cercanos. Para
    valores con parte decimal se conserva la tolerancia histórica, necesaria
    para redondeos como 0.9701 -> 0.97.
    """
    if candidate.is_integer() and value.is_integer():
        return candidate == value

    return abs(candidate - value) <= tolerance * max(abs(value), 1)


def _numeric_values(rows: list[dict]) -> tuple[set[float], set[float]]:
    """Obtiene valores crudos y equivalentes porcentuales de las filas."""
    raw_values: set[float] = set()
    percentage_values: set[float] = set()

    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (int, float, Decimal)):
                continue

            numeric_value = float(value)
            raw_values.add(numeric_value)

            if _looks_like_percentage_key(str(key)) and abs(numeric_value) <= 1:
                percentage_values.add(numeric_value * 100)

    return raw_values, percentage_values


def _find_metric_column(metric: str, row: dict) -> str | None:
    """Busca una columna cuyo nombre corresponda al identificador de metric.

    El generador nunca recibe instrucción estricta de nombrar la columna
    exactamente como el metric (ej. una fila puede traer
    `avg_resolution_time_hr` en vez de `resolution_time_hr`), así que una
    igualdad exacta produce falsos positivos casi siempre. Se acepta una
    coincidencia por substring en cualquier dirección, sin distinguir
    mayúsculas/minúsculas, como heurística tolerante.
    """
    metric_lower = metric.lower()

    for key in row:
        key_lower = key.lower()

        if metric_lower in key_lower or key_lower in metric_lower:
            return key

    return None


def check_result(
    rows: list[dict],
    optimized_query: OptimizedQuery,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> ResultCheck:
    """Detecta resultado vacío, métrica en NULL o truncación silenciosa.

    Args:
        rows: filas devueltas por `execute_sql`.
        optimized_query: para saber qué columnas son métricas a verificar.
        row_limit: el mismo tope pasado a `execute_sql`; si `len(rows)`
            lo iguala, es señal de que probablemente hay más filas de las
            mostradas (la truncación de `execute_sql` es silenciosa). Usa
            el mismo `DEFAULT_ROW_LIMIT` que `execute_sql` usa para truncar
            filas en Python, para que ambos topes no diverjan.

    Returns:
        `ResultCheck` con `ok=True` solo si no hay ninguna advertencia.
        Las advertencias no bloquean la respuesta, solo la marcan.
    """
    warnings: list[str] = []

    if not rows:
        warnings.append("La consulta no devolvió filas.")

    if len(rows) == row_limit:
        warnings.append(
            f"El resultado se truncó a {row_limit} filas; puede haber más datos."
        )

    if rows:
        for metric in optimized_query.metrics:
            column = _find_metric_column(metric, rows[0])

            # Si ninguna columna se parece al nombre del metric, no hay
            # suficiente evidencia para advertir: puede ser que el SQL
            # simplemente la haya nombrado de forma irreconocible, no que
            # el dato venga vacío.
            if column is None:
                continue

            if all(row.get(column) is None for row in rows):
                warnings.append(f"La métrica '{metric}' vino vacía en todas las filas.")

    return ResultCheck(ok=not warnings, warnings=warnings)


def check_groundedness(
    answer: str,
    rows: list[dict],
    tolerance: float = 0.01,
) -> GroundednessCheck:
    """Verifica que cada número del texto exista entre los valores de las filas.

    Extrae números del texto con una regex y los compara contra los
    valores numéricos de `rows`, con tolerancia de redondeo. Los números con
    un único separador y tres dígitos finales se consideran ambiguos entre
    decimal y miles; se prueban ambas lecturas y solo se acepta una si está
    respaldada por los datos ejecutados.

    Los conteos enteros deben coincidir exactamente; la tolerancia relativa
    se aplica únicamente cuando al menos uno de los valores tiene parte
    decimal. Esto evita falsos negativos en redondeos sin aceptar conteos
    cercanos pero incorrectos.

    Tiene falsos positivos esperables (números de fila, conteos, porcentajes
    derivados de dos columnas): por diseño es una señal de sospecha para
    disparar una única regeneración de la redacción, no un bloqueo automático.

    Args:
        answer: texto redactado por `synthesize_answer`.
        rows: filas reales que `answer` debería estar describiendo.
        tolerance: fracción de tolerancia relativa al comparar (0.01 = 1%).

    Returns:
        `GroundednessCheck` con los números del texto que no encontraron
        respaldo en `rows`.
    """
    number_pattern = re.compile(r"(?P<number>\d+(?:[.,]\d+)?)(?P<percent>\s*%)?")
    numbers_in_answer = list(number_pattern.finditer(answer))
    row_values, percentage_values = _numeric_values(rows)

    unsupported = []
    for match in numbers_in_answer:
        raw = match.group("number")
        candidates = _parse_answer_number_candidates(raw)
        candidate_values = percentage_values if match.group("percent") else row_values

        if not any(
            _numbers_match(candidate, value, tolerance)
            for candidate in candidates
            for value in candidate_values
        ):
            unsupported.append(raw + (" %" if match.group("percent") else ""))

    return GroundednessCheck(ok=not unsupported, unsupported_numbers=unsupported)
