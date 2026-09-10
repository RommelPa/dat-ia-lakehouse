"""Reglas de negocio globales, independientes de cualquier tabla.

Una política dentro de `data/ddl.json` solo llega al generador si su tabla
ya fue recuperada. Para conocimiento que decide QUÉ tabla hay que traer
(ej. "estado" sin dueño explícito significa el estado del cliente) ese
enganche es circular: la regla nunca se activaría porque la tabla que
necesita nunca llegó a recuperarse.

Este módulo resuelve eso por fuera del retrieval semántico: evalúa la
pregunta contra un catálogo fijo de reglas (`data/business_rules.json`) y,
cuando una coincide, fuerza la recuperación exacta de las tablas que exige
(mismo canal que `suggested_tables` usa hoy) y aporta su texto al prompt
del generador.

Activación determinística por términos, no por LLM: mismo criterio que ya
usa `query_optimizer._detect_filters` para no dejar en manos del modelo
decisiones que cambian qué se recupera o qué tabla se filtra.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_BUSINESS_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "business_rules.json"
)


@dataclass(frozen=True)
class BusinessRule:
    id: str
    activacion: tuple[str, ...]
    excluir: tuple[str, ...]
    tablas_requeridas: tuple[str, ...]
    tablas_alternativas: tuple[tuple[str, tuple[str, ...]], ...]
    regla: str


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_for_matching(value: str) -> str:
    """Sin acentos, minúsculas, espacios colapsados. Independiente del
    optimizer a propósito: evita un import circular (el optimizer es quien
    llama a este módulo)."""
    cleaned = re.sub(r"\s+", " ", value).strip()
    return _strip_accents(cleaned).lower()


def _contains_term(normalized_text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(normalize_for_matching(term)) + r"\b"
    return re.search(pattern, normalized_text) is not None


@lru_cache(maxsize=1)
def load_business_rules(
    path: Path = DEFAULT_BUSINESS_RULES_PATH,
) -> tuple[BusinessRule, ...]:
    """Carga y cachea `data/business_rules.json`. El cache es intencional:
    el archivo no cambia durante la vida del proceso."""
    raw_entries = json.loads(path.read_text(encoding="utf-8"))
    rules = []

    for entry in raw_entries:
        alternativas = tuple(
            (pattern, tuple(tables))
            for pattern, tables in (entry.get("tablas_alternativas") or {}).items()
        )
        rules.append(
            BusinessRule(
                id=entry["id"],
                activacion=tuple(entry.get("activacion") or []),
                excluir=tuple(entry.get("excluir") or []),
                tablas_requeridas=tuple(entry.get("tablas_requeridas") or []),
                tablas_alternativas=alternativas,
                regla=entry["regla"],
            )
        )

    return tuple(rules)


def match_business_rules(
    question: str,
    rules: tuple[BusinessRule, ...] | None = None,
) -> list[BusinessRule]:
    """Devuelve las reglas cuyo término de activación aparece en la
    pregunta y ninguno de sus términos de exclusión."""
    normalized_text = normalize_for_matching(question)
    candidate_rules = rules if rules is not None else load_business_rules()

    matched = []

    for rule in candidate_rules:
        if not any(_contains_term(normalized_text, term) for term in rule.activacion):
            continue

        if any(_contains_term(normalized_text, term) for term in rule.excluir):
            continue

        matched.append(rule)

    return matched


def _tables_for_rule(rule: BusinessRule, normalized_text: str) -> tuple[str, ...]:
    """Las tablas alternativas reemplazan a las requeridas cuando su patrón
    coincide (ej. "vendedores" reemplaza customer_state por seller_state,
    en vez de sumar ambas y ensuciar el contexto)."""
    for pattern, tables in rule.tablas_alternativas:
        if re.search(pattern, normalized_text):
            return tables

    return rule.tablas_requeridas


def required_tables(
    matched_rules: list[BusinessRule],
    question: str,
) -> list[str]:
    normalized_text = normalize_for_matching(question)
    tables: list[str] = []

    for rule in matched_rules:
        for table in _tables_for_rule(rule, normalized_text):
            if table not in tables:
                tables.append(table)

    return tables


def excluded_tables(
    matched_rules: list[BusinessRule],
    question: str,
) -> list[str]:
    """Tablas que una alternativa de negocio reemplaza explícitamente.

    Si una regla activa una alternativa, las tablas por defecto que no están
    en esa alternativa no deben reingresar por retrieval semántico. Esto
    permite que una regla no solo agregue contexto, sino que también elimine
    contexto incompatible con una interpretación de negocio más específica.
    """
    normalized_text = normalize_for_matching(question)
    excluded: list[str] = []

    for rule in matched_rules:
        selected = _tables_for_rule(rule, normalized_text)
        if selected == rule.tablas_requeridas:
            continue

        for table in rule.tablas_requeridas:
            if table not in selected and table not in excluded:
                excluded.append(table)

    return excluded


def render_business_rules(matched_rules: list[BusinessRule]) -> str:
    if not matched_rules:
        return ""

    return "\n".join(f"- {rule.regla}" for rule in matched_rules)
