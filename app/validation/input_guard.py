"""Guardas determinísticas para entradas Text-to-SQL.

SQLPromptShield se conserva como segunda capa probabilística. Estas reglas
cubren instrucciones de mutación e inyección inequívocas que no deberían
quedar a merced de un clasificador.
"""

from __future__ import annotations

import re
import unicodedata

MALICIOUS_CLASSIFIER_THRESHOLD = 0.60

_SQL_MUTATION_PATTERNS = (
    r"\bdelete\s+from\b",
    r"\bdrop\s+table\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+[a-z_][\w.]*\s+set\b",
    r"\btruncate(?:\s+table)?\b",
    r"\balter\s+table\b",
    r"\bcreate\s+table\b",
)
_SPANISH_MUTATION_PATTERNS = (
    r"\binserta(?:r)?\b.{0,80}\ben\s+la\s+tabla\b",
    r"\bcambi(?:a|ar|e|es)\b.{0,80}\btodas?\s+las?\b",
    r"\bborra(?:r)?\b.{0,80}\b(?:tabla|registros?|ventas?)\b",
    r"\belimina(?:r)?\b.{0,80}\b(?:tabla|registros?|ventas?)\b",
)
_PROMPT_INJECTION_PATTERNS = (
    r"\bignor(?:a|ar|e|es|en)\b.{0,80}\binstrucciones?\b",
    r"\bignore\b.{0,80}\binstructions?\b",
)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def deterministic_input_block_reason(text: str) -> str | None:
    """Devuelve una razón cuando la entrada contiene un ataque inequívoco."""
    normalized = _normalize(text)

    for pattern in (
        *_SQL_MUTATION_PATTERNS,
        *_SPANISH_MUTATION_PATTERNS,
        *_PROMPT_INJECTION_PATTERNS,
    ):
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "La entrada contiene una instrucción de mutación o inyección no permitida."

    return None


def should_block_input(text: str, classifier_label: str, classifier_score: float) -> bool:
    """Combina reglas determinísticas con SQLPromptShield.

    Una predicción MALICIOUS de baja confianza no bloquea por sí sola. La
    frontera 0.60 deja una zona de incertidumbre para falsos positivos del
    clasificador; las mutaciones/inyecciones literales siguen bloqueándose
    antes por reglas determinísticas.
    """
    if deterministic_input_block_reason(text) is not None:
        return True

    return (
        classifier_label == "MALICIOUS"
        and classifier_score >= MALICIOUS_CLASSIFIER_THRESHOLD
    )
