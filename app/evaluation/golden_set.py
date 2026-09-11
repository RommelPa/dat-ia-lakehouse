"""Carga, valida y sincroniza el golden set de evaluación de Dat-IA."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from langsmith import Client

GOLDEN_SET_DATASET_NAME = "dat_ia_prd_golden_v2"
GOLDEN_SET_VERSION = "2.1.0"
DEFAULT_GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "evaluation"
    / "datasets"
    / "dat_ia_golden_set_v2.jsonl"
)

_EXAMPLE_ID_NAMESPACE = uuid.UUID("dc75eef4-3d98-43f2-b494-13b25a87ba11")
_SQL_COMMENT_PATTERN = re.compile(r"--[^\r\n]*|/\*.*?\*/", flags=re.DOTALL)
_READ_ONLY_START_PATTERN = re.compile(r"^\s*(?:SELECT|WITH)\b", flags=re.IGNORECASE)
_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(?:ALTER|CALL|COPY|CREATE|DELETE|DO|DROP|EXECUTE|GRANT|INSERT|"
    r"MERGE|REVOKE|TRUNCATE|UPDATE|VACUUM)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_TEXT_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_ANSWER_NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d.,]*(?![\w])")
_ANSWER_PERCENT_PATTERN = re.compile(
    r"(?<![\w])(?P<number>[-+]?\d[\d.,]*)\s*%(?![\w])"
)
_PERCENTAGE_FIELD_HINTS = {
    "pct",
    "percent",
    "percentage",
    "porcentaje",
    "rate",
    "ratio",
    "tasa",
}
_MONTH_NAMES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}
_UUID_PATTERN = re.compile(
    r"\b(?:[0-9a-fA-F]{32}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")


class GoldenSetValidationError(ValueError):
    """Indica que el archivo canónico no cumple el contrato esperado."""


def load_golden_set(
    path: str | Path = DEFAULT_GOLDEN_SET_PATH,
) -> list[dict[str, Any]]:
    """Carga el JSONL UTF-8 y valida su estructura básica."""
    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise GoldenSetValidationError(
            f"No se encontró el golden set en: {dataset_path}"
        )

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as golden_file:
        for line_number, raw_line in enumerate(golden_file, start=1):
            if not raw_line.strip():
                continue

            try:
                case = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise GoldenSetValidationError(
                    f"JSON inválido en la línea {line_number}: {exc.msg}"
                ) from exc

            _validate_case(case, line_number=line_number, case_ids=case_ids)
            case_ids.add(case["case_id"])

            if case.get("enabled", True):
                cases.append(case)

    if not cases:
        raise GoldenSetValidationError("El golden set no contiene casos habilitados.")

    return cases


def golden_set_content_hash(cases: Sequence[Mapping[str, Any]]) -> str:
    """Genera una huella estable para identificar el contenido evaluado."""
    import hashlib

    canonical = json.dumps(
        list(cases),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sync_golden_set(
    client: Client,
    *,
    cases: Sequence[Mapping[str, Any]],
    dataset_name: str = GOLDEN_SET_DATASET_NAME,
    prune: bool = False,
) -> dict[str, Any]:
    """Crea o actualiza el dataset de LangSmith usando IDs deterministas."""
    content_hash = golden_set_content_hash(cases)

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        created_dataset = False
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Golden set end-to-end de Dat-IA. "
                "Fuente canónica: tests/evaluation/datasets/"
                "dat_ia_golden_set_v2.jsonl."
            ),
            metadata={
                "solution": "dat_ia_test",
                "dataset_version": GOLDEN_SET_VERSION,
                "content_sha256": content_hash,
            },
        )
        created_dataset = True

    local_examples = {
        str(example["id"]): example
        for example in _to_langsmith_examples(cases, dataset_name=dataset_name)
    }
    remote_examples = {
        str(example.id): example
        for example in client.list_examples(dataset_id=dataset.id)
    }

    created_examples: list[dict[str, Any]] = []
    updated_count = 0
    unchanged_count = 0

    for example_id, local in local_examples.items():
        remote = remote_examples.get(example_id)

        if remote is None:
            created_examples.append(local)
            continue

        if _same_example(remote, local):
            unchanged_count += 1
            continue

        client.update_example(
            example_id,
            dataset_id=dataset.id,
            inputs=local["inputs"],
            outputs=local["outputs"],
            metadata=local["metadata"],
            split=local["split"],
        )
        updated_count += 1

    if created_examples:
        client.create_examples(
            dataset_id=dataset.id,
            examples=created_examples,
            max_concurrency=1,
        )

    remote_only_ids = sorted(set(remote_examples) - set(local_examples))

    if prune:
        for example_id in remote_only_ids:
            client.delete_example(example_id)

    return {
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "dataset_created": created_dataset,
        "created": len(created_examples),
        "updated": updated_count,
        "unchanged": unchanged_count,
        "remote_only": 0 if prune else len(remote_only_ids),
        "deleted": len(remote_only_ids) if prune else 0,
        "content_sha256": content_hash,
    }


def response_status_matches_expected(
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> bool:
    """Evalúa si el estado funcional coincide con el esperado."""
    return outputs.get("status") == reference_outputs.get("expected_status")


def reported_source_tables_match_expected(
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> bool:
    """Compara el conjunto de tablas declarado por la respuesta."""
    expected = {
        str(table).strip().casefold()
        for table in reference_outputs.get("expected_sources", [])
        if str(table).strip()
    }
    actual = _normalize_sources(outputs.get("sources"))
    return actual == expected


def generated_sql_is_read_only(outputs: Mapping[str, Any]) -> bool:
    """Comprueba que el SQL generado sea una consulta de solo lectura."""
    return is_read_only_sql(str(outputs.get("sql") or ""))


def result_facts_match_expected(
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> bool:
    """Compara hechos por fila sin depender del nombre de las columnas."""
    return compare_result_facts(
        outputs.get("data"),
        reference_outputs.get("expected_result"),
    )


def answer_contains_expected_facts(
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> bool:
    """Comprueba que la respuesta redactada incluya los hechos esperados."""
    answer = outputs.get("answer")
    expected = reference_outputs.get("expected_result")

    if not isinstance(answer, str) or not answer.strip():
        return False

    if not isinstance(expected, Mapping):
        return False

    expected_rows = expected.get("rows")
    tolerance = expected.get("numeric_tolerance", 0.0)

    if (
        not isinstance(expected_rows, Sequence)
        or isinstance(expected_rows, (str, bytes))
        or not isinstance(tolerance, (int, float))
    ):
        return False

    return all(
        _answer_contains_expected_fact(
            answer,
            expected_key,
            expected_value,
            tolerance=float(tolerance),
        )
        for row in expected_rows
        if isinstance(row, Mapping)
        for expected_key, expected_value in row.items()
    )


def compare_result_facts(actual: Any, expected: Any) -> bool:
    """Compara todos los hechos sin depender de alias ni del orden de filas."""
    contract = _result_comparison_contract(actual, expected)

    if contract is None:
        return False

    actual_rows, expected_rows, tolerance = contract
    unmatched_actual_rows = list(actual_rows)

    for expected_row in expected_rows:
        matching_index = next(
            (
                index
                for index, actual_row in enumerate(unmatched_actual_rows)
                if _row_contains_expected_facts(
                    actual_row,
                    expected_row,
                    tolerance=tolerance,
                )
            ),
            None,
        )

        if matching_index is None:
            return False

        unmatched_actual_rows.pop(matching_index)

    return True


def is_read_only_sql(sql: str) -> bool:
    """Acepta SELECT/CTE y rechaza instrucciones que mutan la base."""
    cleaned = _SQL_COMMENT_PATTERN.sub(" ", sql).strip()

    if not cleaned or not _READ_ONLY_START_PATTERN.match(cleaned):
        return False

    return _FORBIDDEN_SQL_PATTERN.search(cleaned) is None


def _validate_case(
    case: Any,
    *,
    line_number: int,
    case_ids: set[str],
) -> None:
    if not isinstance(case, dict):
        raise GoldenSetValidationError(
            f"La línea {line_number} debe contener un objeto JSON."
        )

    case_id = case.get("case_id")

    if not isinstance(case_id, str) or not case_id:
        raise GoldenSetValidationError(
            f"La línea {line_number} no tiene un case_id válido."
        )

    if case_id in case_ids:
        raise GoldenSetValidationError(f"case_id duplicado: {case_id}")

    inputs = case.get("inputs")

    if not isinstance(inputs, dict) or not str(inputs.get("question") or "").strip():
        raise GoldenSetValidationError(
            f"{case_id}: inputs.question debe ser un texto no vacío."
        )

    reference = case.get("reference_outputs")

    if not isinstance(reference, dict):
        raise GoldenSetValidationError(
            f"{case_id}: reference_outputs debe ser un objeto."
        )

    # if reference.get("expected_status") != "success":
    #     raise GoldenSetValidationError(
    #         f"{case_id}: el split core debe esperar status=success."
    #     )

    # sources = reference.get("expected_sources")

    # if not isinstance(sources, list) or not sources:
    #     raise GoldenSetValidationError(
    #         f"{case_id}: expected_sources debe ser una lista no vacía."
    #     )

    # sql = reference.get("reference_sql")

    # if not isinstance(sql, str) or not is_read_only_sql(sql):
    #     raise GoldenSetValidationError(
    #         f"{case_id}: reference_sql debe ser una consulta de solo lectura."
    #     )

    result = reference.get("expected_result")

    if not isinstance(result, dict):
        raise GoldenSetValidationError(
            f"{case_id}: expected_result debe ser un objeto."
        )

    rows = result.get("rows")
    row_count = result.get("row_count")

    # if not isinstance(rows, list) or not rows:
    #     raise GoldenSetValidationError(
    #         f"{case_id}: expected_result.rows debe contener al menos una fila."
    #     )

    if not isinstance(row_count, int) or row_count != len(rows):
        raise GoldenSetValidationError(
            f"{case_id}: row_count debe coincidir con todas las filas de referencia."
        )

    metadata = case.get("metadata")

    if not isinstance(metadata, dict):
        raise GoldenSetValidationError(f"{case_id}: metadata debe ser un objeto.")

    if metadata.get("dataset_version") != GOLDEN_SET_VERSION:
        raise GoldenSetValidationError(
            f"{case_id}: dataset_version debe ser {GOLDEN_SET_VERSION}."
        )


def _to_langsmith_examples(
    cases: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["case_id"])
        metadata = dict(case["metadata"])
        metadata["case_id"] = case_id

        examples.append(
            {
                "id": uuid.uuid5(
                    _EXAMPLE_ID_NAMESPACE,
                    f"{dataset_name}:{case_id}",
                ),
                "inputs": dict(case["inputs"]),
                "outputs": dict(case["reference_outputs"]),
                "metadata": metadata,
                "split": str(case.get("split") or "core"),
            }
        )

    return examples


def _same_example(remote: Any, local: Mapping[str, Any]) -> bool:
    return (
        remote.inputs == local["inputs"]
        and remote.outputs == local["outputs"]
        and (remote.metadata or {}) == local["metadata"]
    )


def _normalize_sources(value: Any) -> set[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence):
        values = value
    else:
        return set()

    return {str(item).strip().casefold() for item in values if str(item).strip()}


def _result_comparison_contract(
    actual: Any,
    expected: Any,
) -> tuple[Sequence[Any], Sequence[Any], float] | None:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        return None

    if not isinstance(expected, Mapping):
        return None

    expected_rows = expected.get("rows")
    expected_row_count = expected.get("row_count")
    tolerance = expected.get("numeric_tolerance", 0.0)

    if (
        not isinstance(expected_rows, Sequence)
        or isinstance(expected_rows, (str, bytes))
        or not isinstance(expected_row_count, int)
        or not isinstance(tolerance, (int, float))
    ):
        return None

    if len(actual) != expected_row_count or len(expected_rows) != len(actual):
        return None

    return actual, expected_rows, float(tolerance)


def _row_contains_expected_facts(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False

    unmatched_actual_values = list(actual.values())

    for expected_value in expected.values():
        matching_index = next(
            (
                index
                for index, actual_value in enumerate(unmatched_actual_values)
                if _values_equivalent(
                    actual_value,
                    expected_value,
                    tolerance=tolerance,
                )
            ),
            None,
        )

        if matching_index is None:
            return False

        unmatched_actual_values.pop(matching_index)

    return True


def _values_equivalent(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    actual_number = _as_decimal(actual)
    expected_number = _as_decimal(expected)

    if actual_number is not None and expected_number is not None:
        return abs(actual_number - expected_number) <= Decimal(str(tolerance))

    if isinstance(actual, str) and isinstance(expected, str):
        actual_uuid = _normalize_uuid(actual)
        expected_uuid = _normalize_uuid(expected)

        if actual_uuid is not None and expected_uuid is not None:
            return actual_uuid == expected_uuid

        if _MONTH_PATTERN.fullmatch(expected.strip()):
            return actual.strip().startswith(expected.strip())

        return _normalize_text(actual) == _normalize_text(expected)

    return actual == expected


def _looks_like_percentage_field(field_name: str) -> bool:
    tokens = re.findall(
        r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+",
        str(field_name).casefold(),
    )
    return bool(set(tokens) & _PERCENTAGE_FIELD_HINTS)


def _answer_contains_expected_fact(
    answer: str,
    field_name: str,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    """Compara hechos redactados conservando equivalencias semánticas comunes."""
    if isinstance(expected, str) and _MONTH_PATTERN.fullmatch(expected):
        year_text, month_text = expected.split("-", 1)
        month_name = _MONTH_NAMES_ES.get(int(month_text))
        if month_name is not None:
            normalized_answer = _normalize_text(answer)
            if f"{month_name} {year_text}" in normalized_answer:
                return True

    expected_number = _as_decimal(expected)
    if (
        expected_number is not None
        and abs(expected_number) <= Decimal("1")
        and _looks_like_percentage_field(field_name)
    ):
        for match in _ANSWER_PERCENT_PATTERN.finditer(answer):
            for candidate in _numeric_token_candidates(match.group("number")):
                ratio_candidate = candidate / Decimal("100")
                if abs(ratio_candidate - expected_number) <= Decimal(str(tolerance)):
                    return True

    return _answer_contains_value(
        answer,
        expected,
        tolerance=tolerance,
    )


def _answer_contains_value(
    answer: str,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    expected_number = _as_decimal(expected)

    if expected_number is not None:
        return any(
            abs(candidate - expected_number) <= Decimal(str(tolerance))
            for token in _ANSWER_NUMBER_PATTERN.findall(answer)
            for candidate in _numeric_token_candidates(token)
        )

    if isinstance(expected, str):
        expected_uuid = _normalize_uuid(expected)

        if expected_uuid is not None:
            return expected_uuid in {
                normalized
                for value in _UUID_PATTERN.findall(answer)
                if (normalized := _normalize_uuid(value)) is not None
            }

        normalized_answer = _normalize_text(answer)
        normalized_expected = _normalize_text(expected)

        if len(normalized_expected) <= 3:
            return (
                re.search(
                    rf"(?<!\w){re.escape(normalized_expected)}(?!\w)",
                    normalized_answer,
                )
                is not None
            )

        return normalized_expected in normalized_answer

    return str(expected) in answer


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    if isinstance(value, str) and _NUMERIC_TEXT_PATTERN.fullmatch(value.strip()):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return None

    return None


def _numeric_token_candidates(token: str) -> set[Decimal]:
    cleaned = token.strip().strip(".,")

    if not cleaned:
        return set()

    candidates: set[Decimal] = set()

    def add_candidate(value: str) -> None:
        try:
            candidates.add(Decimal(value))
        except InvalidOperation:
            return

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(".") > cleaned.rfind(","):
            add_candidate(cleaned.replace(",", ""))
        else:
            add_candidate(cleaned.replace(".", "").replace(",", "."))
        return candidates

    separator = "," if "," in cleaned else "." if "." in cleaned else None

    if separator is None:
        add_candidate(cleaned)
        return candidates

    sign = ""
    unsigned = cleaned

    if unsigned[0] in "+-":
        sign, unsigned = unsigned[0], unsigned[1:]

    parts = unsigned.split(separator)

    if len(parts) == 2:
        add_candidate(f"{sign}{parts[0]}.{parts[1]}")

    if len(parts) > 1 and all(len(group) == 3 for group in parts[1:]):
        add_candidate(sign + "".join(parts))

    return candidates


def _normalize_uuid(value: str) -> str | None:
    stripped = value.strip()

    if _UUID_PATTERN.fullmatch(stripped) is None:
        return None

    return stripped.replace("-", "").casefold()


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    with_spaces = re.sub(r"[_\s]+", " ", without_accents)
    return with_spaces.strip()
