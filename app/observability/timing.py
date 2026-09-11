"""Medición ligera de latencia por etapa del pipeline.

No depende de MLflow ni LangSmith. Solo acumula milisegundos en un mapping
proporcionado por el llamador, por lo que puede usarse en producción, tests y
benchmarks sin estado global compartido.
"""

from __future__ import annotations

import time
from collections.abc import Callable, MutableMapping
from typing import Any, TypeVar

T = TypeVar("T")


class StageExecutionError(RuntimeError):
    """Error seguro que conserva la etapa sin exponer el mensaje original."""

    def __init__(self, stage: str, original: Exception) -> None:
        self.stage = stage
        self.error_type = type(original).__name__
        super().__init__(f"{stage} failed with {self.error_type}")


def timed_call(
    timings_ms: MutableMapping[str, float] | None,
    stage: str,
    func: Callable[..., T],
    *args: Any,
    call_counts: MutableMapping[str, int] | None = None,
    wrap_exceptions: bool = False,
    **kwargs: Any,
) -> T:
    """Ejecuta una función y acumula su duración bajo stage.

    Cuando timings_ms es None no añade medición temporal. Si call_counts se
    proporciona, registra cada invocación visible de la etapa incluso cuando
    la función falla. Si una etapa se ejecuta varias veces (por ejemplo, un
    reintento del generador o de la síntesis), sus tiempos y llamadas se
    acumulan. Si wrap_exceptions=True, cualquier excepción se envuelve en
    StageExecutionError para conservar la etapa y el tipo de error sin copiar
    el mensaje potencialmente sensible de la dependencia externa.
    """
    if call_counts is not None:
        call_counts[stage] = int(call_counts.get(stage, 0)) + 1

    started = time.perf_counter() if timings_ms is not None else None

    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if wrap_exceptions:
            raise StageExecutionError(stage, exc) from exc
        raise
    finally:
        if timings_ms is not None and started is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            accumulated = float(timings_ms.get(stage, 0.0)) + elapsed_ms
            timings_ms[stage] = round(accumulated, 3)
