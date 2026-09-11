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


def timed_call(
    timings_ms: MutableMapping[str, float] | None,
    stage: str,
    func: Callable[..., T],
    *args: Any,
    call_counts: MutableMapping[str, int] | None = None,
    **kwargs: Any,
) -> T:
    """Ejecuta una función y acumula su duración bajo stage.

    Cuando timings_ms es None no añade medición temporal. Si call_counts se
    proporciona, registra cada invocación visible de la etapa incluso cuando
    la función falla. Si una etapa se ejecuta varias veces (por ejemplo, un
    reintento del generador o de la síntesis), sus tiempos y llamadas se
    acumulan.
    """
    if call_counts is not None:
        call_counts[stage] = int(call_counts.get(stage, 0)) + 1

    if timings_ms is None:
        return func(*args, **kwargs)

    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        accumulated = float(timings_ms.get(stage, 0.0)) + elapsed_ms
        timings_ms[stage] = round(accumulated, 3)
