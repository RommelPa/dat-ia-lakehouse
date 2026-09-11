from app.observability import timing


def test_timed_call_records_elapsed_time(monkeypatch) -> None:
    ticks = iter([10.0, 10.125])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    timings = {}

    result = timing.timed_call(
        timings,
        "optimizer",
        lambda value: value * 2,
        3,
    )

    assert result == 6
    assert timings == {"optimizer": 125.0}


def test_timed_call_accumulates_retries(monkeypatch) -> None:
    ticks = iter([1.0, 1.050, 2.0, 2.075])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    timings = {}

    timing.timed_call(timings, "sql_generation", lambda: None)
    timing.timed_call(timings, "sql_generation", lambda: None)

    assert timings == {"sql_generation": 125.0}


def test_timed_call_can_be_disabled() -> None:
    result = timing.timed_call(None, "stage", lambda: "ok")

    assert result == "ok"
