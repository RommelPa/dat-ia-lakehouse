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


def test_timed_call_counts_visible_invocations(monkeypatch) -> None:
    ticks = iter([1.0, 1.010, 2.0, 2.020])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    timings = {}
    calls = {}

    timing.timed_call(
        timings,
        "sql_judgement",
        lambda: None,
        call_counts=calls,
    )
    timing.timed_call(
        timings,
        "sql_judgement",
        lambda: None,
        call_counts=calls,
    )

    assert calls == {"sql_judgement": 2}
    assert timings == {"sql_judgement": 30.0}


def test_timed_call_counts_failed_invocation() -> None:
    calls = {}

    def fail() -> None:
        raise RuntimeError("boom")

    try:
        timing.timed_call(
            None,
            "optimizer",
            fail,
            call_counts=calls,
        )
    except RuntimeError:
        pass

    assert calls == {"optimizer": 1}
