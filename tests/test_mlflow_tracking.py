from pathlib import Path
from types import SimpleNamespace


from app.observability.mlflow_tracking import (
    _report_params,
    _summary_metrics,
    log_benchmark_report,
)


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.run_name = None
        self.params = None
        self.metrics = None
        self.artifacts = []

    def set_tracking_uri(self, value):
        self.tracking_uri = value

    def set_experiment(self, value):
        self.experiment_name = value

    def start_run(self, run_name=None):
        self.run_name = run_name
        return _FakeRun("run-123")

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


def _report() -> dict:
    return {
        "benchmark_type": "text_to_sql_end_to_end",
        "backend": "databricks",
        "dataset_version": "2.1.0",
        "dataset_content_sha256": "abc123",
        "selected_cases": 30,
        "skipped_cases": 0,
        "ready": {
            "backend": "databricks",
            "database": "connected",
        },
        "summary": {
            "metrics": {
                "effective_result_match": {
                    "passed": 30,
                    "total": 30,
                    "rate": 1.0,
                }
            },
            "runner_errors": 0,
            "cases_with_warnings": 2,
            "avg_attempts": 1.1,
            "avg_latency_ms": 1234.5,
            "min_latency_ms": 500.0,
            "max_latency_ms": 3000.0,
        },
    }


def test_summary_metrics_flattens_benchmark_metrics() -> None:
    metrics = _summary_metrics(_report())

    assert metrics["effective_result_match.rate"] == 1.0
    assert metrics["effective_result_match.passed"] == 30.0
    assert metrics["avg_latency_ms"] == 1234.5


def test_report_params_keep_low_cardinality_metadata() -> None:
    params = _report_params(_report())

    assert params["backend"] == "databricks"
    assert params["dataset_version"] == "2.1.0"
    assert params["ready_backend"] == "databricks"


def test_log_benchmark_report_uses_local_mlflow_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark.json"
    report_path.write_text("{}", encoding="utf-8")
    fake = FakeMlflow()

    run_id = log_benchmark_report(
        _report(),
        report_path=report_path,
        tracking_uri="file:./mlruns-test",
        experiment_name="dat_ia_tests",
        run_name="baseline-v2",
        mlflow_module=fake,
    )

    assert run_id == "run-123"
    assert fake.tracking_uri == "file:./mlruns-test"
    assert fake.experiment_name == "dat_ia_tests"
    assert fake.run_name == "baseline-v2"
    assert fake.params["backend"] == "databricks"
    assert fake.metrics["effective_result_match.rate"] == 1.0
    assert fake.artifacts == [(str(report_path), "reports")]


def test_log_benchmark_report_skips_missing_artifact(tmp_path: Path) -> None:
    fake = FakeMlflow()

    run_id = log_benchmark_report(
        _report(),
        report_path=tmp_path / "missing.json",
        mlflow_module=fake,
    )

    assert run_id == "run-123"
    assert fake.artifacts == []
