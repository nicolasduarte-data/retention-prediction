"""Unit tests for src/retention/models/tracking.py — log_run() coverage.

Why isolation matters here:

    log_run() has two global side-effects:
    1. mlflow.set_tracking_uri() — redirects all subsequent mlflow calls to
       a specific directory on disk.
    2. mlflow.set_experiment() — sets the active experiment in MLflow's
       global state.

    Without isolation each test would write to the real ``mlruns/`` directory
    (polluting the experiment store) and leave global state that affects
    subsequent tests.

Isolation strategy — two layers:

    Layer 1 — config redirect (monkeypatch):
        ``log_run()`` reads ``config.PROJECT_ROOT`` to build the tracking URI.
        ``monkeypatch.setattr("retention.models.tracking.config", _FakeConfig(tmp_path))``
        causes log_run() to call ``mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())``
        instead of the real project root. No real mlruns/ is touched.

    Layer 2 — URI teardown (yield fixture):
        mlflow.set_tracking_uri() is a global call — monkeypatch can't undo it
        automatically. The ``isolated_store`` fixture saves the original URI
        before the test and restores it in teardown, so subsequent tests that
        use mlflow directly (e.g. nbmake smoke tests) see the correct store.

Reading back from the store:

    After log_run() returns, MLflow's global tracking URI is already pointing
    to tmp_path (set inside log_run()). MlflowClient(tracking_uri=...) creates
    an independent client scoped to that URI, making the read-back explicit and
    not reliant on the global state ordering.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import mlflow
import mlflow.exceptions
import mlflow.tracking
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from retention.models.tracking import log_run, register_champion


# ------------------------------------------------------------------ #
# Fake config + fixture                                                 #
# ------------------------------------------------------------------ #


@dataclass
class _FakeConfig:
    """Minimal stand-in for the ``retention.config`` module.

    ``log_run()`` accesses exactly one attribute: ``config.PROJECT_ROOT``.
    This dataclass satisfies that contract and nothing else — keeping it
    minimal means the test doesn't break if config grows new attributes.
    """

    PROJECT_ROOT: Path


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect log_run() to write to tmp_path/mlruns instead of the real store.

    Yields ``tmp_path`` so tests can build expected paths and pass them to
    MlflowClient(tracking_uri=...) for read-back assertions.

    Teardown (after yield):
        Restores the tracking URI that was active before the test so that
        subsequent mlflow calls in the same process see the correct store.
        monkeypatch handles the config restoration automatically.

    Why not use autouse?
        Only tests in this file need the isolation fixture. autouse would
        apply it to the whole session, adding unnecessary overhead.
    """
    original_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr("retention.models.tracking.config", _FakeConfig(PROJECT_ROOT=tmp_path))
    yield tmp_path
    # Layer 2: undo the global tracking URI side-effect from log_run().
    mlflow.set_tracking_uri(original_uri)


def _client(store_root: Path) -> mlflow.tracking.MlflowClient:
    """Return a scoped MlflowClient pointing at store_root/mlruns.

    Using an explicit URI in the client constructor means the read-back
    assertions are independent of the global tracking URI state — we know
    exactly which store we're querying.
    """
    return mlflow.tracking.MlflowClient(tracking_uri=(store_root / "mlruns").as_uri())


def _unique_experiment() -> str:
    """Return a unique experiment name so tests never share an experiment."""
    return f"test-tracking-{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------ #
# Return value contract                                                 #
# ------------------------------------------------------------------ #


def test_log_run_returns_non_empty_string(isolated_store: Path) -> None:
    """log_run() must return a non-empty run_id string.

    The run_id is the downstream handle for champion registration
    (Story 2.7.10: ``mlflow.register_model(f"runs:/{run_id}/model", ...)``)
    so its type and emptiness are contractual.
    """
    run_id = log_run(
        run_name="contract_test",
        params={"model": "LR"},
        metrics={"auc_pr": 0.5},
        experiment_name=_unique_experiment(),
    )

    # Type: must be str (not None, not int, not mlflow.ActiveRun)
    assert isinstance(run_id, str), f"log_run() must return str, got {type(run_id).__name__!r}"
    # Non-empty: an empty string would make f"runs:/{run_id}/model" malformed
    assert len(run_id) > 0, "log_run() returned an empty run_id string"


def test_log_run_run_id_looks_like_mlflow_uuid(isolated_store: Path) -> None:
    """The returned run_id should be a 32-character hex string (MLflow UUID format).

    MLflow run IDs are UUIDs with hyphens stripped, e.g.
    ``"3a4ae17e4d154e6f8bd403c14478abc1"``.  32 characters, hex only.  # pragma: allowlist secret
    Verifying this format catches future changes to MLflow's run ID generation
    that would silently break downstream registry calls.
    """
    run_id = log_run(
        run_name="uuid_format_test",
        params={},
        metrics={"brier": 0.158},
        experiment_name=_unique_experiment(),
    )

    assert len(run_id) == 32, (
        f"Expected 32-character run_id (MLflow UUID), got {len(run_id)} chars: {run_id!r}"
    )
    assert all(c in "0123456789abcdef" for c in run_id), (
        f"run_id contains non-hex characters: {run_id!r}"
    )


# ------------------------------------------------------------------ #
# Metrics written to store                                              #
# ------------------------------------------------------------------ #


def test_log_run_metrics_are_written_to_store(isolated_store: Path) -> None:
    """Metrics passed to log_run() must be readable from the MLflow store.

    This is the core contract: the function must actually persist the metrics,
    not just receive them. A broken mlflow.log_metrics() call would silently
    succeed (no exception) but leave an empty Metrics tab in the UI.
    """
    metrics = {"auc_pr": 0.309, "auc_roc": 0.661, "brier": 0.158}
    run_id = log_run(
        run_name="metrics_write_test",
        params={},
        metrics=metrics,
        experiment_name=_unique_experiment(),
    )

    # Read back through an explicit-URI client (not the global URI)
    stored = _client(isolated_store).get_run(run_id).data.metrics

    for key, expected_value in metrics.items():
        assert key in stored, (
            f"Metric key {key!r} not found in stored run. "
            "log_run() may have called log_metrics() with wrong data."
        )
        assert stored[key] == pytest.approx(expected_value, rel=1e-6), (
            f"Metric {key!r}: expected {expected_value}, got {stored[key]}"
        )


def test_log_run_empty_metrics_dict_does_not_error(isolated_store: Path) -> None:
    """log_run() with metrics={} must succeed — some future callers may omit metrics."""
    run_id = log_run(
        run_name="empty_metrics_test",
        params={"model": "sanity"},
        metrics={},
        experiment_name=_unique_experiment(),
    )
    assert isinstance(run_id, str) and len(run_id) > 0


# ------------------------------------------------------------------ #
# Params written to store                                               #
# ------------------------------------------------------------------ #


def test_log_run_params_are_written_to_store(isolated_store: Path) -> None:
    """Params passed to log_run() must be readable from the MLflow store.

    MLflow stores params as strings internally (``mlflow.log_params()``
    calls ``str()`` on each value). The test uses string params to avoid
    roundtrip conversion ambiguity — the exact form of the stored value
    is verified against the str() representation.
    """
    params = {
        "model": "GBM",
        "cohort": "hybrid",
        "n_estimators": "300",
        "learning_rate": "0.05",
    }
    run_id = log_run(
        run_name="params_write_test",
        params=params,
        metrics={"auc_pr": 0.309},
        experiment_name=_unique_experiment(),
    )

    stored = _client(isolated_store).get_run(run_id).data.params

    for key, expected_value in params.items():
        assert key in stored, f"Param key {key!r} not found in stored run."
        assert stored[key] == str(expected_value), (
            f"Param {key!r}: expected {str(expected_value)!r}, got {stored[key]!r}"
        )


# ------------------------------------------------------------------ #
# Run name and experiment name                                          #
# ------------------------------------------------------------------ #


def test_log_run_run_name_is_stored(isolated_store: Path) -> None:
    """The run_name argument must appear in the MLflow run's metadata.

    run_name is the human-readable label in the UI run list. If it's not
    stored, the comparison notebook's ``GBM_hybrid`` labels would all show
    as unnamed runs, making the UI useless for comparison.
    """
    run_id = log_run(
        run_name="GBM_hybrid",
        params={},
        metrics={"auc_pr": 0.309},
        experiment_name=_unique_experiment(),
    )

    run = _client(isolated_store).get_run(run_id)
    # MLflow stores run_name in run.info.run_name (MLflow ≥ 1.24)
    assert run.info.run_name == "GBM_hybrid", (
        f"Expected run_name='GBM_hybrid', got {run.info.run_name!r}"
    )


def test_log_run_uses_custom_experiment_name(isolated_store: Path) -> None:
    """Runs must land in the experiment matching the experiment_name argument.

    The experiment_name kwarg is the isolation mechanism for tests — without
    it every test would write to 'rp-loop2' and produce 'Test Run' clutter
    in the production experiment store (visible in the UI after ``make train``).
    """
    experiment_name = _unique_experiment()
    run_id = log_run(
        run_name="exp_name_test",
        params={},
        metrics={"auc_pr": 0.5},
        experiment_name=experiment_name,
    )

    client = _client(isolated_store)
    run = client.get_run(run_id)
    # Resolve which experiment this run belongs to
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment is not None
    assert experiment.name == experiment_name, (
        f"Expected experiment {experiment_name!r}, got {experiment.name!r}"
    )


def test_log_run_default_experiment_name_is_rp_loop2(isolated_store: Path) -> None:
    """When experiment_name is omitted, runs must land in 'rp-loop2'.

    'rp-loop2' is the canonical Loop 2 experiment name defined in
    ``_EXPERIMENT_NAME``. This test pins that default — a future refactor
    that renames it would break the README's 3-step reproduction recipe
    (``make mlflow-ui`` → sort by auc_pr).
    """
    run_id = log_run(
        run_name="default_exp_test",
        params={},
        metrics={"auc_pr": 0.3},
        # experiment_name intentionally omitted → should use _EXPERIMENT_NAME
    )

    client = _client(isolated_store)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment is not None
    assert experiment.name == "rp-loop2", (
        f"Default experiment should be 'rp-loop2', got {experiment.name!r}. "
        "If you changed _EXPERIMENT_NAME, update the README and methodology.md too."
    )


# ------------------------------------------------------------------ #
# artifact_paths code paths                                             #
# ------------------------------------------------------------------ #


def test_log_run_artifact_paths_none_does_not_error(isolated_store: Path) -> None:
    """artifact_paths=None must complete without error and return a run_id.

    None is the documented default: "Pass None or [] to skip artifact logging."
    This is the common case for runs where model persistence is handled
    separately (e.g. stored as a serialised pipeline, not an MLflow artifact).
    """
    run_id = log_run(
        run_name="artifact_none_test",
        params={},
        metrics={"auc_pr": 0.5},
        artifact_paths=None,
        experiment_name=_unique_experiment(),
    )
    assert isinstance(run_id, str) and len(run_id) > 0


def test_log_run_artifact_paths_empty_list_does_not_error(isolated_store: Path) -> None:
    """artifact_paths=[] must complete without error and return a run_id.

    [] is the other documented skip path. Callers may prefer [] over None
    when building the list programmatically (appending paths conditionally).
    The ``if artifact_paths:`` guard in log_run() handles both falsy values.
    """
    run_id = log_run(
        run_name="artifact_empty_test",
        params={},
        metrics={"auc_pr": 0.5},
        artifact_paths=[],
        experiment_name=_unique_experiment(),
    )
    assert isinstance(run_id, str) and len(run_id) > 0


def test_log_run_artifact_paths_with_real_file_logs_artifact(
    isolated_store: Path, tmp_path: Path
) -> None:
    """artifact_paths=[path] must copy the file into the MLflow artifact store.

    This exercises the ``for path in artifact_paths: mlflow.log_artifact(path)``
    branch in log_run(). Without this test, a broken log_artifact() call
    (e.g. wrong path type, file not found) would go undetected until a
    reviewer runs ``make train`` and finds an empty Artifacts tab.

    We create a real temp file (``tmp_path / "report.txt"``) and verify it
    is present in the run's artifact directory after logging.
    """
    # Create a real file to log — mlflow.log_artifact() reads from disk
    artifact_file = tmp_path / "report.txt"
    artifact_file.write_text("loop2 comparison output")

    experiment_name = _unique_experiment()
    run_id = log_run(
        run_name="artifact_file_test",
        params={},
        metrics={"auc_pr": 0.5},
        artifact_paths=[str(artifact_file)],
        experiment_name=experiment_name,
    )

    # Verify the artifact was logged by listing artifacts for this run
    client = _client(isolated_store)
    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]

    assert "report.txt" in artifact_names, (
        f"Expected 'report.txt' in logged artifacts, got: {artifact_names}. "
        "Check that log_run() calls mlflow.log_artifact() when artifact_paths is non-empty."
    )


# ------------------------------------------------------------------ #
# register_champion — Model Registry (Story 2.7.10)                     #
# ------------------------------------------------------------------ #


def _tiny_fitted_model() -> LogisticRegression:
    """A minimally-fitted sklearn estimator for registry tests.

    register_champion only needs a fitted, sklearn-compatible object to hand to
    ``mlflow.sklearn.log_model`` — the data is irrelevant to what we assert
    (name, version, stage, logged params/metrics), so we use the smallest fit
    that still has both classes present. ``pip_requirements=["scikit-learn"]``
    is passed at every call site to skip MLflow's ~20s environment inference.
    """
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.array([0, 1] * 5)
    return LogisticRegression(max_iter=1000).fit(X, y)


def test_register_champion_returns_run_id_and_version(isolated_store: Path) -> None:
    """register_champion returns ``(run_id: str, version: int)`` — first reg is v1.

    Each test gets a fresh ``tmp_path`` store via ``isolated_store``, so the
    registry starts empty and the first registration is always version 1.
    """
    run_id, version = register_champion(
        _tiny_fitted_model(),
        experiment_name=_unique_experiment(),
        pip_requirements=["scikit-learn"],
    )
    assert isinstance(run_id, str) and len(run_id) > 0
    assert isinstance(version, int)
    assert version == 1


def test_register_champion_registers_under_name(isolated_store: Path) -> None:
    """The model is registered under the given name and is queryable by it."""
    name = "rp-champion-test"
    register_champion(
        _tiny_fitted_model(),
        registered_name=name,
        experiment_name=_unique_experiment(),
        pip_requirements=["scikit-learn"],
    )
    model = _client(isolated_store).get_registered_model(name)
    assert model.name == name


def test_register_champion_promotes_to_production(isolated_store: Path) -> None:
    """The registered version is transitioned to the Production stage.

    This is the win-condition contract: ``models:/rp-champion/Production`` must
    resolve, which requires the version to actually carry the Production stage.
    """
    name = "rp-champion-test"
    _, version = register_champion(
        _tiny_fitted_model(),
        registered_name=name,
        stage="Production",
        experiment_name=_unique_experiment(),
        pip_requirements=["scikit-learn"],
    )
    mv = _client(isolated_store).get_model_version(name, str(version))
    assert mv.current_stage == "Production"


def test_register_champion_logs_params_and_metrics(isolated_store: Path) -> None:
    """Provenance params/metrics passed to register_champion land on the run."""
    run_id, _ = register_champion(
        _tiny_fitted_model(),
        params={"model": "GBM", "cohort": "hybrid"},
        metrics={"test_auc_pr": 0.27},
        experiment_name=_unique_experiment(),
        pip_requirements=["scikit-learn"],
    )
    run = _client(isolated_store).get_run(run_id)
    assert run.data.params["model"] == "GBM"
    assert run.data.params["cohort"] == "hybrid"
    assert run.data.metrics["test_auc_pr"] == pytest.approx(0.27)


def test_register_champion_second_call_increments_version(isolated_store: Path) -> None:
    """Registering the same name twice in one store yields versions 1 then 2."""
    name = "rp-champion-test"
    exp = _unique_experiment()
    _, v1 = register_champion(
        _tiny_fitted_model(),
        registered_name=name,
        experiment_name=exp,
        pip_requirements=["scikit-learn"],
    )
    _, v2 = register_champion(
        _tiny_fitted_model(),
        registered_name=name,
        experiment_name=exp,
        pip_requirements=["scikit-learn"],
    )
    assert (v1, v2) == (1, 2)


def test_register_champion_raises_actionable_error_on_registry_failure(
    isolated_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If registration fails AFTER the model is logged, register_champion raises
    an actionable RuntimeError — not a bare MlflowException (feast T2-SEL-3).

    The model is logged inside the run (which closes first); only the
    register/promote step is made to fail. The raised error must name the run
    and the intact model URI so the operator re-runs registration, not training.
    """

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise mlflow.exceptions.MlflowException("simulated registry write failure")

    monkeypatch.setattr("mlflow.register_model", _boom)

    with pytest.raises(RuntimeError, match="re-run register_champion"):
        register_champion(
            _tiny_fitted_model(),
            experiment_name=_unique_experiment(),
            pip_requirements=["scikit-learn"],
        )
