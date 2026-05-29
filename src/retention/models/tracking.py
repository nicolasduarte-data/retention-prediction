"""MLflow experiment tracking helpers — Story 2.7.

A thin wrapper over raw mlflow calls that keeps the tracking contract typed,
testable, and cwd-independent. Why a wrapper instead of calling mlflow
directly in the notebook?

- **Typed, auditable contract.** The function signature documents exactly
  which params and metrics the project guarantees to log for every run.
  Notebook cells bypass mypy; this module doesn't.
- **cwd-independence.** MLflow's default tracking URI is ``./mlruns``
  (relative to wherever Python was launched). Under ``jupyter nbconvert``
  the cwd is ``notebooks/``, which would create a stray ``notebooks/mlruns/``.
  ``log_run()`` pins the URI to ``config.PROJECT_ROOT / "mlruns"`` so every
  caller — notebook, test, script — writes to the same project-root store.
- **Reproducibility narrative.** Logging every training run (not just the
  winner) means a fresh-clone reviewer can see the full experiment history
  after running ``make train``. That's the portfolio signal.

Story 2.7.10 (champion registration, after Story 3.6) will extend this module
with a ``register_champion(run_id, model_uri)`` helper that wraps
``mlflow.register_model()`` and promotes to the local Production stage.
"""

from __future__ import annotations

from typing import Any

import mlflow

from retention import config

_EXPERIMENT_NAME: str = "rp-loop2"
"""Default experiment name.  All Loop 2 runs land here so the UI shows them
side-by-side and sortable by AUC-PR.  Override via the ``experiment_name``
argument for future loops or for test isolation."""


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[str] | None = None,
    *,
    experiment_name: str = _EXPERIMENT_NAME,
) -> str:
    """Log one training run to the local MLflow backend store.

    Creates (or reuses) the experiment, opens a run, logs params and metrics,
    and optionally logs files as artifacts.  Returns the ``run_id`` so callers
    can reference this run later — e.g. for ``mlflow.register_model()`` in
    Story 2.7.10 (champion registration after Epic 3 champion selection).

    The backend URI is always pinned to ``config.PROJECT_ROOT / "mlruns"``
    (an absolute path) so the store location is independent of the Python
    process's working directory.  This is critical for notebook execution via
    ``jupyter nbconvert``, which runs with ``cwd=notebooks/``.

    Args:
        run_name: Human-readable label shown in the MLflow UI run list.
            Convention: ``"{MODEL}_{COHORT}"``, e.g. ``"GBM_hybrid"``.
        params: Hyperparameters to log.  MLflow converts values to strings
            internally; pass str, int, float, or bool.  Logged under the
            "Parameters" tab.  Keys must match ``[a-zA-Z0-9._\\- /]+``
            (no ``@`` or ``%``).
        metrics: Scalar evaluation metrics (float).  Keys must match the
            same pattern.  Use ``prec_at_10`` not ``Prec@10%``.  Logged
            under the "Metrics" tab and used for UI column sorting.
        artifact_paths: Local file paths to copy into the run's artifact
            store.  Pass ``None`` or ``[]`` to skip (default — for runs
            where model persistence is handled separately or deferred).
        experiment_name: MLflow experiment name.  Default: ``"rp-loop2"``.
            Override in tests to avoid polluting the real experiment store.

    Returns:
        The MLflow ``run_id`` (UUID string), e.g. for later registry calls::

            run_id = log_run("GBM_hybrid", params, metrics)
            mlflow.register_model(f"runs:/{run_id}/model", "rp-champion")

    Raises:
        mlflow.exceptions.MlflowException: if MLflow cannot write to the
            backend store (permissions, disk full, etc.).

    Example::

        run_id = log_run(
            run_name="GBM_hybrid",
            params={"model": "GBM", "cohort": "hybrid", "n_estimators": 300},
            metrics={"auc_pr": 0.309, "brier": 0.158},
        )
        print(f"Logged as {run_id[:8]}…")
    """
    # Pin the URI to the project root so it's cwd-independent — see module
    # docstring for why this matters under nbconvert.
    # Use Path.as_uri() to produce a proper file:///... URI (important on
    # Windows: a bare absolute path like C:\... is parsed as scheme "C", which
    # MLflow doesn't recognise as a supported file-store scheme).
    mlflow.set_tracking_uri((config.PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)
        return str(run.info.run_id)


__all__ = ["log_run"]
