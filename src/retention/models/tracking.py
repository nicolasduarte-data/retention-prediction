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

Story 2.7.10 (champion registration, bundled with Story 3.6) adds
``register_champion()``: it logs the chosen estimator as an MLflow model,
registers it under the ``rp-champion`` name, and promotes that version to the
``Production`` stage in the local file-store registry — the registry view the
README's "Reproducing the MLflow experiment view" subsection screenshots.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn

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


def register_champion(
    model: Any,
    *,
    run_name: str = "champion_registration",
    registered_name: str = "rp-champion",
    stage: str = "Production",
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    pip_requirements: list[str] | None = None,
    experiment_name: str = _EXPERIMENT_NAME,
) -> tuple[str, int]:
    """Log the champion estimator and promote it to the Production stage.

    This is the Model Registry counterpart to ``log_run``. Where ``log_run``
    records *every* training run for the experiment-view narrative, this records
    the *one* winner and elevates it to a named, versioned, stage-tagged entry —
    the registry view a fresh-clone reviewer sees, and the URI Loop 4 will load
    from (``models:/rp-champion/Production``).

    The three steps, all against the same cwd-independent file store:

    1. **Log the model** inside a run via ``mlflow.sklearn.log_model``. This
       serialises the fitted estimator into the run's artifacts and returns a
       ``ModelInfo`` whose ``model_uri`` points at the logged model.
    2. **Register** that URI under ``registered_name``. The first call creates
       the registered model and version 1; later calls add versions 2, 3, ….
    3. **Promote** the new version to ``stage`` so ``models:/{name}/{stage}``
       resolves to it.

    Args:
        model: A *fitted, sklearn-compatible* estimator (has ``fit`` /
            ``predict`` / ``predict_proba``). For the GBM champion pass the
            underlying ``RetentionModel.pipeline`` (the sklearn ``Pipeline``),
            not the ``RetentionModel`` wrapper — ``mlflow.sklearn`` needs the
            sklearn object itself. LR/EBM champions would pass their pipeline
            directly. Typed ``Any`` because ``mlflow.sklearn`` accepts any
            sklearn-duck-typed object and over-constraining would reject valid
            estimators.
        run_name: Label for the registration run in the experiment view.
        registered_name: Registry entry name. Default ``"rp-champion"`` — the
            name the win condition and Loop 4 load path both reference.
        stage: Stage to promote the new version to. Default ``"Production"``.
        params: Optional params to log on the registration run (e.g. the
            champion's hyperparameters) for provenance. ``None`` → log none.
        metrics: Optional metrics to log (e.g. test AUC-PR / Brier) so the
            registry run carries the headline numbers. ``None`` → log none.
        pip_requirements: Optional explicit pip requirement list passed straight
            to ``log_model``. Passing it (e.g. ``["scikit-learn", "xgboost"]``)
            *skips* MLflow's default environment inference, which otherwise runs
            a ~20s ``uv``/``pip`` export on every call. Leave ``None`` in
            production to capture the true environment; set it in tests for speed.
        experiment_name: Experiment the registration run lands in. Default
            ``"rp-loop2"``; override for test isolation.

    Returns:
        ``(run_id, version)`` — the registration run's id and the integer
        registry version that was promoted. Returned so the notebook can echo
        "registered rp-champion v3 (run a1b2c3d4)" and tests can assert on it.

    Raises:
        mlflow.exceptions.MlflowException: if the store cannot be written or the
            model cannot be logged/registered.

    Teaching note — why ``transition_model_version_stage`` despite the warning:
        MLflow 2.9+ deprecated *stages* in favour of *aliases*
        (``set_registered_model_alias``). We deliberately use stages anyway: the
        win condition specifies ``rp-champion/Production`` (stage semantics), the
        registry UI renders the recognisable "Production" badge from a stage, and
        the call is still fully functional in our pinned MLflow. The deprecation
        warning is cosmetic. If a future major removes stages, switch to an alias
        and update the load URI to ``models:/rp-champion@production``.
    """
    # Same cwd-independence pin as log_run — see module docstring. The registry
    # URI defaults to the tracking URI for the file store, so this one call
    # points both the experiment store and the model registry at project root.
    mlflow.set_tracking_uri((config.PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        # name= (not the deprecated artifact_path=) is the MLflow 3.x spelling.
        # The returned model_info.model_uri (a models:/m-… URI) is what
        # register_model wants — passing runs:/…/model still works but emits a
        # resolution warning.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            pip_requirements=pip_requirements,
        )
        run_id = str(run.info.run_id)

    # Register + promote after the run closes — the logged model already exists
    # in the store, so these operate on a committed artifact.
    model_version = mlflow.register_model(model_info.model_uri, registered_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_name,
        version=model_version.version,
        stage=stage,
    )
    return run_id, int(model_version.version)


__all__ = ["log_run", "register_champion"]
