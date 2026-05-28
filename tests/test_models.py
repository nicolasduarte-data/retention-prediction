"""Story 2.6 — Wrapper tests for all three model types (LR, GBM, EBM).

Coverage:
  2.6.1  test_lr_fits_on_tiny_data          — train_lr() returns fitted Pipeline
  2.6.2  test_gbm_uses_scale_pos_weight     — RetentionModel passes scale_pos_weight to XGBClassifier
  2.6.3  test_no_smote_in_pipeline          — neither LR nor GBM pipeline contains SMOTE
  2.6.4  test_ebm_fits                      — added in Story 2.4 (EBM not yet written)

Design note — why these tests live here rather than test_model.py:
  test_model.py covers the XGBoost wrapper in depth (9 tests). This file
  covers cross-model behavioral guarantees (no SMOTE, scale_pos_weight) and
  the two new model types (LR, EBM). Keeping them separate makes the test
  tree mirror the models/ package structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from retention.models.ebm import train_ebm
from retention.models.lr import train_lr
from retention.models.xgb import RetentionModel


# ------------------------------------------------------------------ #
# Shared fixture                                                        #
# ------------------------------------------------------------------ #


@pytest.fixture()
def tiny_data() -> tuple[pd.DataFrame, pd.Series]:
    """Minimal synthetic dataset: 30 rows, hybrid feature set (13 cols).

    30 rows is enough for LR / GBM to fit without convergence issues.
    All survey columns are present so the hybrid preprocessor doesn't raise.
    The hris_only tests pass a 10-column slice of this fixture.
    """
    rng = np.random.default_rng(42)
    n = 30

    X = pd.DataFrame(
        {
            # HRIS features (10)
            "tenure_months": rng.uniform(1, 60, n),
            "compa_ratio": rng.uniform(0.7, 1.3, n),
            "successor_count": rng.integers(0, 4, n).astype(float),
            "age_at_window_close": rng.uniform(22, 60, n),
            "performance_tier": rng.choice(["2", "3", "4", "5"], n),
            "gender": rng.choice(["M", "F", "NB"], n),
            "is_critical_role": rng.choice([True, False], n),
            # survey features (3)
            "enps": rng.uniform(-100, 100, n),
            "engagement_score": rng.uniform(1, 5, n),
            "manager_relationship_score": rng.uniform(1, 5, n),
        }
    )
    # ~25% exit rate
    prob = 0.1 + 0.3 * (X["compa_ratio"] < 0.9).astype(float)
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="voluntary_exit_label")
    return X, y


@pytest.fixture()
def hris_cols() -> list[str]:
    """The 10 HRIS-only feature columns (no survey signal)."""
    return [
        "tenure_months",
        "compa_ratio",
        "successor_count",
        "age_at_window_close",
        "performance_tier",
        "gender",
        "is_critical_role",
    ]


# ------------------------------------------------------------------ #
# Story 2.6.1 — LR wrapper (train_lr)                                  #
# ------------------------------------------------------------------ #


def test_lr_fits_on_tiny_data(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_lr() must return a fitted Pipeline without raising."""
    X, y = tiny_data
    pipeline = train_lr(X, y)
    assert isinstance(pipeline, Pipeline), "train_lr must return an sklearn Pipeline"


def test_lr_pipeline_has_preprocessor_and_classifier(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Returned Pipeline must have exactly two named steps: preprocessor + classifier."""
    X, y = tiny_data
    pipeline = train_lr(X, y)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["preprocessor", "classifier"], (
        f"Expected ['preprocessor', 'classifier'], got {step_names}"
    )


def test_lr_predict_proba_shape(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """predict_proba(X) must return shape (n_samples, 2)."""
    X, y = tiny_data
    pipeline = train_lr(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected ({len(X)}, 2), got {proba.shape}"


def test_lr_predict_proba_values_in_unit_interval(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """All probabilities must be in [0, 1]."""
    X, y = tiny_data
    pipeline = train_lr(X, y)
    proba = pipeline.predict_proba(X)
    assert (proba >= 0.0).all(), "Negative probability encountered."
    assert (proba <= 1.0).all(), "Probability > 1 encountered."


def test_lr_row_probabilities_sum_to_one(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Each row's probabilities must sum to 1.0 within floating-point tolerance."""
    X, y = tiny_data
    pipeline = train_lr(X, y)
    proba = pipeline.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_lr_class_weight_is_balanced(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """LR classifier must use class_weight='balanced' — FLIP-RISK mitigation.

    The FLIP-RISK (Loop 1 risk register) states that under-detecting exits
    causes the wrong model to ship. Balanced weighting is the agreed mitigation.
    If this test fails it means someone changed the class_weight — that requires
    an explicit risk-register update, not a silent config change.
    """
    X, y = tiny_data
    pipeline = train_lr(X, y)
    clf = pipeline.named_steps["classifier"]
    assert clf.class_weight == "balanced", (
        f"Expected class_weight='balanced', got '{clf.class_weight}'. "
        "Changing this requires a risk-register update (FLIP-RISK, Loop 1)."
    )


def test_lr_hris_only_cohort(
    tiny_data: tuple[pd.DataFrame, pd.Series],
    hris_cols: list[str],
) -> None:
    """train_lr with cohort='hris_only' must fit on a 10-column DataFrame without KeyError."""
    X_full, y = tiny_data
    # Only pass the 7 HRIS columns present in the fixture (plus booleans already there)
    # The actual hris_only cohort has 10 features — we use the fixture's subset.
    # What matters: no survey columns (enps, engagement_score, manager_relationship_score).
    X_hris = X_full[hris_cols]
    pipeline = train_lr(X_hris, y, cohort="hris_only")
    proba = pipeline.predict_proba(X_hris)
    assert proba.shape == (len(X_hris), 2)


# ------------------------------------------------------------------ #
# Story 2.6.2 — GBM scale_pos_weight passthrough                       #
# ------------------------------------------------------------------ #


def test_gbm_uses_scale_pos_weight(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """RetentionModel must pass scale_pos_weight to XGBClassifier when provided.

    scale_pos_weight is an alternative imbalance strategy to class_weight='balanced'.
    We expose it so the Loop 2 comparison notebook can test both approaches.
    Verifying passthrough prevents a silent misconfiguration where the parameter
    is accepted but never reaches the underlying estimator.
    """
    X, y = tiny_data
    pos_weight = 4.0
    model = RetentionModel(scale_pos_weight=pos_weight)
    model.fit(X, y)
    assert model.pipeline is not None
    xgb_clf = model.pipeline.named_steps["classifier"]
    actual = xgb_clf.get_params().get("scale_pos_weight")
    assert actual == pos_weight, (
        f"scale_pos_weight not passed through to XGBClassifier: expected {pos_weight}, got {actual}"
    )


# ------------------------------------------------------------------ #
# Story 2.6.3 — No SMOTE in any pipeline                               #
# ------------------------------------------------------------------ #


def test_no_smote_in_pipeline(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Neither the LR nor GBM pipeline must contain any SMOTE step.

    SMOTE oversampling violates the temporal ordering assumption — synthetic
    minority samples can cross the temporal split boundary, leaking future
    signal into training. Balanced class_weight / scale_pos_weight achieves
    the same imbalance correction without touching the data distribution.

    This test guards against accidental reintroduction of SMOTE during
    hyperparameter search (Story 3.3) or refactoring.
    """
    X, y = tiny_data

    # LR pipeline
    lr_pipeline = train_lr(X, y)
    for step_name, estimator in lr_pipeline.steps:
        estimator_class = type(estimator).__name__.lower()
        assert "smote" not in estimator_class, (
            f"LR pipeline step '{step_name}' ({type(estimator).__name__}) looks like SMOTE. "
            "Remove it — SMOTE violates temporal ordering. Use class_weight='balanced' instead."
        )

    # GBM pipeline
    gbm_model = RetentionModel()
    gbm_model.fit(X, y)
    assert gbm_model.pipeline is not None
    for step_name, estimator in gbm_model.pipeline.steps:
        estimator_class = type(estimator).__name__.lower()
        assert "smote" not in estimator_class, (
            f"GBM pipeline step '{step_name}' ({type(estimator).__name__}) looks like SMOTE. "
            "Remove it — SMOTE violates temporal ordering. Use scale_pos_weight instead."
        )


# ------------------------------------------------------------------ #
# Story 2.6.4 — EBM wrapper (train_ebm)                                #
# ------------------------------------------------------------------ #


def test_ebm_fits(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_ebm() must return a fitted Pipeline without raising."""
    X, y = tiny_data
    pipeline = train_ebm(X, y)
    assert isinstance(pipeline, Pipeline), "train_ebm must return an sklearn Pipeline"


def test_ebm_pipeline_has_preprocessor_and_classifier(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """EBM pipeline must have exactly two named steps: preprocessor + classifier."""
    X, y = tiny_data
    pipeline = train_ebm(X, y)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["preprocessor", "classifier"], (
        f"Expected ['preprocessor', 'classifier'], got {step_names}"
    )


def test_ebm_predict_proba_shape(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """EBM predict_proba(X) must return shape (n_samples, 2)."""
    X, y = tiny_data
    pipeline = train_ebm(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected ({len(X)}, 2), got {proba.shape}"


def test_ebm_predict_proba_values_in_unit_interval(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """All EBM probabilities must be in [0, 1]."""
    X, y = tiny_data
    pipeline = train_ebm(X, y)
    proba = pipeline.predict_proba(X)
    assert (proba >= 0.0).all(), "Negative probability encountered."
    assert (proba <= 1.0).all(), "Probability > 1 encountered."


def test_ebm_row_probabilities_sum_to_one(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Each row's EBM probabilities must sum to 1.0 within floating-point tolerance."""
    X, y = tiny_data
    pipeline = train_ebm(X, y)
    proba = pipeline.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ebm_preprocessor_uses_no_ohe(
    tiny_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """EBM preprocessor must not contain OneHotEncoder — EBM handles categoricals natively.

    This is the key architectural distinction between the EBM and LR/GBM models.
    If OHE is accidentally added to the EBM pipeline (e.g. by reusing
    build_preprocessor() instead of build_ebm_preprocessor()), EBM would treat
    each one-hot column as an independent binary feature rather than a level of
    the original categorical. This destroys the native categorical advantage.
    """
    from sklearn.preprocessing import OneHotEncoder

    X, y = tiny_data
    pipeline = train_ebm(X, y)
    preprocessor = pipeline.named_steps["preprocessor"]

    for _name, transformer, _cols in preprocessor.transformers_:
        # Walk into nested Pipelines (e.g. the boolean step)
        steps_to_check = (
            [("inner", transformer)] if not hasattr(transformer, "steps") else transformer.steps
        )
        for _step_name, step_obj in steps_to_check:
            assert not isinstance(step_obj, OneHotEncoder), (
                "EBM preprocessor step contains OneHotEncoder. "
                "Use build_ebm_preprocessor() (no OHE) — not build_preprocessor() — "
                "so EBM can handle categoricals natively."
            )


def test_ebm_hris_only_cohort(
    tiny_data: tuple[pd.DataFrame, pd.Series],
    hris_cols: list[str],
) -> None:
    """train_ebm with cohort='hris_only' must fit on a 10-column DataFrame without KeyError."""
    X_full, y = tiny_data
    X_hris = X_full[hris_cols]
    pipeline = train_ebm(X_hris, y, cohort="hris_only")
    proba = pipeline.predict_proba(X_hris)
    assert proba.shape == (len(X_hris), 2)
