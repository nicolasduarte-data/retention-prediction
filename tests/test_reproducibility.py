"""Story 2.8 — Reproducibility smoke test.

Verifies that the full training pipeline is bitwise-deterministic: given the
same input DataFrame, the same model type, and the same random seed, two
independent ``fit()`` → ``predict_proba()`` passes produce exactly equal
output arrays.

Why this matters for the portfolio:

    A senior PA reviewer will run ``make train`` on their machine and compare
    the AUC-PR numbers to the README table. If the numbers don't match — even
    by a tiny rounding artifact — they'll either distrust the project or file
    an issue. Reproducibility is a senior signal; most public HR repos don't
    ship it at all.

What ``SEED = 42`` buys us:

    Each model wrapper passes ``random_state=config.SEED`` to the underlying
    estimator (LogisticRegression, XGBClassifier, ExplainableBoostingClassifier).
    This controls the stochastic components:

    - LR (lbfgs): single-threaded, so seed controls initialisation path.
    - GBM (XGBoost): seeds the subsampling RNG and tree construction order.
    - EBM: seeds the outer-bag sampling and interaction search.

    NumPy's global seed is set by ``config.set_global_seed()`` before training.
    Without it, numpy operations (e.g. array shuffling inside sklearn's
    ColumnTransformer) would differ across runs.

What this test does NOT cover:

    - ``hash(str)`` determinism in the current process. That requires
      ``PYTHONHASHSEED=42`` to be set *before* the interpreter starts.
      The Makefile ``repro`` target does this: ``PYTHONHASHSEED=42 uv run pytest``.
      Run ``make repro`` for the full determinism envelope.
    - Multi-process / distributed training (no deep learning per Grinsztajn 2022).

Design choice — bitwise equality vs. tolerance:

    We use ``np.testing.assert_array_equal`` (exact equality) rather than
    ``assert_allclose`` (tolerance). Floating-point outputs from
    deterministic code should be bitwise-identical: if they're not, something
    in the randomisation chain is non-deterministic, and that's a bug to fix,
    not a tolerance to loosen.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from retention import config
from retention.data.split import temporal_split
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.models.ebm import train_ebm
from retention.models.lr import train_lr
from retention.models.xgb import RetentionModel

# Type alias for the two valid cohort names — mirrors what the model wrappers
# expect (Literal['hris_only', 'hybrid']). Using Literal here lets mypy verify
# that no invalid cohort name can be passed through to the model constructors.
CohortName = Literal["hris_only", "hybrid"]


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #


def _train_and_predict_lr(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    cohort: CohortName,
) -> npt.NDArray[np.float64]:
    """Fit LR, return positive-class probabilities on X_val."""
    config.set_global_seed()
    pipeline = train_lr(X_tr, y_tr, cohort=cohort)
    result: npt.NDArray[np.float64] = pipeline.predict_proba(X_val)[:, 1]
    return result


def _train_and_predict_gbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    cohort: CohortName,
) -> npt.NDArray[np.float64]:
    """Fit GBM (XGBoost), return positive-class probabilities on X_val."""
    config.set_global_seed()
    model = RetentionModel(cohort=cohort)
    model.fit(X_tr, y_tr)
    result: npt.NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
    return result


def _train_and_predict_ebm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    cohort: CohortName,
) -> npt.NDArray[np.float64]:
    """Fit EBM, return positive-class probabilities on X_val."""
    config.set_global_seed()
    pipeline = train_ebm(X_tr, y_tr, cohort=cohort)
    result: npt.NDArray[np.float64] = pipeline.predict_proba(X_val)[:, 1]
    return result


# ------------------------------------------------------------------ #
# Story 2.8 — Bitwise reproducibility per model type                   #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("cohort", ["hris_only", "hybrid"])
def test_lr_predictions_are_reproducible(
    synthetic_attrition_df: pd.DataFrame,
    cohort: str,
) -> None:
    """LR predictions must be bitwise-identical across two training runs.

    LR uses lbfgs with ``random_state=config.SEED`` + ``set_global_seed()``
    to seed numpy before fit. Both calls must produce exactly equal
    predicted probabilities — no tolerance, exact equality.

    Parameterised over both cohorts: hris_only (7 features) and hybrid
    (10 features) exercise different preprocessing branches.
    """
    # pytest.mark.parametrize produces plain str; cast to the Literal type
    # the model wrappers expect so mypy --strict is satisfied.
    cohort_name = cast(CohortName, cohort)

    train_df, val_df, _ = temporal_split(synthetic_attrition_df, save_indices=False)
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    X_tr, y_tr = extract_X_y(train_cohorts[cohort_name], cohort_name)
    X_val, _ = extract_X_y(val_cohorts[cohort_name], cohort_name)

    # Two independent training runs with the same seed
    proba_run1 = _train_and_predict_lr(X_tr, y_tr, X_val, cohort_name)
    proba_run2 = _train_and_predict_lr(X_tr, y_tr, X_val, cohort_name)

    np.testing.assert_array_equal(
        proba_run1,
        proba_run2,
        err_msg=(
            f"[LR/{cohort}] Predictions differ between run 1 and run 2. "
            "set_global_seed() must be called before every fit(). "
            "Ensure train_lr() does not rely on external random state."
        ),
    )


@pytest.mark.parametrize("cohort", ["hris_only", "hybrid"])
def test_gbm_predictions_are_reproducible(
    synthetic_attrition_df: pd.DataFrame,
    cohort: str,
) -> None:
    """GBM (XGBoost) predictions must be bitwise-identical across two training runs.

    XGBoost is seeded via ``random_state=config.SEED`` passed to XGBClassifier.
    ``set_global_seed()`` additionally seeds numpy for any numpy-backed
    operations in the preprocessing pipeline (ColumnTransformer, SimpleImputer).
    Both calls must produce exactly equal predicted probabilities.
    """
    cohort_name = cast(CohortName, cohort)

    train_df, val_df, _ = temporal_split(synthetic_attrition_df, save_indices=False)
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    X_tr, y_tr = extract_X_y(train_cohorts[cohort_name], cohort_name)
    X_val, _ = extract_X_y(val_cohorts[cohort_name], cohort_name)

    proba_run1 = _train_and_predict_gbm(X_tr, y_tr, X_val, cohort_name)
    proba_run2 = _train_and_predict_gbm(X_tr, y_tr, X_val, cohort_name)

    np.testing.assert_array_equal(
        proba_run1,
        proba_run2,
        err_msg=(
            f"[GBM/{cohort}] Predictions differ between run 1 and run 2. "
            "XGBoost requires random_state to be passed explicitly. "
            "Check RetentionModel.__init__() passes random_state=config.SEED."
        ),
    )


@pytest.mark.parametrize("cohort", ["hris_only", "hybrid"])
def test_ebm_predictions_are_reproducible(
    synthetic_attrition_df: pd.DataFrame,
    cohort: str,
) -> None:
    """EBM predictions must be bitwise-identical across two training runs.

    EBM uses random bagging (outer_bags=8) and interaction search — both
    seeded via ``random_state=config.SEED``. Without the seed, EBM's bag
    composition changes each run, producing slightly different predictions.
    This test catches any future change to train_ebm() that drops the seed.
    """
    cohort_name = cast(CohortName, cohort)

    train_df, val_df, _ = temporal_split(synthetic_attrition_df, save_indices=False)
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    X_tr, y_tr = extract_X_y(train_cohorts[cohort_name], cohort_name)
    X_val, _ = extract_X_y(val_cohorts[cohort_name], cohort_name)

    proba_run1 = _train_and_predict_ebm(X_tr, y_tr, X_val, cohort_name)
    proba_run2 = _train_and_predict_ebm(X_tr, y_tr, X_val, cohort_name)

    np.testing.assert_array_equal(
        proba_run1,
        proba_run2,
        err_msg=(
            f"[EBM/{cohort}] Predictions differ between run 1 and run 2. "
            "EBM uses random bagging; random_state must be passed to "
            "ExplainableBoostingClassifier. Check train_ebm() signature."
        ),
    )


# ------------------------------------------------------------------ #
# Cross-model seed isolation                                            #
# ------------------------------------------------------------------ #


def test_seed_reset_between_models_is_sufficient(
    synthetic_attrition_df: pd.DataFrame,
) -> None:
    """Seeding before each model is enough: LR → GBM sequence must equal GBM alone.

    This test verifies that ``set_global_seed()`` correctly resets the
    global numpy/random state before each training call. If the seed is
    set once globally and not reset, the second model in a sequence would
    consume a different numpy state than it would if trained in isolation.

    Concretely: training GBM after LR must produce the same predictions
    as training GBM in isolation (with a fresh seed).

    This models the notebook execution pattern: the comparison notebook
    trains 6 models sequentially. Each must be reproducible independently
    of training order.
    """
    train_df, val_df, _ = temporal_split(synthetic_attrition_df, save_indices=False)
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    X_tr, y_tr = extract_X_y(train_cohorts["hybrid"], "hybrid")
    X_val, _ = extract_X_y(val_cohorts["hybrid"], "hybrid")

    # Train LR first (consumes numpy RNG state)
    _train_and_predict_lr(X_tr, y_tr, X_val, "hybrid")

    # Now train GBM — set_global_seed() must have reset the state
    proba_gbm_after_lr = _train_and_predict_gbm(X_tr, y_tr, X_val, "hybrid")

    # GBM trained in isolation (clean seed from scratch)
    proba_gbm_isolated = _train_and_predict_gbm(X_tr, y_tr, X_val, "hybrid")

    np.testing.assert_array_equal(
        proba_gbm_after_lr,
        proba_gbm_isolated,
        err_msg=(
            "GBM predictions differ when trained after LR vs in isolation. "
            "set_global_seed() must fully reset numpy state before each model. "
            "This means: np.random.seed(seed) is called inside each helper."
        ),
    )
