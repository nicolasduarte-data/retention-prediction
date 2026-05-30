"""Story 3.5 — Nested 5×5 cross-validation for unbiased AUC-PR estimation.

Why nested, not flat?
--------------------
In a **flat** (single-loop) cross-validation the same folds that *measure*
model performance are also used to *select* hyperparameters.  The reported
AUC-PR is therefore optimistic: the procedure found the best hyperparameters
*on* those folds, so evaluation on those same folds over-estimates how well
the model will generalise to truly unseen data.  The bias grows with the size
of the parameter grid and the variance of the dataset.

In **nested** cross-validation:

- The **outer loop** (5 folds here) provides an unbiased performance estimate.
  Outer test folds are *never* touched during hyperparameter selection.
- The **inner loop** (5 folds within each outer training set) tunes
  hyperparameters entirely on training data.  The outer test fold is invisible.
- The outer test score = retrained-on-best-params model evaluated on the fold
  it never saw.  Averaging the 5 outer scores → mean ± std AUC-PR.

Why GBM and not LR or EBM?
---------------------------
LR has one effective regularisation knob (C / class weight) and a smooth
loss surface — flat-CV variance is small enough to be inconsequential.  EBM
is computationally expensive to nest at this sample size and its intrinsic
regularisation (learning_rate × max_bins) is well-behaved.  GBM (XGBoost)
has a larger hyperparameter surface — depth × learning rate × subsampling —
and is known to overfit on small tabular HR datasets.  Nested CV is the
appropriate safeguard for a public portfolio claim about GBM's AUC-PR.

Reference: Cawley & Talbot (2010) "On Over-fitting in Model Selection and
Subsequent Selection Bias in Performance Evaluation", JMLR 11:2079–2107.

Usage::

    from retention.features.cohorts import extract_X_y, split_cohorts
    cohorts = split_cohorts(df)
    X, y = extract_X_y(cohorts["hybrid"], "hybrid")
    result = nested_cv_auc_pr(X, y, cohort="hybrid")
    print(result.summary())
    # → "GBM × hybrid nested 5×5 CV — AUC-PR: 0.302 ± 0.031  (scores: ...)"
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from retention import config
from retention.models.xgb import RetentionModel

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "PARAM_GRID",
    "NestedCVResult",
    "nested_cv_auc_pr",
]

# ── Hyperparameter grid ────────────────────────────────────────────────────────

# Deliberately small: the synthetic PA-warehouse dataset has ~900 training rows.
# Tuning 20+ hyperparameter combinations on ~720 inner-fold rows would select on
# noise, not signal.  The grid targets the two knobs that most affect GBM
# bias/variance on small tabular data:
#   - max_depth (bias): shallower trees underfit; deeper ones overfit on small n.
#   - learning_rate (regularisation proxy): lower lr relies on more trees;
#     combined with n_estimators it controls the effective regularisation.
# subsample + colsample_bytree stay at their RetentionModel defaults (0.8) —
# there is not enough data to usefully distinguish them from 1.0 here.
PARAM_GRID: list[dict[str, int | float]] = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10},
]
"""Hyperparameter grid for inner-loop selection (4 combinations).

Each dict is forwarded to ``RetentionModel(**params)`` as keyword arguments.
``scale_pos_weight`` is computed separately from the outer training fold and
merged in at runtime — it is not a grid axis because it is a data-derived
statistic, not a model choice.
"""


# ── Result type ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class NestedCVResult:
    """Immutable record from a nested cross-validation run.

    Frozen so the result cannot be accidentally mutated after the expensive
    multi-fold fit completes.  The ``summary()`` method produces a
    publication-ready one-liner for notebooks and progress logs.

    Attributes:
        scores: Per-fold AUC-PR values from the outer loop.
                Length equals ``outer_splits``.
        mean: Mean of outer-loop AUC-PR scores.
              Report this as the headline unbiased estimate.
        std: **Sample** standard deviation of outer-loop AUC-PR scores (ddof=1).
             Report this as the uncertainty band. Sample std (not population,
             ddof=0) is the conventional choice for a "mean ± std" band over a
             small sample of CV folds — it does not understate the spread.
        cohort: Cohort label — "hris_only" or "hybrid".
        outer_splits: Number of outer (evaluation) folds.
        inner_splits: Number of inner (hyperparameter-selection) folds.
    """

    scores: list[float]
    mean: float
    std: float
    cohort: str
    outer_splits: int
    inner_splits: int

    def summary(self) -> str:
        """One-line human-readable summary for notebooks and progress logs.

        Example output::

            GBM × hybrid nested 5×5 CV — AUC-PR: 0.302 ± 0.031
            (scores: 0.271, 0.318, 0.285, 0.341, 0.297)
        """
        score_str = ", ".join(f"{s:.3f}" for s in self.scores)
        return (
            f"GBM × {self.cohort} nested {self.outer_splits}×{self.inner_splits} CV — "
            f"AUC-PR: {self.mean:.3f} ± {self.std:.3f}  "
            f"(scores: {score_str})"
        )


# ── Private helpers ────────────────────────────────────────────────────────────


def _scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost's ``scale_pos_weight`` from a binary target Series.

    ``scale_pos_weight = n_negative / n_positive`` tells XGBoost to weight
    the minority class proportionally.  Computed on the **training fold only**
    — never on the test fold — to prevent class-balance leakage.

    Args:
        y: Binary target Series (1 = voluntary exit).

    Returns:
        Ratio ≥ 1.0 for imbalanced data with more 0 s than 1 s.

    Raises:
        ValueError: If there are no positive examples in ``y``.
    """
    # Cast to int first to guard against boolean Series (True/False) where
    # y == 1 would also be True for True, but the count is ambiguous in some
    # pandas versions.  Casting to int makes the comparison unambiguous.
    y_int = y.astype(int)
    n_pos = int((y_int == 1).sum())
    n_neg = int((y_int == 0).sum())
    if n_pos == 0:
        raise ValueError(
            "No positive examples (exits) in the training fold — "
            "cannot compute scale_pos_weight.  "
            "Use outer_splits ≤ n_positives so every fold retains at least one exit."
        )
    return float(n_neg) / float(n_pos)


def _inner_cv_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cohort: Literal["hris_only", "hybrid"],
    inner_splits: int,
    seed: int,
    spw: float,
) -> dict[str, int | float]:
    """Select the best hyperparameter combination via inner cross-validation.

    Runs ``inner_splits``-fold stratified CV on ``(X_train, y_train)`` for each
    entry in ``PARAM_GRID``.  Returns the combination with the highest mean
    inner-fold AUC-PR.

    Why ``spw`` (scale_pos_weight) comes from the outer training fold:
        ``scale_pos_weight`` is a data-derived statistic (neg/pos ratio).
        Recomputing it from the inner training fold would be more correct in
        principle, but at this dataset size the inner fold is only ~80% of an
        already small dataset — its class ratio is noisier than the outer
        fold's.  Using the outer fold's ratio is a pragmatic simplification
        that keeps the inner-loop class weighting stable across the 4 × inner_splits
        fits.  In production (large dataset, proper tuning budget), you would
        recompute per inner fold.

    Args:
        X_train: Outer fold's training features (no test-fold rows).
        y_train: Outer fold's training targets.
        cohort: Cohort label forwarded to RetentionModel.
        inner_splits: Number of inner folds.
        seed: Random state for StratifiedKFold.
        spw: scale_pos_weight computed from the outer training set.

    Returns:
        Best hyperparameter dict from PARAM_GRID (by mean inner AUC-PR).
    """
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        # Using the same seed as the outer loop is intentional: given a fixed
        # outer fold, the inner folds are also deterministic — no randomness
        # leaks between runs.
        random_state=seed,
    )

    # Start with the first candidate as the default in case all scores are equal.
    best_params: dict[str, int | float] = PARAM_GRID[0]
    best_score: float = float("-inf")

    for params in PARAM_GRID:
        fold_scores: list[float] = []

        for in_train_idx, in_val_idx in inner_cv.split(X_train, y_train):
            # StratifiedKFold.split() returns integer *positional* indices, not
            # label-based indices.  iloc (not loc) is therefore correct regardless
            # of whether X_train has a contiguous 0-based index — it often doesn't
            # after temporal_split + split_cohorts have sliced the original frame.
            X_in_train = X_train.iloc[in_train_idx]
            X_in_val = X_train.iloc[in_val_idx]
            y_in_train = y_train.iloc[in_train_idx]
            y_in_val = y_train.iloc[in_val_idx]

            # Merge scale_pos_weight into the candidate params.
            # RetentionModel.__init__ forwards all kwargs to XGBClassifier.
            fit_params: dict[str, int | float] = {**params, "scale_pos_weight": spw}
            model = RetentionModel(cohort=cohort, **fit_params)  # type: ignore[arg-type]
            model.fit(X_in_train, y_in_train)

            # predict_proba returns shape (n, 2); column 1 is P(exit=1).
            proba = model.predict_proba(X_in_val)[:, 1]
            score = float(average_precision_score(y_in_val, proba))
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params


# ── Public API ─────────────────────────────────────────────────────────────────


def nested_cv_auc_pr(
    X: pd.DataFrame,
    y: pd.Series,
    cohort: Literal["hris_only", "hybrid"],
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
    seed: int = config.SEED,
) -> NestedCVResult:
    """Nested cross-validation: outer AUC-PR estimate, inner hyperparameter selection.

    Runs ``outer_splits``-fold stratified CV.  Within each outer training fold,
    runs ``inner_splits``-fold CV over ``PARAM_GRID`` to select the best
    hyperparameters.  Retrains the chosen configuration on the full outer
    training fold; evaluates on the outer test fold.  Returns the distribution
    of outer AUC-PR scores.

    **Why StratifiedKFold?**
    With class imbalance (~20 % positive rate), random splits can concentrate
    exits in a single fold.  StratifiedKFold preserves the class ratio in every
    fold — each fold sees approximately the same proportion of exits.  This is
    critical when AUC-PR is the target metric: a fold with 0 positives would
    produce an undefined or zero AUC-PR that poisons the mean.

    **Why AUC-PR and not ROC-AUC?**
    AUC-PR is sensitive to the minority class; ROC-AUC is inflated by the
    large TN count in imbalanced settings.  AUC-PR is the project's primary
    metric throughout (ticket Win Conditions).

    **Leakage discipline:**
    ``scale_pos_weight`` is computed from the outer training fold only.
    The outer test fold is never touched until the final scoring step.
    The inner loop sees neither the outer test fold nor the original val/test
    splits — it works entirely within ``X_out_train``.

    Args:
        X: Feature DataFrame (same schema as ``extract_X_y`` output).
        y: Binary target Series (1 = voluntary exit).
        cohort: "hris_only" or "hybrid" — forwarded to RetentionModel.
        outer_splits: Number of outer evaluation folds.  Default: 5.
        inner_splits: Number of inner hyperparameter-selection folds.  Default: 5.
        seed: Random state for both StratifiedKFold instances.  Default: config.SEED (42).

    Returns:
        NestedCVResult with per-fold scores, mean, std, and metadata.

    Raises:
        ValueError: If ``outer_splits`` or ``inner_splits`` < 2.
        ValueError: If any outer training fold contains no positive examples.
    """
    if outer_splits < 2:
        raise ValueError(f"outer_splits must be ≥ 2, got {outer_splits}.")
    if inner_splits < 2:
        raise ValueError(f"inner_splits must be ≥ 2, got {inner_splits}.")

    # StratifiedKFold with shuffle=True + fixed seed → deterministic, reproducible folds.
    # shuffle=True is required when the data has temporal ordering (as ours does after
    # temporal_split): without shuffle, consecutive rows land in the same fold, biasing
    # the inner-loop evaluation.
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )

    outer_scores: list[float] = []

    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        # Step 1 — Partition into outer train / outer test.
        # iloc: StratifiedKFold returns *positional* indices; use iloc regardless of
        # the DataFrame's index values (which may not be 0-based after cohort splitting).
        X_out_train = X.iloc[outer_train_idx]
        X_out_test = X.iloc[outer_test_idx]
        y_out_train = y.iloc[outer_train_idx]
        y_out_test = y.iloc[outer_test_idx]

        # Step 2 — Compute scale_pos_weight from outer training fold only.
        # Leakage rule: any statistic that shapes the model must derive from training
        # data, not the test fold it will be scored on.
        spw = _scale_pos_weight(y_out_train)

        # Step 3 — Inner loop: pick best hyperparameters on outer_train only.
        # The outer test fold is *invisible* to this step.
        best_params = _inner_cv_best_params(
            X_out_train, y_out_train, cohort, inner_splits, seed, spw
        )

        # Step 4 — Retrain with best params on the full outer training set.
        # We retrain on *all* outer training rows (not just the inner-fold subsets)
        # to give the final estimator the maximum available training signal.
        fit_params: dict[str, int | float] = {**best_params, "scale_pos_weight": spw}
        model = RetentionModel(cohort=cohort, **fit_params)  # type: ignore[arg-type]
        model.fit(X_out_train, y_out_train)

        # Step 5 — Score on the outer test fold — the only number that counts.
        # This fold was invisible during both the inner loop (hyperparameter selection)
        # and the outer retraining step, so the score is uncontaminated.
        proba = model.predict_proba(X_out_test)[:, 1]
        score = float(average_precision_score(y_out_test, proba))
        outer_scores.append(score)

    # np.mean / np.std on a list → numpy scalars; float() casts ensure plain Python
    # floats in the dataclass (cleaner repr, no numpy dtype leaking into notebooks).
    # ddof=1 → sample std (feast T2-SEL-1): the conventional band for a small
    # sample of CV folds; population std (ddof=0) understates the spread by √(k/(k−1)).
    return NestedCVResult(
        scores=outer_scores,
        mean=float(np.mean(outer_scores)),
        std=float(np.std(outer_scores, ddof=1)),
        cohort=cohort,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
    )
