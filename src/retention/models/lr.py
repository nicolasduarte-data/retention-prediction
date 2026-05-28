"""Logistic Regression baseline — Story 2.2.

Implements `train_lr()`: the LR arm of the Loop 2 three-model comparison.

Design choices documented here because they will be cited in the Loop 2
comparison write-up and in docs/methodology.md:

**class_weight='balanced'**
    The training set has ~18.8% voluntary exits (positive class). Without
    weight adjustment, LR's loss function optimises accuracy on the dominant
    class (still active), producing a model that rarely predicts exit. Balanced
    weighting sets w_i = n_samples / (2 * n_class_i), penalising misclassification
    of exits proportionally to their rarity. This is the FLIP-RISK mitigation
    (Loop 1 risk register): if we under-detect exits we ship the wrong model.

**No SMOTE**
    SMOTE oversampling violates the temporal ordering assumption (synthetic
    minority samples span the temporal split boundary). Balanced class_weight
    achieves the same effect without touching the data. Enforced by the
    `test_no_smote_in_pipeline` test in test_models.py.

**Solver: lbfgs (default for multiclass=False)**
    Works for binary classification with balanced weights. saga would be
    faster for very large datasets, but 1,274 rows doesn't warrant it.

**max_iter=1000**
    The default 100 sometimes fails to converge on imbalanced data — 1,000
    is generous for this dataset size and eliminates convergence warnings.

**Preprocessing: build_preprocessor(cohort=cohort)**
    Uses the cohort-aware preprocessor introduced in Loop 2 (Story 1.3).
    - 'hris_only' → 10 HRIS features, OHE for categoricals.
    - 'hybrid'    → 13 features (adds survey signal), same routing.

    LR and GBM share this OHE preprocessor. EBM uses native categorical
    handling — see docs/methodology.md → Cross-Model Comparison Methodology
    for why this is an acknowledged tradeoff, not a hidden one.

Story 2.2 — see `backlog/epic-2-baseline-models.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from retention import config
from retention.features.preprocessing import build_preprocessor

if TYPE_CHECKING:
    import pandas as pd


def train_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cohort: Literal["hris_only", "hybrid"] = "hybrid",
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = config.SEED,
) -> Pipeline:
    """Train a Logistic Regression retention model.

    Fits a preprocessor on X_train (no val leakage), then fits LR on the
    transformed training features. Returns a fully fitted sklearn Pipeline
    that can call `.predict_proba(X_raw)` directly on raw DataFrames.

    Args:
        X_train: Raw feature DataFrame matching the cohort's column set.
            For cohort='hris_only': 10 HRIS feature columns.
            For cohort='hybrid': 13 feature columns (HRIS + survey).
            Pass `cohorts[cohort].drop(...)` from `split_cohorts()`.
        y_train: Binary target (1 = voluntary exit, 0 = still active).
        cohort: Which feature set X_train was built from. Controls which
            columns the preprocessor selects. Must match the DataFrame's
            actual columns — mismatch raises KeyError at fit_transform time.
        C: Inverse regularisation strength. Smaller = stronger L2 penalty.
            Default 1.0 (untuned); hyperparameter search is Story 3.3 / Loop 3.
        max_iter: Convergence budget. 1000 prevents ConvergenceWarning on
            imbalanced class distributions.
        random_state: For reproducibility (seeds lbfgs initialisation path).

    Returns:
        Fitted sklearn Pipeline with steps:
            - 'preprocessor': ColumnTransformer (StandardScaler + OHE + bool cast)
            - 'classifier':   LogisticRegression(class_weight='balanced')
        Call `.predict_proba(X_raw)[:, 1]` to get P(voluntary_exit=1).

    Teaching note — sklearn Pipeline.predict_proba flow:
        When you call `pipeline.predict_proba(X_raw)`:
        1. `pipeline.transform(X_raw)` calls each step EXCEPT the last in order.
           Here that's just the preprocessor (StandardScaler + OHE + bool cast).
        2. The final step's `.predict_proba()` runs on the transformed output.
        The preprocessor was fitted on X_train only — no val/test data leakage.
    """
    preprocessor = build_preprocessor(cohort=cohort)
    X_train_t = preprocessor.fit_transform(X_train)

    lr = LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train_t, y_train.to_numpy())

    # Assemble pre-fitted steps into a Pipeline so the caller can call
    # pipeline.predict_proba(X_raw) without a separate transform step.
    # sklearn uses transform() (not fit_transform()) on non-final steps
    # inside predict_proba — safe because preprocessor is already fitted.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lr),
        ]
    )


__all__ = ["train_lr"]
