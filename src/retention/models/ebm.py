"""Explainable Boosting Machine (EBM) retention model — Story 2.4.

Implements `train_ebm()`: the EBM arm of the Loop 2 three-model comparison.

Design choices documented here — they are cited in the Loop 2 comparison
write-up and in docs/methodology.md → Cross-Model Comparison Methodology:

**Native categorical handling (no OHE preprocessor for categoricals)**
    EBM's GAM internally buckets feature values into bins. For categorical
    columns (performance_tier, gender) this means each unique string value
    gets its own bin and its own shape function term — equivalent to learning
    a separate effect per category level without the dummy-variable explosion
    that OHE causes for high-cardinality features.
    LR and GBM use build_preprocessor() (OHE). EBM uses build_ebm_preprocessor()
    (no OHE). This is an acknowledged tradeoff: not a hidden inconsistency.
    Documented in docs/methodology.md → Cross-Model Comparison Methodology.

**No StandardScaler**
    EBM is a boosted GAM (tree-based learner). It is invariant to monotone
    transforms of numeric features — scaling changes nothing about the fitted
    model output. Omitting it reduces pipeline complexity and makes EBM global
    explanations easier to interpret (raw feature units in SHAP plots).

**Imbalance handling: compute_sample_weight('balanced')**
    ExplainableBoostingClassifier does not expose class_weight='balanced' in
    all versions of interpret. The universal sklearn-compatible approach is
    compute_sample_weight('balanced', y), which computes w_i = n / (2 * n_class_i)
    and passes it to ebm.fit(sample_weight=...). This is mathematically
    identical to class_weight='balanced' in LR. The FLIP-RISK mitigation
    (Loop 1 risk register) is satisfied — exits are up-weighted relative to
    their rarity.

**interactions=10**
    EBM's interaction detector (FAST algorithm) finds and fits the top-k
    pairwise interaction terms. 10 is the library default — enough to capture
    real interactions on a 1,274-row dataset without overfitting. Loop 3
    hyperparameter search will tune this.

**max_bins=256**
    Default — 256 bins per continuous feature. Generous for this dataset size.
    Controls granularity of the piecewise-linear shape functions.

**outer_bags=8**
    Default — EBM uses bagging internally for variance reduction. 8 bags is
    the library default; increasing to 16–32 improves stability at the cost
    of training time. Held at 8 for the baseline run.

Story 2.4 scope — see `backlog/epic-2-baseline-models.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from retention import config
from retention.features.preprocessing import build_ebm_preprocessor

if TYPE_CHECKING:
    import pandas as pd


def train_ebm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cohort: Literal["hris_only", "hybrid"] = "hybrid",
    *,
    interactions: int = 10,
    max_bins: int = 256,
    outer_bags: int = 8,
    random_state: int = config.SEED,
) -> Pipeline:
    """Train an EBM retention model.

    Fits an imputation-only preprocessor on X_train (no OHE, no scaling),
    computes balanced sample weights, then fits ExplainableBoostingClassifier
    with native categorical handling. Returns a fitted sklearn Pipeline that
    can call `.predict_proba(X_raw)` directly on raw DataFrames.

    The preprocessor uses `set_output(transform='pandas')` so EBM receives a
    DataFrame from the Pipeline's transform step — column names and string
    dtypes are preserved, enabling EBM's native categorical bin detection.

    Args:
        X_train: Raw feature DataFrame matching the cohort's column set.
            For cohort='hris_only': 10 HRIS feature columns (no survey).
            For cohort='hybrid': 13 feature columns (HRIS + survey).
        y_train: Binary target (1 = voluntary exit, 0 = still active).
        cohort: Which feature set X_train was built from. Controls which
            columns the EBM preprocessor selects. Must match the DataFrame's
            actual columns — mismatch raises KeyError at fit_transform time.
        interactions: Number of pairwise interaction terms EBM's FAST algorithm
            will detect and fit. Default 10 (library default). Tune in Loop 3.
        max_bins: Maximum bins per continuous feature in the piecewise-linear
            shape functions. Default 256. Higher = finer-grained but slower.
        outer_bags: Bagging rounds for variance reduction. Default 8. Increase
            to 16–32 for more stable shape functions at higher compute cost.
        random_state: Seeds EBM's random bagging and interaction search.

    Returns:
        Fitted sklearn Pipeline with steps:
            - 'preprocessor': ColumnTransformer (imputation-only, no OHE,
                               set_output='pandas' for DataFrame passthrough)
            - 'classifier':   ExplainableBoostingClassifier with balanced
                               sample weights baked into the fitted model
        Call `.predict_proba(X_raw)[:, 1]` to get P(voluntary_exit=1).
        Call `.named_steps['classifier'].explain_global()` for global
        explanation (shape functions + interaction heatmaps).

    Teaching note — EBM shape functions:
        EBM learns one smooth function f_j(x_j) per feature and one 2D
        surface f_ij(x_i, x_j) per interaction term. The log-odds prediction
        is the sum: score = intercept + Σ f_j(x_j) + Σ f_ij(x_i, x_j).
        Because each term is additive and independent of the others, the shape
        function for 'compa_ratio' is exactly interpretable as "the effect of
        compa_ratio holding everything else constant" — unlike a tree ensemble
        where interactions are implicit and mixed.

    Teaching note — why sample_weight vs class_weight:
        sklearn estimators that support class imbalance expose it either as a
        class_weight parameter (LR, RandomForest) or as sample_weight in fit().
        EBM uses the latter. compute_sample_weight('balanced', y) computes
        w_i = n_samples / (n_classes * n_i_class), then passes per-sample
        weights to the internal loss minimization. The effect is identical to
        class_weight='balanced' — minority-class errors are penalised in
        proportion to their rarity.
    """
    preprocessor = build_ebm_preprocessor(cohort=cohort)
    X_train_t = preprocessor.fit_transform(X_train)

    # Balanced sample weights: FLIP-RISK mitigation for EBM.
    # compute_sample_weight is mathematically equivalent to class_weight='balanced'
    # in LR — exits are up-weighted proportionally to their rarity (~18.8%).
    y_train_arr = y_train.to_numpy()
    sample_weights = compute_sample_weight("balanced", y_train_arr)

    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        max_bins=max_bins,
        outer_bags=outer_bags,
        random_state=random_state,
    )
    ebm.fit(X_train_t, y_train_arr, sample_weight=sample_weights)

    # Assemble pre-fitted steps into a Pipeline. The preprocessor's
    # set_output("pandas") persists through Pipeline.transform() calls, so
    # EBM always receives a DataFrame with column names — native categorical
    # handling works end-to-end on raw DataFrames.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ebm),
        ]
    )


__all__ = ["train_ebm"]
