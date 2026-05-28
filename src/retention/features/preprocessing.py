"""Preprocessing pipeline — Story 2.1 (cohort-aware update — Story 1.3 / Loop 2).

Builds a sklearn ColumnTransformer driven by FEATURE_CATALOG roles, dtypes,
and — as of Loop 2 — the cohort the model is being trained for.

Column routing:
  numeric       → SimpleImputer(median) → StandardScaler
  categorical   → SimpleImputer(constant='__missing__') → OneHotEncoder
  boolean       → FunctionTransformer(cast to float64, preserves NaN-free booleans)
  date/string   → dropped (snapshot_date is temporal_anchor; no free-text features)
  identifier    → dropped (employee_id never enters the model)
  temporal_anchor → dropped
  label         → dropped (y is handled separately by the caller)

**Cohort awareness (Loop 2):**
  build_preprocessor(cohort='hris_only') → selects 10 HRIS feature columns only.
  build_preprocessor(cohort='hybrid')    → selects all 13 feature columns (default).

  The default is 'hybrid' for backward compatibility with Loop 1 tests and the
  existing XGBoost wrapper. HRIS-only training passes cohort='hris_only'
  explicitly so the ColumnTransformer never tries to select survey columns
  that are absent from the hris_only cohort DataFrame.

  Why not just use remainder='drop' and pass the full 13-col DataFrame to both?
      It works, but it obscures intent — a reviewer reading the code can't tell
      whether the absence of survey columns in the HRIS-only model is deliberate
      or an accident. Explicit cohort selection makes the experiment legible.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from retention.features.catalog import FEATURE_CATALOG
from retention.features.cohorts import get_cohort_feature_names


def _cast_bool_to_float(X: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Cast boolean columns to float64 for sklearn compatibility."""
    return X.astype(np.float64)


def build_preprocessor(
    cohort: Literal["hris_only", "hybrid"] = "hybrid",
) -> ColumnTransformer:
    """Return a fitted-on-demand ColumnTransformer built from FEATURE_CATALOG.

    The transformer is NOT fitted here — call `.fit_transform(X_train)` and
    `.transform(X_val)` / `.transform(X_test)` in the training pipeline.

    Args:
        cohort: Which feature set to build for.
            - 'hybrid'    (default): all 13 features — HRIS + survey signal.
            - 'hris_only': 10 HRIS features only; survey columns excluded.
            The cohort must match the DataFrame being passed to fit_transform —
            pass cohorts['hris_only'] with cohort='hris_only', etc.

    Returns:
        ColumnTransformer with remainder='drop' so any column not in the cohort's
        feature list (identifiers, label, temporal anchor, and — for hris_only —
        survey columns) is silently dropped.
        `verbose_feature_names_out=False` keeps output names as bare column
        names for legibility in SHAP plots.

    Teaching note — why cohort matters here:
        ColumnTransformer selects columns BY NAME from the input DataFrame.
        If build_preprocessor() lists 'enps' but the hris_only DataFrame doesn't
        have 'enps', sklearn raises KeyError at fit_transform time — not a
        graceful "ignore missing" but a hard failure. Explicit cohort filtering
        prevents that failure and makes the experiment transparent.
    """
    # cohort_feature_names gives us ONLY the feature columns for this cohort
    # (no identifiers, no temporal anchor, no label).
    cohort_features = set(get_cohort_feature_names(cohort))

    numeric_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "numeric" and s.name in cohort_features
    ]
    categorical_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "categorical" and s.name in cohort_features
    ]
    boolean_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "boolean" and s.name in cohort_features
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    boolean_pipe = Pipeline(
        steps=[
            (
                "cast",
                FunctionTransformer(
                    _cast_bool_to_float,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if boolean_cols:
        transformers.append(("boolean", boolean_pipe, boolean_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_columns(
    cohort: Literal["hris_only", "hybrid"] = "hybrid",
) -> list[str]:
    """Return the ordered list of input feature columns for a given cohort.

    This is the list of columns that must be present in X before calling
    preprocessor.fit_transform / preprocessor.transform for this cohort.

    Args:
        cohort: 'hybrid' (default, 13 features) or 'hris_only' (10 features).

    Returns:
        Ordered list of feature column names. Identifiers, temporal anchors,
        and labels are NOT included.
    """
    cohort_features = set(get_cohort_feature_names(cohort))
    return [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature"
        and s.dtype in ("numeric", "categorical", "boolean")
        and s.name in cohort_features
    ]


def build_ebm_preprocessor(
    cohort: Literal["hris_only", "hybrid"] = "hybrid",
) -> ColumnTransformer:
    """Return an imputation-only ColumnTransformer for EBM — no OHE, no scaling.

    EBM (ExplainableBoostingClassifier) handles categorical features natively
    by bucketing string values into bins internally. Applying OHE before EBM
    destroys that advantage — EBM would then see one-hot columns and treat each
    value as an independent binary feature rather than a level of the same
    categorical. This preprocessor intentionally omits OHE so EBM can use its
    native GAM + interaction handling on raw category levels.

    It also omits StandardScaler. EBM is a tree-based learner (boosted GAMs),
    not a gradient-descent method, so it is invariant to monotone feature
    transforms. Scaling would change nothing about the model output.

    Column routing (EBM-specific):
      numeric     → SimpleImputer(median)              [fills NaN; no scaling]
      categorical → SimpleImputer(constant='__miss__') [preserves string dtype]
      boolean     → FunctionTransformer(cast to float) [0.0 / 1.0]
      others      → dropped (remainder='drop')

    set_output(transform='pandas'):
        Returns a DataFrame (not ndarray) so EBM auto-detects feature types
        from column dtypes — float64 → 'continuous', object → 'nominal'.
        Column names are preserved for SHAP + EBM global explanation.

    Cross-model OHE tradeoff (acknowledged, not hidden):
        LR and GBM share build_preprocessor() which OHE-encodes categoricals.
        EBM uses this function and skips OHE. The comparison table (Story 2.5)
        therefore compares:
          LR/GBM: OHE → each category level is a feature coefficient
          EBM:    native → each category level is a bin in the GAM term
        This is documented in docs/methodology.md → Cross-Model Comparison.

    Args:
        cohort: 'hybrid' (default, 13 features) or 'hris_only' (10 features).

    Returns:
        Unfitted ColumnTransformer with set_output('pandas') active.
        Call `.fit_transform(X_train)` then `.transform(X_val/X_test)`.
    """
    cohort_features = set(get_cohort_feature_names(cohort))

    numeric_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "numeric" and s.name in cohort_features
    ]
    categorical_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "categorical" and s.name in cohort_features
    ]
    boolean_cols = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype == "boolean" and s.name in cohort_features
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        # No OHE — preserve string dtype so EBM handles the column as nominal.
        transformers.append(
            (
                "categorical",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
                categorical_cols,
            )
        )
    if boolean_cols:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    steps=[
                        (
                            "cast",
                            FunctionTransformer(
                                _cast_bool_to_float,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        )
                    ]
                ),
                boolean_cols,
            )
        )

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # set_output("pandas") makes transform() return a DataFrame so EBM
    # auto-detects feature types from column dtypes (float64 → continuous,
    # object → nominal). Required for native categorical handling to work.
    ct.set_output(transform="pandas")
    return ct


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names after fitting.

    Wraps sklearn's `get_feature_names_out()` and converts to a plain list.
    Must be called AFTER `preprocessor.fit()` or `preprocessor.fit_transform()`.

    Args:
        preprocessor: A fitted ColumnTransformer returned by `build_preprocessor`.

    Returns:
        List of output column names. OHE columns appear as '<original_col>_<value>'.
        Numeric and boolean columns keep their original name.
    """
    return list(preprocessor.get_feature_names_out())
