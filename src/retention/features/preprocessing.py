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
