"""Preprocessing pipeline — Story 2.1.

Builds a sklearn ColumnTransformer driven entirely by FEATURE_CATALOG roles
and dtypes. The catalog is the single source of truth; adding a new feature
there automatically routes it through the right transformer here.

Column routing:
  numeric       → SimpleImputer(median) → StandardScaler
  categorical   → SimpleImputer(constant='__missing__') → OneHotEncoder
  boolean       → FunctionTransformer(cast to float64, preserves NaN-free booleans)
  date/string   → dropped (snapshot_date is temporal_anchor; no free-text features)
  identifier    → dropped (employee_id never enters the model)
  temporal_anchor → dropped
  label         → dropped (y is handled separately by the caller)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from retention.features.catalog import FEATURE_CATALOG

if TYPE_CHECKING:
    pass


def _cast_bool_to_float(X: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Cast boolean columns to float64 for sklearn compatibility."""
    return X.astype(np.float64)


def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-on-demand ColumnTransformer built from FEATURE_CATALOG.

    The transformer is NOT fitted here — call `.fit_transform(X_train)` and
    `.transform(X_val)` / `.transform(X_test)` in the training pipeline.

    Returns:
        ColumnTransformer with remainder='drop' so any column not explicitly
        listed (identifiers, label, temporal anchor) is silently dropped.
        `verbose_feature_names_out=False` keeps output names as bare column
        names for legibility in SHAP plots.
    """
    numeric_cols = [s.name for s in FEATURE_CATALOG if s.role == "feature" and s.dtype == "numeric"]
    categorical_cols = [
        s.name for s in FEATURE_CATALOG if s.role == "feature" and s.dtype == "categorical"
    ]
    boolean_cols = [s.name for s in FEATURE_CATALOG if s.role == "feature" and s.dtype == "boolean"]

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


def get_feature_columns() -> list[str]:
    """Return the ordered list of input columns expected by the preprocessor.

    This is the union of numeric + categorical + boolean feature names from
    the catalog — i.e. the columns that must be present in X before calling
    preprocessor.fit_transform / preprocessor.transform.

    Identifiers, temporal anchors, and labels are NOT in this list; they
    should be dropped from the DataFrame before it reaches the preprocessor.
    The ColumnTransformer would drop them via remainder='drop' anyway, but
    being explicit avoids accidental shape surprises.
    """
    return [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype in ("numeric", "categorical", "boolean")
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
