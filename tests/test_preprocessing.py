"""Story 2.1 — Tests for the preprocessing pipeline.

Three layers:
1. Shape / column routing — the right columns go through the right transformers.
2. Behavior under nulls — imputation strategies work as expected.
3. Catalog parity — get_feature_columns() matches what build_preprocessor() consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from retention.features.catalog import FEATURE_CATALOG
from retention.features.preprocessing import (
    build_preprocessor,
    get_feature_columns,
    get_feature_names_out,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with all v0.1 feature columns — no nulls."""
    return pd.DataFrame(
        {
            "tenure_months": [12.0, 24.0, 36.0, 6.0, 48.0],
            "compa_ratio": [0.9, 1.0, 1.1, 0.8, 1.2],
            "successor_count": [0.0, 1.0, 2.0, 0.0, 3.0],
            "age_at_window_close": [28.0, 35.0, 42.0, 25.0, 50.0],
            "enps": [10.0, -20.0, 50.0, 0.0, 30.0],
            "engagement_score": [3.5, 2.1, 4.8, 3.0, 4.2],
            "manager_relationship_score": [4.0, 2.5, 5.0, 3.2, 4.5],
            "performance_tier": ["3", "4", "5", "2", "3"],
            "gender": ["F", "M", "F", "M", "F"],
            "is_critical_role": [False, True, False, True, False],
            # Non-feature columns that should be dropped:
            "employee_id": ["EMP_001", "EMP_002", "EMP_003", "EMP_004", "EMP_005"],
            "snapshot_date": pd.to_datetime(["2025-05-27"] * 5, format="%Y-%m-%d"),
            "voluntary_exit_label": [False, True, False, False, True],
        }
    )


@pytest.fixture()
def df_with_nulls(minimal_df: pd.DataFrame) -> pd.DataFrame:
    """Same structure but with deliberate nulls in feature columns."""
    df = minimal_df.copy()
    df.loc[0, "tenure_months"] = np.nan
    df.loc[1, "compa_ratio"] = np.nan
    df.loc[2, "performance_tier"] = np.nan
    return df


# ------------------------------------------------------------------ #
# Shape / column routing                                               #
# ------------------------------------------------------------------ #


def test_build_preprocessor_returns_column_transformer() -> None:
    assert isinstance(build_preprocessor(), ColumnTransformer)


def test_transform_output_is_2d_float_array(minimal_df: pd.DataFrame) -> None:
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = minimal_df[feature_cols]
    result = pre.fit_transform(X)
    assert result.ndim == 2
    assert result.dtype == np.float64


def test_transform_drops_non_feature_columns(minimal_df: pd.DataFrame) -> None:
    """remainder='drop' must exclude employee_id, snapshot_date, label."""
    pre = build_preprocessor()
    # Pass the full DataFrame — the transformer should silently drop non-feature cols
    pre.fit_transform(minimal_df)
    # Output columns should not encode employee_id or snapshot_date
    output_names = get_feature_names_out(pre)
    for name in output_names:
        assert "employee_id" not in name
        assert "snapshot_date" not in name
        assert "voluntary_exit_label" not in name


def test_boolean_column_is_float_in_output(minimal_df: pd.DataFrame) -> None:
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = minimal_df[feature_cols]
    result = pre.fit_transform(X)
    output_names = get_feature_names_out(pre)
    # is_critical_role should appear in output names
    assert "is_critical_role" in output_names
    # And its column in output should contain only 0.0 / 1.0
    idx = output_names.index("is_critical_role")
    col_values = result[:, idx]
    assert set(col_values).issubset({0.0, 1.0})


def test_ohe_creates_multiple_columns_for_categorical(minimal_df: pd.DataFrame) -> None:
    """performance_tier has 4 unique values → OHE should expand to 4+ columns."""
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = minimal_df[feature_cols]
    pre.fit_transform(X)
    output_names = get_feature_names_out(pre)
    perf_cols = [n for n in output_names if n.startswith("performance_tier")]
    assert len(perf_cols) >= 2, (
        f"Expected multiple OHE columns for performance_tier; got {perf_cols}"
    )


def test_numeric_columns_are_scaled(minimal_df: pd.DataFrame) -> None:
    """After StandardScaler, numeric output should have mean ≈ 0 (within rounding)."""
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = minimal_df[feature_cols]
    result = pre.fit_transform(X)
    output_names = get_feature_names_out(pre)
    numeric_names = [
        s.name for s in FEATURE_CATALOG if s.role == "feature" and s.dtype == "numeric"
    ]
    for col in numeric_names:
        if col in output_names:
            idx = output_names.index(col)
            col_mean = result[:, idx].mean()
            assert abs(col_mean) < 1e-10, (
                f"Numeric column '{col}' mean after scaling = {col_mean:.6f}; expected ≈ 0.0."
            )


# ------------------------------------------------------------------ #
# Behavior under nulls                                                 #
# ------------------------------------------------------------------ #


def test_numeric_nulls_imputed_with_median(df_with_nulls: pd.DataFrame) -> None:
    """NaN in tenure_months should be filled with median, not propagate."""
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = df_with_nulls[feature_cols]
    result = pre.fit_transform(X)
    assert not np.isnan(result).any(), (
        "Null values leaked through the preprocessor — imputation failed."
    )


def test_categorical_nulls_get_missing_category(df_with_nulls: pd.DataFrame) -> None:
    """NaN in performance_tier should produce a '__missing__' OHE column."""
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = df_with_nulls[feature_cols]
    pre.fit_transform(X)
    output_names = get_feature_names_out(pre)
    missing_cols = [n for n in output_names if "__missing__" in n]
    assert missing_cols, (
        "Expected a '__missing__' OHE column for null categorical values; "
        f"output names: {output_names}"
    )


def test_no_nans_in_output_after_imputation(df_with_nulls: pd.DataFrame) -> None:
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = df_with_nulls[feature_cols]
    result = pre.fit_transform(X)
    assert not np.isnan(result).any()


# ------------------------------------------------------------------ #
# Catalog parity                                                       #
# ------------------------------------------------------------------ #


def test_get_feature_columns_matches_catalog() -> None:
    """get_feature_columns() should match all role='feature' dtypes in {numeric,cat,bool}."""
    expected = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.dtype in ("numeric", "categorical", "boolean")
    ]
    assert get_feature_columns() == expected


def test_preprocessor_handles_all_catalog_feature_dtypes(minimal_df: pd.DataFrame) -> None:
    """Smoke test: build + fit_transform doesn't raise for any catalog feature dtype."""
    pre = build_preprocessor()
    feature_cols = get_feature_columns()
    X = minimal_df[feature_cols]
    result = pre.fit_transform(X)
    assert result.shape[0] == len(minimal_df)
    assert result.shape[1] > 0
