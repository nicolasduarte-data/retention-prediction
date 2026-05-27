"""Story 2.3 — Tests for the XGBoost retention model wrapper.

Scope: smoke + behavioral. No hyperparameter tuning. The goal is to verify:
1. fit() → predict_proba() pipeline works end-to-end.
2. fit() without val set works.
3. fit() with val set works (eval_set path).
4. predict_proba() raises before fit().
5. predict() returns binary array at threshold.
6. Class 1 column of predict_proba sums to something plausible (not all-zeros).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from retention.models.xgb import RetentionModel


# ------------------------------------------------------------------ #
# Fixture                                                              #
# ------------------------------------------------------------------ #


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Small synthetic HR dataset — deterministic, no BQ dependency."""
    rng = np.random.default_rng(42)
    n = 80

    X = pd.DataFrame(
        {
            "tenure_months": rng.uniform(1, 60, n),
            "compa_ratio": rng.uniform(0.7, 1.3, n),
            "successor_count": rng.integers(0, 4, n).astype(float),
            "age_at_window_close": rng.uniform(22, 60, n),
            "performance_tier": rng.choice(["2", "3", "4", "5"], n),
            "gender": rng.choice(["M", "F", "NB"], n),
            "is_critical_role": rng.choice([True, False], n),
        }
    )
    # ~25% attrition rate, correlated with low compa_ratio + low tenure
    prob = 0.1 + 0.3 * (X["compa_ratio"] < 0.9).astype(float)
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="voluntary_exit_label")

    split = 60
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ------------------------------------------------------------------ #
# Basic fit / predict                                                  #
# ------------------------------------------------------------------ #


def test_fit_returns_self(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, _, _ = synthetic_data
    model = RetentionModel()
    result = model.fit(X_train, y_train)
    assert result is model


def test_pipeline_is_set_after_fit(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, _, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    assert isinstance(model.pipeline, Pipeline)


def test_predict_proba_before_fit_raises() -> None:
    model = RetentionModel()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict_proba(pd.DataFrame())


def test_predict_proba_shape(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_predict_proba_col1_sums_to_positive(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Model should not output all-zero probabilities on test set."""
    X_train, y_train, X_test, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba[:, 1].sum() > 0, "All exit probabilities are zero — model is degenerate."


def test_predict_proba_values_in_unit_interval(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_predict_returns_binary_array(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1}), f"predict() returned non-binary values: {set(preds)}"


def test_fit_with_val_set_works(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """eval_set path must not raise and should produce valid predictions."""
    X_train, y_train, X_val, y_val = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2)
    assert not np.isnan(proba).any()


def test_row_probabilities_sum_to_one(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    model = RetentionModel()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
