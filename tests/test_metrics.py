"""Story 3.1 — Tests for evaluation metrics.

Three things to verify:
1. auc_pr() returns a float in [0, 1].
2. A perfect ranker scores 1.0; a constant scorer scores near class prevalence.
3. format_rung1_caption() produces the exact Rung 1 caption format.
"""

from __future__ import annotations

import numpy as np
import pytest

from retention.evaluation.metrics import auc_pr, format_rung1_caption


# ------------------------------------------------------------------ #
# auc_pr()                                                             #
# ------------------------------------------------------------------ #


def test_auc_pr_perfect_ranker() -> None:
    """When all positives rank above all negatives, AUC-PR = 1.0."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auc_pr(y_true, y_proba) == pytest.approx(1.0, abs=1e-9)


def test_auc_pr_returns_float() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.8, 0.4, 0.6, 0.3])
    result = auc_pr(y_true, y_proba)
    assert isinstance(result, float)


def test_auc_pr_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    result = auc_pr(y_true, y_proba)
    assert 0.0 <= result <= 1.0


def test_auc_pr_accepts_2d_proba() -> None:
    """predict_proba returns shape (n, 2); auc_pr should accept it and use col 1."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba_1d = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
    y_proba_2d = np.column_stack([1 - y_proba_1d, y_proba_1d])
    assert auc_pr(y_true, y_proba_1d) == pytest.approx(auc_pr(y_true, y_proba_2d), abs=1e-9)


def test_auc_pr_rejects_nan_proba() -> None:
    y_true = np.array([1, 0, 1])
    y_proba = np.array([0.8, np.nan, 0.6])
    with pytest.raises(ValueError, match="NaN"):
        auc_pr(y_true, y_proba)


def test_auc_pr_rejects_wrong_2d_shape() -> None:
    y_true = np.array([1, 0, 1])
    y_proba = np.ones((3, 3)) / 3
    with pytest.raises(ValueError, match="2 columns"):
        auc_pr(y_true, y_proba)


def test_auc_pr_constant_predictor_near_prevalence() -> None:
    """A model that always predicts 0.5 should score ≈ class prevalence."""
    rng = np.random.default_rng(7)
    n = 500
    prevalence = 0.2
    y_true = (rng.uniform(size=n) < prevalence).astype(int)
    y_proba = np.full(n, 0.5)
    result = auc_pr(y_true, y_proba)
    # For a constant predictor, average_precision_score == prevalence
    assert abs(result - prevalence) < 0.02, (
        f"Expected constant predictor AUC-PR ≈ {prevalence:.2f}; got {result:.4f}"
    )


# ------------------------------------------------------------------ #
# format_rung1_caption()                                               #
# ------------------------------------------------------------------ #


def test_rung1_caption_exact_format() -> None:
    caption = format_rung1_caption(0.712)
    assert caption == "AUC-PR = 0.712 — associational, not causal (Rung 1)"


def test_rung1_caption_rounding() -> None:
    caption = format_rung1_caption(0.7126)
    assert caption == "AUC-PR = 0.713 — associational, not causal (Rung 1)"


def test_rung1_caption_contains_rung1_tag() -> None:
    caption = format_rung1_caption(0.5)
    assert "Rung 1" in caption


def test_rung1_caption_contains_not_causal() -> None:
    caption = format_rung1_caption(0.5)
    assert "not causal" in caption
