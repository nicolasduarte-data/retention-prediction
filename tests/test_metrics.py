"""Story 3.1 / Story 2.5 — Tests for evaluation metrics.

Coverage:
  auc_pr()           — primary metric, Story 3.1
  format_rung1_caption() — epistemic caption, Story 3.1
  auc_roc()          — secondary metric, Story 2.5
  precision_at_k()   — operational HR metric, Story 2.5
  recall_at_k()      — FLIP-RISK mitigation metric, Story 2.5
  brier_score()      — calibration-aware metric, Story 2.5
"""

from __future__ import annotations

import numpy as np
import pytest

from retention.evaluation.metrics import (
    auc_pr,
    auc_roc,
    brier_score,
    format_rung1_caption,
    precision_at_k,
    recall_at_k,
)


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


# ------------------------------------------------------------------ #
# auc_roc() — Story 2.5                                                #
# ------------------------------------------------------------------ #


def test_auc_roc_perfect_ranker() -> None:
    """Perfect ranking: all positives above all negatives → AUC-ROC = 1.0."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_roc(y_true, y_proba) == pytest.approx(1.0, abs=1e-9)


def test_auc_roc_in_unit_interval() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, 80)
    y_proba = rng.uniform(0, 1, 80)
    result = auc_roc(y_true, y_proba)
    assert 0.0 <= result <= 1.0


def test_auc_roc_accepts_2d_proba() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba_1d = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
    y_proba_2d = np.column_stack([1 - y_proba_1d, y_proba_1d])
    assert auc_roc(y_true, y_proba_1d) == pytest.approx(auc_roc(y_true, y_proba_2d), abs=1e-9)


# ------------------------------------------------------------------ #
# precision_at_k() — Story 2.5                                         #
# ------------------------------------------------------------------ #


def test_precision_at_k_perfect_model() -> None:
    """Top-k all positives: precision@k = 1.0."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    # k=0.33 → ceil(6 * 0.33) = ceil(1.98) = 2 rows (exactly the two positives)
    result = precision_at_k(y_true, y_proba, k=0.33)
    assert result == pytest.approx(1.0, abs=1e-9)


def test_precision_at_k_in_unit_interval() -> None:
    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, 50)
    y_proba = rng.uniform(0, 1, 50)
    for k in [0.10, 0.20, 0.50]:
        result = precision_at_k(y_true, y_proba, k=k)
        assert 0.0 <= result <= 1.0


def test_precision_at_k_invalid_k_raises() -> None:
    y_true = np.array([1, 0])
    y_proba = np.array([0.8, 0.2])
    with pytest.raises(ValueError, match="k must be in"):
        precision_at_k(y_true, y_proba, k=0.0)
    with pytest.raises(ValueError, match="k must be in"):
        precision_at_k(y_true, y_proba, k=1.5)


# ------------------------------------------------------------------ #
# recall_at_k() — Story 2.5                                            #
# ------------------------------------------------------------------ #


def test_recall_at_k_all_positives_captured() -> None:
    """k=1.0 (full list): all positives are in the flag list → recall = 1.0."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.1])
    assert recall_at_k(y_true, y_proba, k=1.0) == pytest.approx(1.0, abs=1e-9)


def test_recall_at_k_no_positives_returns_zero() -> None:
    """Degenerate case: no positives in y_true → recall = 0.0 (not ZeroDivisionError)."""
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.9, 0.7, 0.5, 0.1])
    assert recall_at_k(y_true, y_proba, k=0.5) == pytest.approx(0.0, abs=1e-9)


def test_recall_at_k_in_unit_interval() -> None:
    rng = np.random.default_rng(13)
    y_true = rng.integers(0, 2, 50)
    y_proba = rng.uniform(0, 1, 50)
    for k in [0.10, 0.20, 0.50]:
        result = recall_at_k(y_true, y_proba, k=k)
        assert 0.0 <= result <= 1.0


# ------------------------------------------------------------------ #
# brier_score() — Story 2.5                                            #
# ------------------------------------------------------------------ #


def test_brier_score_perfect_model() -> None:
    """Perfect probability predictions → Brier = 0.0."""
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)


def test_brier_score_in_unit_interval() -> None:
    rng = np.random.default_rng(21)
    y_true = rng.integers(0, 2, 80)
    y_proba = rng.uniform(0, 1, 80)
    result = brier_score(y_true, y_proba)
    assert 0.0 <= result <= 1.0


def test_brier_score_accepts_2d_proba() -> None:
    y_true = np.array([1, 0, 1])
    y_proba_1d = np.array([0.9, 0.1, 0.7])
    y_proba_2d = np.column_stack([1 - y_proba_1d, y_proba_1d])
    assert brier_score(y_true, y_proba_1d) == pytest.approx(
        brier_score(y_true, y_proba_2d), abs=1e-9
    )
