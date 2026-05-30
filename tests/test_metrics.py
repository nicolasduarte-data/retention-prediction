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


def test_precision_at_k_accepts_2d_proba() -> None:
    """predict_proba returns (n, 2); precision_at_k must use column 1.

    Story 3.7: mutation testing showed the `proba.ndim == 2` branch and the
    `[:, 1]` column extract were untested for precision_at_k (mutants `== 3`
    and `[:, 2]` survived). predict_proba ALWAYS returns (n, 2) in production,
    so this path is the real one.
    """
    y = np.array([1, 1, 0, 0, 1, 0])
    p_1d = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    p_2d = np.column_stack([1 - p_1d, p_1d])
    assert precision_at_k(y, p_1d, k=0.50) == pytest.approx(precision_at_k(y, p_2d, k=0.50))


def test_recall_at_k_accepts_2d_proba() -> None:
    """recall_at_k must also reduce a (n, 2) predict_proba to column 1."""
    y = np.array([1, 1, 0, 0, 1, 0])
    p_1d = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    p_2d = np.column_stack([1 - p_1d, p_1d])
    assert recall_at_k(y, p_1d, k=0.50) == pytest.approx(recall_at_k(y, p_2d, k=0.50))


def test_recall_at_k_invalid_k_raises() -> None:
    """recall_at_k validates k ∈ (0, 1] — Story 3.7 found this guard untested.

    Kills the `0 < k` → `0 <= k` and `k <= 1` → `k <= 2` mutants that survived
    because recall_at_k (unlike precision_at_k) had no invalid-k test.
    """
    y = np.array([1, 0])
    p = np.array([0.8, 0.2])
    with pytest.raises(ValueError, match="k must be in"):
        recall_at_k(y, p, k=0.0)
    with pytest.raises(ValueError, match="k must be in"):
        recall_at_k(y, p, k=1.5)


def test_precision_at_k_floor_selects_at_least_one() -> None:
    """`max(1, ceil(n·k))` guarantees ≥ 1 flagged even when n·k rounds below 1.

    Story 3.7: kills the `max(1, …)` → `max(2, …)` and `n·k` → `n/k` mutants.
    With n=10, k=0.05 → ceil(0.5)=1 → exactly one row flagged (the top-ranked).
    The single highest-proba row here is a true exit, so precision = 1.0; a
    `max(2, …)` mutant would flag two and drop precision to 0.5.
    """
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.99, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.10])
    # top-1 (proba 0.99) is index 0, a true exit → precision@(k=0.05) = 1/1 = 1.0
    assert precision_at_k(y, p, k=0.05) == pytest.approx(1.0)


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


def test_recall_at_k_n_top_count_is_exact() -> None:
    """recall_at_k flags exactly max(1, ceil(n·k)) rows — Story 3.7.

    Kills recall_at_k's `max(1, …)` → `max(2, …)` (#486) and `n·k` → `n/k`
    (#487), which survived because recall had no known-output count test.

    Two positives sit at the very top. k=0.05, n=10 → n_top = ceil(0.5) = 1,
    so only the single highest-proba positive is captured → recall = 1/2 = 0.5.
      `max(2, …)` flags 2 → captures both positives → recall 1.0 ≠ 0.5
      `n/k` = 200    → flags all 10 → captures both       → recall 1.0 ≠ 0.5
    """
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.99, 0.98, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.10])
    assert recall_at_k(y, p, k=0.05) == pytest.approx(0.5)


def test_recall_at_k_2d_proba_known_value() -> None:
    """recall_at_k reduces (n, 2) predict_proba to column 1 — Story 3.7.

    Kills recall_at_k's `proba.ndim == 2` → `== 3` (#496): with the mutant a
    2-D array is never reduced, so `argsort` runs on the wrong axis and the
    captured count diverges from the known answer. Top-2 (k=0.5) are the two
    positives → recall = 2/2 = 1.0.
    """
    y = np.array([1, 1, 0, 0])
    p_1d = np.array([0.9, 0.8, 0.2, 0.1])
    p_2d = np.column_stack([1 - p_1d, p_1d])
    assert recall_at_k(y, p_2d, k=0.5) == pytest.approx(1.0)


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


# ------------------------------------------------------------------ #
# Shared proba-validation contract — feast T2-1 / T2-2 / T2-EV-2       #
# ------------------------------------------------------------------ #


class TestSharedProbaValidation:
    """Every metric routes proba through ``_validation.extract_positive_proba``.

    Before the consolidation, ``precision_at_k``/``recall_at_k`` did NOT reject
    NaN — ``np.argsort`` sorts NaN to the end, so the descending reversal placed
    a NaN-scored employee at the TOP of the risk ranking (silent wrong answer).
    These tests lock the now-uniform contract: non-finite, empty, and wrong-2-D
    inputs are rejected identically across the package.
    """

    def test_precision_rejects_nan_proba(self) -> None:
        """The NaN-hijack path is closed: a NaN probability now raises."""
        with pytest.raises(ValueError, match="non-finite"):
            precision_at_k(np.array([1, 0, 1]), np.array([0.5, np.nan, 0.3]))

    def test_recall_rejects_nan_proba(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            recall_at_k(np.array([1, 0, 1]), np.array([0.5, np.nan, 0.3]))

    def test_auc_pr_rejects_inf_proba(self) -> None:
        """auc_pr's old `np.isnan` guard missed Inf; the shared `isfinite` catches it."""
        with pytest.raises(ValueError, match="non-finite"):
            auc_pr(np.array([1, 0, 1]), np.array([0.5, np.inf, 0.3]))

    def test_auc_roc_rejects_nan_proba(self) -> None:
        """auc_roc previously had no validation at all."""
        with pytest.raises(ValueError, match="non-finite"):
            auc_roc(np.array([1, 0, 1]), np.array([0.5, np.nan, 0.3]))

    def test_auc_roc_rejects_wrong_2d_shape(self) -> None:
        with pytest.raises(ValueError, match="2 columns"):
            auc_roc(np.array([1, 0, 1]), np.ones((3, 3)) / 3)

    def test_metrics_reject_empty_proba(self) -> None:
        """Empty input is now a uniform error (was NaN for precision, 0.0 for recall)."""
        empty = np.array([])
        for fn in (auc_pr, auc_roc, brier_score, precision_at_k, recall_at_k):
            with pytest.raises(ValueError, match="empty"):
                fn(empty, empty)
