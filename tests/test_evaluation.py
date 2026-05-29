"""Tests for src/retention/evaluation/ — Stories 3.1 + 3.2.

Coverage scope:
    3.1  lift_at_k (metrics.py addition) — known-input known-output
    3.2  expected_calibration_error (calibration.py)
    3.2  reliability_diagram (calibration.py) — smoke + contract tests

Design principle:
    Every assertion uses a *constructed* known-input / known-output case.
    Avoid fixtures that depend on the real dataset — metric functions are
    pure computations and should be tested on arrays you can reason about
    without running the full pipeline.

    The one exception: `test_reliability_diagram_returns_figure` uses a
    minimal 20-row fixture — enough to populate 2–3 bins, not enough to
    load the real data.

Coverage strategy:
    - metrics.py functions not tested here (auc_pr, auc_roc, precision_at_k,
      recall_at_k, brier_score) are already covered by the main test suite
      (test_models.py + test_tracking.py use them indirectly). This file
      adds tests for the new lift_at_k and all calibration functions.
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np
import pytest

from retention.evaluation.calibration import expected_calibration_error, reliability_diagram
from retention.evaluation.metrics import lift_at_k, precision_at_k


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #


@pytest.fixture()
def perfect_ranker_20():
    """Perfect ranker: top 10 of 20 employees are all exits.

    y_true: [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]  (10 positives)
    proba:  descending 0.95..0.05 — positive class always gets highest score.

    Known properties at k=0.50 (top 10 / 20):
        precision@50% = 10/10 = 1.0
        base_rate     = 10/20 = 0.50
        lift@50%      = 1.0 / 0.50 = 2.0

    Known properties at k=0.10 (top 2 / 20):
        top-2 = indices with proba 0.95, 0.90 → both positive
        precision@10% = 2/2 = 1.0
        lift@10%      = 1.0 / 0.50 = 2.0
    """
    y = np.array([1] * 10 + [0] * 10, dtype=int)
    p = np.linspace(0.95, 0.05, 20)  # [0.95, 0.90, ..., 0.05]
    return y, p


@pytest.fixture()
def random_ranker_40():
    """Random ranker: a model that cannot distinguish exits from stays.

    Constructed so predicted probabilities are uncorrelated with labels:
        y_true: alternating 1/0 (20 positives out of 40)
        proba:  flat 0.5 for everyone

    Expected lift at any k: precision@k / base_rate.
    With flat proba=0.5, top-k contains half positives → precision = 0.5.
    base_rate = 20/40 = 0.5.
    lift = 0.5 / 0.5 = 1.0 (no better than random).
    """
    y = np.tile([1, 0], 20)  # [1,0,1,0,...] — 20 positives
    p = np.full(40, 0.5)
    return y, p


# ------------------------------------------------------------------ #
# Story 3.1 — lift_at_k                                                #
# ------------------------------------------------------------------ #


class TestLiftAtK:
    """Known-input known-output tests for lift_at_k."""

    def test_perfect_ranker_lift_equals_two(
        self,
        perfect_ranker_20: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """Perfect ranker: all positives rank first → lift = 1/base_rate.

        At k=0.50 (top 10 of 20):
            precision@50% = 1.0, base_rate = 0.50 → lift = 2.0.
        """
        y, p = perfect_ranker_20
        result = lift_at_k(y, p, k=0.50)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_random_ranker_lift_equals_one(
        self,
        random_ranker_40: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """Flat-probability ranker cannot beat random: lift = 1.0.

        precision@k / base_rate = 0.5 / 0.5 = 1.0 for any k with flat proba.
        """
        y, p = random_ranker_40
        result = lift_at_k(y, p, k=0.10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_lift_equals_precision_over_base_rate(
        self,
        perfect_ranker_20: tuple[np.ndarray, np.ndarray],  # type: ignore[type-arg]
    ) -> None:
        """lift_at_k = precision_at_k / mean(y_true) — by definition."""
        y, p = perfect_ranker_20
        for k in (0.10, 0.20, 0.50):
            prec = precision_at_k(y, p, k=k)
            base = float(y.mean())
            expected_lift = prec / base
            result = lift_at_k(y, p, k=k)
            assert result == pytest.approx(expected_lift, rel=1e-9), (
                f"lift@{k} = {result:.6f}, expected {expected_lift:.6f} "
                "(precision_at_k / base_rate)"
            )

    def test_lift_returns_zero_for_all_negative_y(self) -> None:
        """lift_at_k returns 0.0 when y_true is all zeros (no positives).

        base_rate = 0 → division by zero is guarded; return 0.0.
        """
        y = np.zeros(20, dtype=int)
        p = np.linspace(0.9, 0.1, 20)
        result = lift_at_k(y, p, k=0.10)
        assert result == 0.0

    def test_lift_rejects_invalid_k(self) -> None:
        """k must be in (0, 1] — 0 and negative values raise ValueError."""
        y = np.array([1, 0, 1, 0])
        p = np.array([0.8, 0.6, 0.4, 0.2])
        with pytest.raises(ValueError, match="k must be in"):
            lift_at_k(y, p, k=0.0)
        with pytest.raises(ValueError, match="k must be in"):
            lift_at_k(y, p, k=-0.1)

    def test_lift_k_equals_one_uses_full_population(self) -> None:
        """At k=1.0 precision@k = base_rate, so lift = 1.0 always."""
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 50)
        p = rng.uniform(0, 1, 50)
        result = lift_at_k(y, p, k=1.0)
        # precision at k=1.0 = base_rate → lift = 1.0
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_lift_accepts_2d_proba(self) -> None:
        """lift_at_k handles predict_proba shape (n, 2) — extracts column 1."""
        y = np.array([1, 1, 0, 0, 1, 0])
        p_1d = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
        p_2d = np.column_stack([1 - p_1d, p_1d])  # shape (6, 2)
        result_1d = lift_at_k(y, p_1d, k=0.50)
        result_2d = lift_at_k(y, p_2d, k=0.50)
        assert result_1d == pytest.approx(result_2d, rel=1e-9)


# ------------------------------------------------------------------ #
# Story 3.2 — expected_calibration_error                               #
# ------------------------------------------------------------------ #


class TestExpectedCalibrationError:
    """Known-input known-output tests for ECE."""

    def test_perfect_calibration_gives_zero_ece(self) -> None:
        """ECE = 0 when predicted probability = observed rate in every bin.

        Construction: 10 equally-spaced confidence levels, each with
        exactly the matching positive rate.
            Bin 0: predicted=0.05, 100 obs, frac_pos=0.05 → |diff|=0
            Bin 1: predicted=0.15, 100 obs, frac_pos=0.15 → |diff|=0
            ...
        ECE = Σ (100/1000) × 0 = 0.
        """
        rng = np.random.default_rng(0)
        n_per_bin = 100
        y_list = []
        p_list = []
        for mid in np.linspace(0.05, 0.95, 10):
            # Bernoulli samples with p = mid → E[frac_pos] = mid
            # Use n_per_bin large enough that law of large numbers kicks in
            labels = rng.binomial(1, mid, n_per_bin)
            probs = np.full(n_per_bin, mid)
            y_list.append(labels)
            p_list.append(probs)

        y = np.concatenate(y_list)
        p = np.concatenate(p_list)

        ece = expected_calibration_error(y, p, n_bins=10)
        # Law of large numbers: ECE should be < 0.05 for n=100 per bin
        # (not exactly 0 due to random sampling, but very close)
        assert ece < 0.05, (
            f"ECE = {ece:.4f} for a near-perfect calibrator. Expected < 0.05 with n=100 per bin."
        )

    def test_overconfident_model_has_positive_ece(self) -> None:
        """A model that always predicts 0.9 for a 20% base rate has high ECE.

        Predicted mean in the top bin: 0.9
        Observed fraction positive: ~0.2
        ECE ≈ 1.0 × |0.9 − 0.2| = 0.7 (all examples land in the top bin).
        """
        n = 500
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.20, n)
        p = np.full(n, 0.90)  # model always says 90%

        ece = expected_calibration_error(y, p, n_bins=10)
        # ECE ≈ |0.9 - 0.2| = 0.7; allow generous tolerance for sampling variance
        assert ece > 0.50, f"ECE = {ece:.4f} for overconfident model; expected > 0.50."

    def test_base_rate_predictor_has_near_zero_ece(self) -> None:
        """A model that always predicts the base rate has ECE near 0.

        If everyone gets predicted = base_rate, all examples land in one bin.
        That bin's confidence = base_rate and frac_pos ≈ base_rate → ECE ≈ 0.
        """
        n = 400
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.20, n)
        base_rate = float(y.mean())
        p = np.full(n, base_rate)

        ece = expected_calibration_error(y, p, n_bins=10)
        # All examples in one bin with conf = base_rate ≈ frac_pos
        assert ece < 0.02, f"ECE = {ece:.4f} for base-rate constant predictor; expected < 0.02."

    def test_ece_is_non_negative(self) -> None:
        """ECE is always >= 0 by construction (absolute difference)."""
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.30, 200)
        p = rng.uniform(0.0, 1.0, 200)
        ece = expected_calibration_error(y, p)
        assert ece >= 0.0

    def test_ece_rejects_non_finite_proba(self) -> None:
        """NaN or Inf in y_proba must raise ValueError."""
        y = np.array([1, 0, 1, 0])
        p_nan = np.array([0.5, np.nan, 0.3, 0.7])
        p_inf = np.array([0.5, np.inf, 0.3, 0.7])
        with pytest.raises(ValueError, match="non-finite"):
            expected_calibration_error(y, p_nan)
        with pytest.raises(ValueError, match="non-finite"):
            expected_calibration_error(y, p_inf)

    def test_ece_rejects_n_bins_less_than_2(self) -> None:
        """n_bins must be >= 2."""
        y = np.array([1, 0])
        p = np.array([0.8, 0.2])
        with pytest.raises(ValueError, match="n_bins must be"):
            expected_calibration_error(y, p, n_bins=1)

    def test_ece_accepts_2d_proba(self) -> None:
        """ECE handles predict_proba shape (n, 2) — extracts column 1."""
        rng = np.random.default_rng(4)
        y = rng.binomial(1, 0.25, 100)
        p_1d = rng.uniform(0, 1, 100)
        p_2d = np.column_stack([1 - p_1d, p_1d])
        ece_1d = expected_calibration_error(y, p_1d)
        ece_2d = expected_calibration_error(y, p_2d)
        assert ece_1d == pytest.approx(ece_2d, abs=1e-9)

    def test_ece_range_in_zero_one(self) -> None:
        """ECE is bounded in [0, 1] for valid inputs."""
        rng = np.random.default_rng(5)
        for _ in range(20):
            n = rng.integers(30, 300)
            y = rng.binomial(1, rng.uniform(0.1, 0.5), n)
            p = rng.uniform(0, 1, n)
            ece = expected_calibration_error(y, p)
            assert 0.0 <= ece <= 1.0, f"ECE = {ece} out of [0, 1]"


# ------------------------------------------------------------------ #
# Story 3.2 — reliability_diagram                                      #
# ------------------------------------------------------------------ #


class TestReliabilityDiagram:
    """Smoke + contract tests for reliability_diagram.

    We don't test visual output (the exact pixels of a bar chart are
    not contractual). We test:
      - The function returns a matplotlib Figure object.
      - It runs without error for all supported input shapes.
      - It handles edge cases (empty bins, single-class y_true).
      - Axes have the expected labels.
    """

    def _make_inputs(self, n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        """Convenience: generate (y_true, y_proba) for smoke tests."""
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, 0.25, n)
        p = rng.uniform(0, 1, n)
        return y, p

    def test_returns_matplotlib_figure(self) -> None:
        """reliability_diagram must return a matplotlib Figure."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        fig = reliability_diagram(y, p)
        assert isinstance(fig, matplotlib.figure.Figure), (
            f"Expected matplotlib.figure.Figure, got {type(fig).__name__}"
        )
        plt.close(fig)

    def test_runs_without_error_1d_proba(self) -> None:
        """reliability_diagram must not raise for 1D probability input."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        fig = reliability_diagram(y, p)
        plt.close(fig)

    def test_runs_without_error_2d_proba(self) -> None:
        """reliability_diagram handles predict_proba shape (n, 2)."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p_1d = self._make_inputs()
        p_2d = np.column_stack([1 - p_1d, p_1d])
        fig = reliability_diagram(y, p_2d)
        plt.close(fig)

    def test_model_name_and_cohort_appear_in_title(self) -> None:
        """Title must contain both model_name and cohort when provided."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        fig = reliability_diagram(y, p, model_name="GBM", cohort="hybrid")
        ax = fig.axes[0]
        assert "GBM" in ax.get_title(), f"'GBM' not in title: {ax.get_title()!r}"
        assert "hybrid" in ax.get_title(), f"'hybrid' not in title: {ax.get_title()!r}"
        plt.close(fig)

    def test_axes_labels_are_set(self) -> None:
        """x-axis and y-axis labels must be non-empty."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        fig = reliability_diagram(y, p)
        ax = fig.axes[0]
        assert len(ax.get_xlabel()) > 0, "x-axis label is empty"
        assert len(ax.get_ylabel()) > 0, "y-axis label is empty"
        plt.close(fig)

    def test_ece_annotation_present_when_show_ece_true(self) -> None:
        """When show_ece=True, the figure should have a text annotation with 'ECE'."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        fig = reliability_diagram(y, p, show_ece=True)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("ECE" in t for t in texts), f"No 'ECE' annotation found in axes texts: {texts!r}"
        plt.close(fig)

    def test_handles_all_negatives_y(self) -> None:
        """reliability_diagram must not raise when y_true is all zeros."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y = np.zeros(30, dtype=int)
        p = np.linspace(0.05, 0.95, 30)
        fig = reliability_diagram(y, p)
        plt.close(fig)

    def test_different_n_bins_produces_figure(self) -> None:
        """n_bins parameter is accepted without error for valid values."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        y, p = self._make_inputs()
        for n_bins in (5, 10, 20):
            fig = reliability_diagram(y, p, n_bins=n_bins)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)
