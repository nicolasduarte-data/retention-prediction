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

    def test_lift_rejects_nan_proba(self) -> None:
        """lift_at_k delegates to precision_at_k, so the shared validator closes
        the NaN-hijack here too (feast T2-1)."""
        with pytest.raises(ValueError, match="non-finite"):
            lift_at_k(np.array([1, 0, 1]), np.array([0.5, np.nan, 0.3]), k=0.5)


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

    def test_ece_single_class_returns_finite_not_nan(self) -> None:
        """A single-class label set yields a FINITE ECE, not NaN (T2-CAL-1).

        The docstring once claimed ECE returns NaN for a degenerate (all-0/all-1)
        label set; the code never did. This locks the *actual* behavior: with all
        labels equal, every populated bin's observed rate is 0, so ECE collapses
        to the weighted mean predicted probability — a finite number.
        """
        n = 200
        rng = np.random.default_rng(11)
        y = np.zeros(n, dtype=int)  # single class
        p = rng.uniform(0.0, 1.0, n)
        ece = expected_calibration_error(y, p, n_bins=10)
        assert np.isfinite(ece), f"ECE should be finite for a single-class set, got {ece}"
        # all labels 0 → every bin accuracy is 0 → ECE == weighted mean predicted prob == mean(p)
        assert ece == pytest.approx(float(p.mean()), abs=1e-9)


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


# ------------------------------------------------------------------ #
# Story 3.7 — mutation-targeted ECE precision tests                    #
# ------------------------------------------------------------------ #


class TestECEBinLevelPrecision:
    """Analytically exact ECE tests targeting bin-edge and accumulation mutants.

    Story 3.7 (mutation testing) revealed 7 surviving mutants in the ECE
    binning logic that the existing threshold-based tests could not kill:

        Mutant 34  — ``ece =`` instead of ``ece +=`` (only last bin counted)
        Mutants 14, 15, 19, 20 — wrong linspace args / empty thresholds
        Mutant 16  — ``n_bins - 1`` linspace (coarser bins split one correct bin)
        Mutant 17  — ``n_bins + 2`` linspace (finer bins split one correct bin)

    Why threshold-based tests fail:
        ``assert ece < 0.05`` passes even when bin edges are wrong, as long as
        the predictions happen to fall in the same relative bins.  Exact-value
        assertions on constructed inputs force a failure whenever any step of
        the ECE formula (binning, accumulation) deviates.

    Construction principle:
        All predictions within a group are identical (same float) so that
        ``proba[mask].mean()`` is exact regardless of which bin the group lands
        in.  Labels are 0 or 1 exclusively so ``y_arr[mask].mean()`` is an
        exact integer ratio.
    """

    def test_ece_two_adjacent_bins_exact_value(self) -> None:
        """ECE sums contributions from bin 0 and bin 1 — exact accumulation.

        Construction:
            Group A — 100 samples, p=0.05 (bin 0: [0.0, 0.1)), all positive.
                conf=0.05, acc=1.0, error=0.95, weight=0.5 → contrib=0.475
            Group B — 100 samples, p=0.15 (bin 1: [0.1, 0.2)), all negative.
                conf=0.15, acc=0.0, error=0.15, weight=0.5 → contrib=0.075
            Expected ECE = 0.475 + 0.075 = 0.550

        Killed mutants:
            #34 (ece= not ece+=): only Group B retained → ECE=0.075 ≠ 0.550.
            #14, #20 (all in bin 0): A+B merged, conf=0.10, acc=0.50
                → ECE=|0.10−0.50|=0.40 ≠ 0.550.
            #15, #19 (first threshold at 0.2): A (0.05) and B (0.15) both
                land below 0.2 → merged → ECE=0.40 ≠ 0.550.
        """
        p = np.concatenate([np.full(100, 0.05), np.full(100, 0.15)])
        y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece == pytest.approx(0.55, abs=1e-10), (
            f"ECE = {ece:.8f}; expected exactly 0.55000000 (bin 0 + bin 1 exact accumulation)."
        )

    def test_ece_coarse_linspace_merges_two_predictions(self) -> None:
        """Correct bin edges keep p=0.12 and p=0.13 together in bin 1.

        Construction:
            Both groups land in bin 1 ([0.1, 0.2)) under correct equal-width
            binning — they are merged into a single calibration bin.
                Group A: p=0.12 (100 samples), y=0 → acc=0.0
                Group B: p=0.13 (100 samples), y=1 → acc=1.0
            Merged: conf=0.125, acc=0.5, error=0.375, weight=1.0
            Expected ECE = 0.375

        Killed mutant:
            #16 (linspace n_bins−1=9 pts → thresholds [0.125, 0.25, ...]):
                A (0.12) < 0.125 → bin 0; B (0.13) ≥ 0.125 → bin 1. Separate:
                    bin 0 contrib = 0.5 × |0.12−0.0| = 0.060
                    bin 1 contrib = 0.5 × |0.13−1.0| = 0.435
                    ECE = 0.495 ≠ 0.375. ✗ Killed.
        """
        p = np.concatenate([np.full(100, 0.12), np.full(100, 0.13)])
        y = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece == pytest.approx(0.375, abs=1e-10), (
            f"ECE = {ece:.8f}; expected exactly 0.37500000 "
            "(p=0.12 and p=0.13 correctly merged into bin 1)."
        )

    def test_ece_fine_linspace_keeps_two_predictions_merged(self) -> None:
        """Correct bin edges keep p=0.050 and p=0.095 together in bin 0.

        Construction:
            Both groups land in bin 0 ([0.0, 0.1)) under correct equal-width
            binning — they are merged.
                Group A: p=0.050 (100 samples), y=1 → acc=1.0
                Group B: p=0.095 (100 samples), y=0 → acc=0.0
            Merged: conf=0.0725, acc=0.5, error=0.4275, weight=1.0
            Expected ECE = 0.4275

        Killed mutant:
            #17 (linspace n_bins+2=12 pts → first threshold ≈ 0.0909):
                A (0.050) < 0.0909 → bin 0; B (0.095) ≥ 0.0909 → bin 1. Separate:
                    bin 0 contrib = 0.5 × |0.050−1.0| = 0.475
                    bin 1 contrib = 0.5 × |0.095−0.0| = 0.0475
                    ECE = 0.5225 ≠ 0.4275. ✗ Killed.
        """
        p = np.concatenate([np.full(100, 0.050), np.full(100, 0.095)])
        y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece == pytest.approx(0.4275, abs=1e-10), (
            f"ECE = {ece:.8f}; expected exactly 0.42750000 "
            "(p=0.050 and p=0.095 correctly merged into bin 0)."
        )


# ------------------------------------------------------------------ #
# Story 3.7 — mutation-targeted reliability_diagram structural tests   #
# ------------------------------------------------------------------ #


class TestReliabilityDiagramStructural:
    """Bar-count tests targeting reliability_diagram binning mutants.

    Story 3.7 found 6 surviving mutants in reliability_diagram binning:

        Mutant 60  — ``bin_ids != k`` (inverted mask: filled bins look empty,
                      empty bins look full)
        Mutants 50, 51 — wrong linspace point count (finer/coarser edges split
                          predictions that belong in one bin)
        Mutants 53, 54 — wrong threshold slice (first threshold missing → bin 0
                          and bin 1 merge; empty threshold array → all in bin 0)
        Mutant 55  — ``bin_edges[1:-2]`` drops the last threshold (0.9), merging
                      bins 8 and 9

    Why count patches?
        Each ``axes.bar(x, y, ...)`` call adds exactly 1 Rectangle to
        ``ax.patches``.  The perfect-calibration diagonal is a Line2D — not
        a patch.  So ``len(ax.patches)`` == number of non-empty bins drawn.
        Binning mutations change which predictions group together, directly
        changing the bar count for carefully constructed inputs.
    """

    def test_bar_count_single_bin(self) -> None:
        """All predictions in one bin → exactly 1 bar.

        Mutant 60 inverts the mask (``bin_ids != k``): the bin holding all
        predictions appears empty (``continue``-d), every other bin appears full.
        9 bars are drawn instead of 1.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        p = np.full(100, 0.15)  # all in bin 1 ([0.1, 0.2))
        y = np.concatenate([np.ones(30, dtype=int), np.zeros(70, dtype=int)])

        fig = reliability_diagram(y, p, n_bins=10)
        ax = fig.axes[0]
        n_bars = len(ax.patches)
        plt.close(fig)

        assert n_bars == 1, (
            f"Expected 1 bar (all predictions in bin 1), got {n_bars}. "
            "n_bars > 1 suggests the bin mask is inverted (bin_ids != k)."
        )

    def test_bar_count_two_adjacent_outer_bins(self) -> None:
        """Predictions in bin 0 and bin 1 → exactly 2 bars.

        Mutants 53 (thresholds start at 0.2) and 54 (empty thresholds) cause
        p=0.05 and p=0.15 to land in the same bin → 1 bar instead of 2.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # p=0.05 → bin 0 ([0.0, 0.1)); p=0.15 → bin 1 ([0.1, 0.2))
        p = np.concatenate([np.full(50, 0.05), np.full(50, 0.15)])
        y = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

        fig = reliability_diagram(y, p, n_bins=10)
        ax = fig.axes[0]
        n_bars = len(ax.patches)
        plt.close(fig)

        assert n_bars == 2, (
            f"Expected 2 bars (bin 0 at p≈0.05 and bin 1 at p≈0.15), got {n_bars}. "
            "n_bars == 1 suggests the first threshold (0.1) is missing or ≥ 0.2."
        )

    def test_bar_count_same_correct_bin_no_split(self) -> None:
        """Two predictions in the same correct bin → exactly 1 bar.

        Mutant 50 (n_bins−1 pts → threshold at 0.125): p=0.115 < 0.125 → bin 0;
            p=0.19 ≥ 0.125 → bin 1. Split → 2 bars.
        Mutant 51 (n_bins+2 pts → threshold at ≈0.1818): p=0.115 → bin 1;
            p=0.19 ≥ 0.1818 → bin 2. Split → 2 bars.
        Correct code: both in bin 1 ([0.1, 0.2)) → merged → 1 bar.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Both in correct bin 1 ([0.1, 0.2)); mutants 50 and 51 split them
        p = np.concatenate([np.full(50, 0.115), np.full(50, 0.19)])
        y = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

        fig = reliability_diagram(y, p, n_bins=10)
        ax = fig.axes[0]
        n_bars = len(ax.patches)
        plt.close(fig)

        assert n_bars == 1, (
            f"Expected 1 bar (p=0.115 and p=0.19 both in correct bin 1), got {n_bars}. "
            "n_bars == 2 suggests a wrong linspace point count split the bin."
        )

    def test_bar_count_high_end_bins_not_merged(self) -> None:
        """Predictions in bins 8 and 9 → exactly 2 bars.

        Mutant 55 uses ``bin_edges[1:-2]``, dropping the last threshold (0.9).
        p=0.85 and p=0.95 land in the same final mutant bin (≥ 0.8) → 1 bar.
        Correct: bin 8 [0.8, 0.9) and bin 9 [0.9, 1.0] → 2 separate bars.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        p = np.concatenate([np.full(50, 0.85), np.full(50, 0.95)])
        y = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

        fig = reliability_diagram(y, p, n_bins=10)
        ax = fig.axes[0]
        n_bars = len(ax.patches)
        plt.close(fig)

        assert n_bars == 2, (
            f"Expected 2 bars (bin 8 at p≈0.85 and bin 9 at p≈0.95), got {n_bars}. "
            "n_bars == 1 suggests the last threshold (0.9) was dropped."
        )
