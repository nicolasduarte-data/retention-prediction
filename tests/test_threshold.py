"""Tests for src/retention/models/threshold.py — Story 3.3.

Coverage scope:
    threshold_sweep        — table shape, decision rule, monotonicity, validation
    optimize_threshold     — KIKO winners, tie-break, criterion dispatch, errors
    _fbeta                 — the recall-weighting identity (why F2 is [FLIP-RISK])
    threshold_sweep_plot   — smoke + Axes-embedding contract

Design principle (mirrors test_evaluation.py):
    Every assertion uses a *constructed* known-input / known-output case you can
    reason about by hand. Threshold logic is a pure computation over a confusion
    matrix — it must be testable without touching the real dataset or a model.

    Two properties are tested instead of fixed outputs because they hold for ALL
    inputs and pin down the confusion-matrix logic precisely:
      - recall is monotone non-increasing as the threshold rises;
      - F2 beats F1 exactly when recall exceeds precision (the recall weighting).
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np
import pytest

from retention.models.threshold import (
    _fbeta,
    optimize_threshold,
    threshold_sweep,
    threshold_sweep_plot,
)


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #


@pytest.fixture()
def perfect_ranker():
    """A perfectly separable problem: 5 exits all out-rank 5 stayers.

    y_true: [1,1,1,1,1, 0,0,0,0,0]
    proba:  [0.90,0.80,0.70,0.60,0.55,  0.45,0.30,0.20,0.10,0.05]

    Every positive scores >= 0.55; every negative scores <= 0.45. So any
    threshold in (0.45, 0.55] separates them perfectly. On the default grid
    (0.05..0.95 step 0.01) the perfect-separation plateau is t ∈ {0.46, …, 0.55}.

    Known properties:
        base rate                       = 5/10 = 0.50
        lowest perfect-separation arm   = 0.46  (0.45 would flag the 0.45 stayer)
        at t=0.05 (all flagged)         → recall 1.0, precision 0.50 (= base rate)
    """
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    p = np.array([0.90, 0.80, 0.70, 0.60, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05])
    return y, p


# ------------------------------------------------------------------ #
# threshold_sweep — table structure & decision rule                    #
# ------------------------------------------------------------------ #


def test_sweep_returns_full_grid(perfect_ranker):
    """Default grid 0.05..0.95 step 0.01 → exactly 91 arms, ascending."""
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)

    assert len(sweep) == 91
    assert sweep["threshold"].iloc[0] == pytest.approx(0.05)
    assert sweep["threshold"].iloc[-1] == pytest.approx(0.95)
    # Strictly ascending, clean grid spacing (no float drift).
    assert np.allclose(np.diff(sweep["threshold"].to_numpy()), 0.01)


def test_sweep_has_expected_columns(perfect_ranker):
    """The experiment log carries every confusion-derived metric per arm."""
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    expected = {
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "youden_j",
    }
    assert set(sweep.columns) == expected


def test_sweep_confusion_counts_conserved(perfect_ranker):
    """tp + fp + fn + tn must equal n at every threshold (no row lost)."""
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    totals = sweep["tp"] + sweep["fp"] + sweep["fn"] + sweep["tn"]
    assert (totals == len(y)).all()


def test_sweep_decision_rule_is_geq(perfect_ranker):
    """At t=0.05 every employee (min proba = 0.05) is flagged → recall 1.0.

    This pins the `>=` decision rule: 0.05 >= 0.05 is True, so the lowest-scoring
    employee is flagged at the lowest threshold. precision there = base rate.
    """
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    first = sweep.iloc[0]
    assert first["recall"] == pytest.approx(1.0)
    assert first["precision"] == pytest.approx(0.5)  # 5 exits / 10 flagged = base rate


def test_sweep_recall_is_monotone_non_increasing(perfect_ranker):
    """Raising the threshold can only flag fewer people → recall never rises.

    A universal property of any sensible threshold sweep; a clean way to catch a
    flipped comparison or a tp/fn mix-up in the confusion logic.
    """
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    recall = sweep["recall"].to_numpy()
    assert np.all(np.diff(recall) <= 1e-12)


# ------------------------------------------------------------------ #
# threshold_sweep — input validation                                   #
# ------------------------------------------------------------------ #


def test_sweep_raises_on_non_finite_proba():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, np.nan, 0.3, 0.9])
    with pytest.raises(ValueError, match="non-finite"):
        threshold_sweep(y, p)


def test_sweep_raises_on_no_positives():
    """All-negative labels → recall/F undefined for every arm → ValueError."""
    y = np.zeros(10, dtype=int)
    p = np.linspace(0.1, 0.9, 10)
    with pytest.raises(ValueError, match="no positive"):
        threshold_sweep(y, p)


def test_sweep_accepts_2d_proba(perfect_ranker):
    """(n, 2) predict_proba output uses column 1 → identical to the 1D sweep."""
    y, p = perfect_ranker
    proba_2d = np.column_stack([1.0 - p, p])
    sweep_1d = threshold_sweep(y, p)
    sweep_2d = threshold_sweep(y, proba_2d)
    assert np.allclose(sweep_1d["f1"].to_numpy(), sweep_2d["f1"].to_numpy())


# ------------------------------------------------------------------ #
# optimize_threshold — winners, tie-break, dispatch                    #
# ------------------------------------------------------------------ #


def test_optimize_returns_grid_value_in_range(perfect_ranker):
    y, p = perfect_ranker
    t = optimize_threshold(y, p, "f1")
    assert 0.05 <= t <= 0.95
    # Lands on the 0.01 grid (within float tolerance).
    assert (round(t * 100) / 100) == pytest.approx(t)


def test_optimize_f1_perfect_separation_achieves_perfect_scores(perfect_ranker):
    """On a separable problem the F1-optimal arm scores precision = recall = 1.

    Verified independently: recompute the confusion matrix at the returned
    threshold rather than trusting the sweep table.
    """
    y, p = perfect_ranker
    t = optimize_threshold(y, p, "f1")
    y_pred = (p >= t).astype(int)
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_optimize_tie_break_picks_lowest_threshold(perfect_ranker):
    """The perfect-separation plateau is 0.46..0.55; the lowest (0.46) wins.

    This is the [FLIP-RISK] tie-break: when arms tie, prefer the higher-recall
    (lower-threshold) one. 0.45 is excluded — it would flag the 0.45 stayer (FP).
    """
    y, p = perfect_ranker
    assert optimize_threshold(y, p, "f1") == pytest.approx(0.46)


def test_optimize_youden_perfect_separation(perfect_ranker):
    """Youden's J = 1 on a separable problem → also lands on the plateau."""
    y, p = perfect_ranker
    t = optimize_threshold(y, p, "youden")
    assert 0.46 <= t <= 0.55


def test_optimize_unknown_criterion_raises(perfect_ranker):
    y, p = perfect_ranker
    with pytest.raises(ValueError, match="criterion must be one of"):
        optimize_threshold(y, p, "banana")


def test_optimize_ev_criterion_points_to_story_3_4(perfect_ranker):
    """'ev' is a deliberate forward-reference, not a silent failure."""
    y, p = perfect_ranker
    with pytest.raises(NotImplementedError, match="Story 3.4"):
        optimize_threshold(y, p, "ev")


# ------------------------------------------------------------------ #
# _fbeta — the recall-weighting identity                               #
# ------------------------------------------------------------------ #


def test_fbeta_f1_is_harmonic_mean():
    """F1 with equal precision/recall is just that shared value."""
    assert _fbeta(0.5, 0.5, beta=1.0) == pytest.approx(0.5)
    assert _fbeta(1.0, 0.5, beta=1.0) == pytest.approx(2 / 3)


def test_fbeta_f2_rewards_recall_over_precision():
    """The crux of why F2 is the [FLIP-RISK] criterion.

    Same F1 (0.667) for both points, but F2 splits them by recall:
        high precision, low recall  (1.0, 0.5) → F2 = 0.556  (penalised)
        low precision, high recall  (0.5, 1.0) → F2 = 0.833  (rewarded)
    """
    f1_high_prec = _fbeta(1.0, 0.5, beta=1.0)
    f1_high_rec = _fbeta(0.5, 1.0, beta=1.0)
    assert f1_high_prec == pytest.approx(f1_high_rec)  # identical F1

    f2_high_prec = _fbeta(1.0, 0.5, beta=2.0)
    f2_high_rec = _fbeta(0.5, 1.0, beta=2.0)
    assert f2_high_prec == pytest.approx(2.5 / 4.5)  # 0.556
    assert f2_high_rec == pytest.approx(2.5 / 3.0)  # 0.833
    assert f2_high_rec > f2_high_prec  # F2 prefers the high-recall arm


def test_fbeta_zero_when_both_zero():
    """precision = recall = 0 → 0.0, not NaN (keeps argmax well-defined)."""
    assert _fbeta(0.0, 0.0, beta=1.0) == 0.0
    assert _fbeta(0.0, 0.0, beta=2.0) == 0.0


# ------------------------------------------------------------------ #
# threshold_sweep_plot — smoke + contract                              #
# ------------------------------------------------------------------ #


def test_plot_returns_figure(perfect_ranker):
    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    t = optimize_threshold(y, p, "f1")
    fig = threshold_sweep_plot(sweep, optimal_threshold=t, model_name="GBM", cohort="hybrid")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_draws_into_provided_axes(perfect_ranker):
    """When an Axes is passed, its parent Figure is returned (subplot embedding)."""
    import matplotlib.pyplot as plt

    y, p = perfect_ranker
    sweep = threshold_sweep(y, p)
    fig, ax = plt.subplots()
    returned = threshold_sweep_plot(sweep, ax=ax)
    assert returned is fig
    plt.close(fig)
