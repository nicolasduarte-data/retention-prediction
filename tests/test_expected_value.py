"""Tests for src/retention/evaluation/expected_value.py — Story 3.4.

Coverage scope:
    ev_at_threshold          — known dollar value, FN-cancellation, negativity,
                               defaults, 2D proba, input validation
    breakeven_p_eff          — closed form, cost-ratio/precision identity,
                               >1 (infeasible) and ∞ (no TP) cases, EV sign-cross
    p_eff_sensitivity_sweep  — table shape, linearity in p_eff, single-source-of-
                               truth consistency, custom grid, validation
    replacement_cost_from_salary — SHRM 1.5× rule + negative guard
    expected_value_plot      — smoke + Axes-embedding + all three breakeven branches

Design principle (mirrors test_threshold.py):
    Every assertion is a *constructed* known-input / known-output case you can
    reason about by hand. EV is a pure computation over a confusion matrix and
    three scalars — it must be testable without the real dataset or a model.

    Two properties are tested because they hold for ALL inputs and pin the
    economics precisely:
      - EV is invariant to false negatives (the FN term cancels vs. do-nothing);
      - EV is linear in p_eff at a fixed threshold (TP, FP held constant).
"""

from __future__ import annotations

import math

import matplotlib.figure
import numpy as np
import pytest

from retention.evaluation.expected_value import (
    DEFAULT_INTERVENTION_COST,
    DEFAULT_P_EFF,
    DEFAULT_REPLACEMENT_COST,
    REPLACEMENT_COST_MULTIPLIER,
    breakeven_p_eff,
    ev_at_threshold,
    expected_value_plot,
    p_eff_sensitivity_sweep,
    replacement_cost_from_salary,
)


# ------------------------------------------------------------------ #
# Shared fixtures                                                       #
# ------------------------------------------------------------------ #


@pytest.fixture()
def known_confusion():
    """A 4-row case with a confusion matrix you can read off by eye.

    y_true: [1, 1, 0, 0]
    proba:  [0.9, 0.8, 0.7, 0.1]

    At threshold 0.5 (decision rule ``proba >= t``) the flag list is
    {0.9, 0.8, 0.7}; only 0.1 is left alone:

        idx  proba  true  flagged?  cell
        0    0.9    1     yes       TP
        1    0.8    1     yes       TP
        2    0.7    0     yes       FP
        3    0.1    0     no        TN

    → tp=2, fp=1, fn=0, tn=1; precision = 2/3, recall = 1.0.

    With p_eff=0.5, rc=100, ic=10 the hand-computed answers are:
        EV        = 0.5·100·2 − 10·(2+1) = 100 − 30 = 70
        breakeven = 10·(2+1) / (100·2)   = 30/200   = 0.15
    """
    y = np.array([1, 1, 0, 0], dtype=int)
    p = np.array([0.9, 0.8, 0.7, 0.1])
    return y, p


# ------------------------------------------------------------------ #
# ev_at_threshold — known values & the FN-cancellation property        #
# ------------------------------------------------------------------ #


def test_ev_known_value(known_confusion):
    """EV = p_eff·rc·TP − ic·(TP+FP). Hand-checked: 0.5·100·2 − 10·3 = 70."""
    y, p = known_confusion
    ev = ev_at_threshold(y, p, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert ev == pytest.approx(70.0)


def test_ev_zero_at_breakeven_p_eff(known_confusion):
    """At p_eff = breakeven (0.15 here) benefit exactly offsets cost → EV 0."""
    y, p = known_confusion
    ev = ev_at_threshold(y, p, 0.5, p_eff=0.15, replacement_cost=100.0, intervention_cost=10.0)
    assert ev == pytest.approx(0.0)


def test_ev_can_be_negative_when_flagging_only_stayers():
    """Flag two stayers, no real exits → benefit 0, pure intervention loss."""
    y = np.array([0, 0, 1], dtype=int)
    p = np.array([0.9, 0.8, 0.1])
    # t=0.5 flags 0.9 & 0.8 (both stayers) → tp=0, fp=2 → EV = 0 − 10·2 = −20.
    ev = ev_at_threshold(y, p, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert ev == pytest.approx(-20.0)


def test_ev_invariant_to_false_negatives(known_confusion):
    """The central claim of the module: the FN term cancels vs. do-nothing.

    Add an unflagged true exit (a false negative) and an unflagged stayer (a true
    negative). TP and FP are unchanged, so EV must be *identical* — a missed exit
    costs the same with or without the model and cannot change its added value.
    A formula that subtracted ``FN·rc`` would fail this test.
    """
    y, p = known_confusion
    ev_base = ev_at_threshold(y, p, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0)

    # Extra exit at 0.2 (FN) and extra stayer at 0.3 (TN); both below 0.5 → unflagged.
    y2 = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    p2 = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    ev_with_fn = ev_at_threshold(
        y2, p2, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0
    )

    assert ev_with_fn == pytest.approx(ev_base)


def test_ev_defaults_match_documented_constants():
    """Guard the cited economics: SHRM 1.5×$60k, $2k conversation, p_eff 0.30.

    These exact numbers are quoted in docs/methodology.md and the ticket win
    conditions. If a refactor changes a default this test fails loudly, rather
    than letting the methodology citation drift silently out of sync.
    """
    assert REPLACEMENT_COST_MULTIPLIER == pytest.approx(1.5)
    assert DEFAULT_REPLACEMENT_COST == pytest.approx(90_000.0)
    assert DEFAULT_INTERVENTION_COST == pytest.approx(2_000.0)
    assert DEFAULT_P_EFF == pytest.approx(0.30)


def test_ev_uses_defaults_when_costs_omitted(known_confusion):
    """Calling without cost kwargs equals an explicit call with the defaults."""
    y, p = known_confusion
    implicit = ev_at_threshold(y, p, 0.5)
    explicit = ev_at_threshold(
        y,
        p,
        0.5,
        p_eff=DEFAULT_P_EFF,
        replacement_cost=DEFAULT_REPLACEMENT_COST,
        intervention_cost=DEFAULT_INTERVENTION_COST,
    )
    assert implicit == pytest.approx(explicit)


def test_ev_accepts_2d_proba(known_confusion):
    """(n, 2) predict_proba output uses column 1 → identical to the 1D EV."""
    y, p = known_confusion
    proba_2d = np.column_stack([1.0 - p, p])
    ev_1d = ev_at_threshold(y, p, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0)
    ev_2d = ev_at_threshold(
        y, proba_2d, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0
    )
    assert ev_2d == pytest.approx(ev_1d)


def test_ev_decision_rule_is_inclusive_at_threshold():
    """A probability exactly equal to the threshold is FLAGGED (`>=`, not `>`).

    Story 3.7: mutation testing showed the decision rule `proba >= threshold`
    → `proba > threshold` survived — no EV test had a probability sitting
    exactly on the threshold, so the documented inclusivity was unverified.

    Construction: two true exits at proba == 0.5 with threshold 0.5.
        Inclusive (`>=`): both flagged → tp=2, fp=0 → EV = 0.5·100·2 − 10·2 = 80
        Exclusive (`>`) : neither flagged → tp=0 → EV = 0
    Asserting EV == 80 kills the `>=` → `>` mutant.
    """
    y = np.array([1, 1, 0], dtype=int)
    p = np.array([0.5, 0.5, 0.4])  # two exits sit exactly on the threshold
    ev = ev_at_threshold(y, p, 0.5, p_eff=0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert ev == pytest.approx(80.0)


# ------------------------------------------------------------------ #
# breakeven_p_eff — closed form & boundary cases                       #
# ------------------------------------------------------------------ #


def test_breakeven_known_value(known_confusion):
    """p_eff* = ic·(TP+FP)/(rc·TP) = 10·3/(100·2) = 0.15."""
    y, p = known_confusion
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert be == pytest.approx(0.15)


def test_breakeven_equals_cost_ratio_over_precision(known_confusion):
    """The closed form: breakeven = (ic/rc) / precision. precision here = 2/3."""
    y, p = known_confusion
    rc, ic = 100.0, 10.0
    precision = 2 / 3
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=rc, intervention_cost=ic)
    assert be == pytest.approx((ic / rc) / precision)


def test_breakeven_above_one_when_too_imprecise(known_confusion):
    """precision (2/3) below the cost ratio (1.0) → breakeven 1.5, returned unclamped.

    A breakeven above 1 is the formula's honest way of saying 'this threshold
    cannot pay off even with a perfect intervention'. We must NOT clamp it.
    """
    y, p = known_confusion
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=100.0, intervention_cost=100.0)
    assert be == pytest.approx(1.5)
    assert be > 1.0


def test_breakeven_infinite_when_no_true_exit_flagged():
    """tp == 0 → no effectiveness can make the program pay → math.inf."""
    y = np.array([0, 0, 1], dtype=int)
    p = np.array([0.9, 0.8, 0.1])  # t=0.5 flags only the two stayers → tp=0, fp=2
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert math.isinf(be)


def test_breakeven_is_the_ev_zero_crossing(known_confusion):
    """EV is negative just below breakeven and positive just above it.

    Ties the two functions together: the value from breakeven_p_eff must be the
    actual zero-crossing of ev_at_threshold, not an independently-derived number.
    """
    y, p = known_confusion
    rc, ic = 100.0, 10.0
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=rc, intervention_cost=ic)
    ev_below = ev_at_threshold(
        y, p, 0.5, p_eff=be - 0.05, replacement_cost=rc, intervention_cost=ic
    )
    ev_above = ev_at_threshold(
        y, p, 0.5, p_eff=be + 0.05, replacement_cost=rc, intervention_cost=ic
    )
    assert ev_below < 0.0 < ev_above


# ------------------------------------------------------------------ #
# p_eff_sensitivity_sweep — shape, linearity, single source of truth   #
# ------------------------------------------------------------------ #


def test_sweep_shape_and_columns(known_confusion):
    """Default sweep → 9 rows over [0.1, 0.9], columns [p_eff, expected_value]."""
    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    assert list(sweep.columns) == ["p_eff", "expected_value"]
    assert len(sweep) == 9
    assert sweep["p_eff"].iloc[0] == pytest.approx(0.1)
    assert sweep["p_eff"].iloc[-1] == pytest.approx(0.9)
    assert np.allclose(np.diff(sweep["p_eff"].to_numpy()), 0.1)


def test_sweep_is_linear_in_p_eff(known_confusion):
    """At a fixed threshold TP and FP are constant → EV is linear in p_eff.

    Equal p_eff steps must produce equal EV steps. Slope = rc·TP·Δp_eff =
    100·2·0.1 = 20 here. A bent line would mean the confusion matrix drifted
    across the sweep — a bug in how the threshold is applied per grid point.
    """
    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    ev_diffs = np.diff(sweep["expected_value"].to_numpy())
    assert np.allclose(ev_diffs, 20.0)


def test_sweep_rows_match_ev_at_threshold(known_confusion):
    """Single source of truth: every row equals a direct ev_at_threshold call.

    Guarantees the sweep, the breakeven, and the headline EV can never drift
    apart — they all route through the one formula.
    """
    y, p = known_confusion
    rc, ic = 100.0, 10.0
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=rc, intervention_cost=ic)
    for _, row in sweep.iterrows():
        direct = ev_at_threshold(
            y, p, 0.5, p_eff=float(row["p_eff"]), replacement_cost=rc, intervention_cost=ic
        )
        assert row["expected_value"] == pytest.approx(direct)


def test_sweep_custom_grid(known_confusion):
    """n_points and p_eff_range are honoured exactly."""
    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(
        y,
        p,
        0.5,
        p_eff_range=(0.2, 0.4),
        n_points=3,
        replacement_cost=100.0,
        intervention_cost=10.0,
    )
    assert len(sweep) == 3
    assert np.allclose(sweep["p_eff"].to_numpy(), [0.2, 0.3, 0.4])


# ------------------------------------------------------------------ #
# replacement_cost_from_salary — the SHRM helper                       #
# ------------------------------------------------------------------ #


def test_replacement_cost_from_salary():
    """SHRM 1.5× rule: $60k salary → $90k replacement cost."""
    assert replacement_cost_from_salary(60_000.0) == pytest.approx(90_000.0)
    assert replacement_cost_from_salary(1_000.0) == pytest.approx(
        REPLACEMENT_COST_MULTIPLIER * 1_000.0
    )


def test_replacement_cost_from_salary_rejects_negative():
    with pytest.raises(ValueError, match="annual_salary must be >= 0"):
        replacement_cost_from_salary(-1.0)


# ------------------------------------------------------------------ #
# Input validation                                                     #
# ------------------------------------------------------------------ #


def test_ev_rejects_non_finite_proba():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, np.nan, 0.3, 0.1])
    with pytest.raises(ValueError, match="non-finite"):
        ev_at_threshold(y, p, 0.5)


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, 2.0])
def test_ev_rejects_threshold_out_of_range(known_confusion, bad_threshold):
    y, p = known_confusion
    with pytest.raises(ValueError, match="threshold must be in"):
        ev_at_threshold(y, p, bad_threshold)


@pytest.mark.parametrize("bad_p_eff", [-0.01, 1.01])
def test_ev_rejects_p_eff_out_of_range(known_confusion, bad_p_eff):
    y, p = known_confusion
    with pytest.raises(ValueError, match="p_eff must be in"):
        ev_at_threshold(y, p, 0.5, p_eff=bad_p_eff)


def test_ev_rejects_negative_costs(known_confusion):
    y, p = known_confusion
    with pytest.raises(ValueError, match="replacement_cost must be >= 0"):
        ev_at_threshold(y, p, 0.5, replacement_cost=-1.0)
    with pytest.raises(ValueError, match="intervention_cost must be >= 0"):
        ev_at_threshold(y, p, 0.5, intervention_cost=-1.0)


def test_breakeven_rejects_zero_replacement_cost(known_confusion):
    """Breakeven divides by replacement_cost → zero is rejected explicitly."""
    y, p = known_confusion
    with pytest.raises(ValueError, match="replacement_cost must be > 0"):
        breakeven_p_eff(y, p, 0.5, replacement_cost=0.0)


def test_sweep_rejects_bad_range(known_confusion):
    y, p = known_confusion
    with pytest.raises(ValueError, match="p_eff_range"):
        p_eff_sensitivity_sweep(y, p, 0.5, p_eff_range=(0.9, 0.1))


def test_sweep_rejects_too_few_points(known_confusion):
    y, p = known_confusion
    with pytest.raises(ValueError, match="n_points"):
        p_eff_sensitivity_sweep(y, p, 0.5, n_points=1)


# ------------------------------------------------------------------ #
# expected_value_plot — smoke + contract + all three breakeven branches #
# ------------------------------------------------------------------ #


def test_plot_returns_figure(known_confusion):
    """In-range breakeven (0.15 ∈ [0.1, 0.9]) draws the dashed vertical line."""
    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    fig = expected_value_plot(sweep, breakeven=be, threshold=0.5, model_name="GBM", cohort="hybrid")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_draws_into_provided_axes(known_confusion):
    """When an Axes is passed, its parent Figure is returned (subplot embedding)."""
    import matplotlib.pyplot as plt

    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    fig, ax = plt.subplots()
    returned = expected_value_plot(sweep, ax=ax)
    assert returned is fig
    plt.close(fig)


def test_plot_handles_below_range_breakeven(known_confusion):
    """A breakeven below the plotted range hits the 'profitable across all' branch."""
    import matplotlib.pyplot as plt

    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    fig = expected_value_plot(sweep, breakeven=0.01)  # 0.01 < lo (0.1)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_handles_infinite_breakeven():
    """A flag-only-stayers sweep has infinite breakeven; the plot must still render.

    Exercises the 'too imprecise to fund' annotation path (shown as ∞), not raise.
    """
    import matplotlib.pyplot as plt

    y = np.array([0, 0, 1], dtype=int)
    p = np.array([0.9, 0.8, 0.1])
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    be = breakeven_p_eff(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)  # inf
    fig = expected_value_plot(sweep, breakeven=be)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_handles_above_range_finite_breakeven(known_confusion):
    """A FINITE breakeven above the plotted range hits the `f"{breakeven:.2f}"`
    annotation branch (feast T2-EV-1) — previously only the ∞ side was tested.
    """
    import matplotlib.pyplot as plt

    y, p = known_confusion
    sweep = p_eff_sensitivity_sweep(y, p, 0.5, replacement_cost=100.0, intervention_cost=10.0)
    fig = expected_value_plot(sweep, breakeven=1.5)  # 1.5 > hi (0.9), finite
    ax = fig.axes[0]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "1.50" in texts and "above plotted range" in texts
    plt.close(fig)


def test_plot_rejects_malformed_sweep():
    """An empty or wrong-shaped sweep raises a clear error, not an opaque one
    (feast T2-EV-1)."""
    import pandas as pd

    with pytest.raises(ValueError, match="non-empty DataFrame"):
        expected_value_plot(pd.DataFrame(columns=["p_eff", "expected_value"]))
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        expected_value_plot(pd.DataFrame({"wrong": [1, 2]}))
