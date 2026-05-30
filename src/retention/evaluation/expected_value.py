"""Expected-value framing for the operating-point decision — Story 3.4.

Story 3.3 chose an operating threshold by *statistical* criteria (F1, F2,
Youden). Every one of those is unit-free — it ranks thresholds without ever
asking what a false positive or a missed exit actually costs. F2 encodes the
[FLIP-RISK] asymmetry (recall matters more than precision) but it has no *brake*:
on this dataset F2 slides to the grid floor and flags 92 % of the workforce,
because nothing in an F-score penalises a wasted retention conversation in
dollars. This module supplies the brake. It puts the decision in money.

────────────────────────────────────────────────────────────────────────────
THE MODEL — expected value of *running the program* vs. doing nothing
────────────────────────────────────────────────────────────────────────────
For each employee we either flag them (predicted exit ≥ threshold → a retention
conversation) or we don't. Three numbers price the consequences:

    replacement_cost  (rc) — fully-loaded cost to backfill a departed employee.
                            SHRM (2024) rule of thumb: ≈ 1.5 × annual salary.
    intervention_cost (ic) — manager + HRBP time for one retention conversation.
                            ≈ $2,000.
    p_eff                  — effectiveness: of the genuine would-leavers we flag
                            and talk to, the fraction we actually retain. ≈ 0.30.

Account for every confusion-matrix cell *relative to the do-nothing baseline*
(flag no-one, eat every exit's replacement cost):

    cell   outcome under the model                         value vs. baseline
    ────   ──────────────────────────────────────────────  ──────────────────
    TP     flag a true exit; retain w.p. p_eff             +p_eff·rc − ic
    FP     flag someone who'd have stayed; waste the chat   −ic
    FN     miss a true exit (same as baseline)              0
    TN     correctly leave a stayer alone                   0

Summing the only two non-zero cells:

    ┌────────────────────────────────────────────────────────────┐
    │   EV = p_eff · replacement_cost · TP  −  intervention_cost · (TP + FP)   │
    └────────────────────────────────────────────────────────────┘

The **FN term cancels** — a missed exit costs the same replacement dollars
whether or not the model exists, so it cannot be part of the model's *added*
value. This is the single most important line in the module, and it is exactly
where the naive textbook formula goes wrong: writing `… − FN · rc` double-counts.
It credits a flagged true-exit both the `p_eff·rc` retention benefit *and* the
full avoided `rc` (by lifting them out of the FN penalty) — yet a flagged exit
is only saved `p_eff` of the time; the `(1 − p_eff)` who leave anyway still cost
`rc`, which the naive formula records as $0. That overstatement (≈ rc + ic per
catch) stampedes the EV-optimal threshold toward "flag everyone" for a reason
that is an accounting error, not an economic truth. We do not make that error.
(The absolute scale of unprevented loss, FN · rc, is real and worth *reporting* —
just as context beside the EV, never inside the objective being optimised.)

────────────────────────────────────────────────────────────────────────────
THE PAYOFF — breakeven effectiveness is governed by precision
────────────────────────────────────────────────────────────────────────────
Set EV = 0 and solve for the effectiveness at which the program starts paying:

    p_eff*  =  intervention_cost · (TP + FP)        intervention_cost / replacement_cost
            ───────────────────────────────  =  ─────────────────────────────────────
                 replacement_cost · TP                        precision

Two readings of the same closed form:
  • As a number: at the F1 threshold from Story 3.3 (precision ≈ 0.34), with
    ic = $2k against rc ≈ $90k, p_eff* ≈ 0.065 — the program pays for itself if
    retention conversations work even ~6–7 % of the time. A low, very crossable
    bar; that is the headline.
  • As a principle: **breakeven falls as precision rises.** A cleaner flag list
    wastes less budget on false positives, so it tolerates *less* effective
    interventions. This is the economic argument for precision@k (Story 3.1),
    and it is the natural counterweight to 3.3's F2 recall pressure — recall
    fills the flag list, precision is what makes funding it defensible.

────────────────────────────────────────────────────────────────────────────
WHY THE SENSITIVITY SWEEP (and not one number)
────────────────────────────────────────────────────────────────────────────
p_eff is the one parameter we cannot measure without running the intervention
program itself — it is a forward assumption, not a dataset fact. Reporting a
single EV at a single assumed p_eff is the move that makes most public EV
write-ups indefensible. So we never report one number: we sweep p_eff across its
plausible range [0.1, 0.9] and show the whole line, plus the breakeven crossing.
The honest claim is not "the model is worth $X" — it is "the model pays off for
any intervention effectiveness above p_eff*, and here is where that sits."

────────────────────────────────────────────────────────────────────────────
LEAKAGE DISCIPLINE — same rule as the threshold sweep
────────────────────────────────────────────────────────────────────────────
EV is computed on the **validation** split, from the same frozen probabilities
the threshold sweep used. It informs the operating-point decision; the test set
is confirmed once, at champion selection (Story 3.6). EV numbers here are
validation-set numbers.

────────────────────────────────────────────────────────────────────────────
API
────────────────────────────────────────────────────────────────────────────
    ev_at_threshold(...)         → float, EV in dollars at one operating point
    p_eff_sensitivity_sweep(...) → DataFrame, EV across the p_eff grid
    breakeven_p_eff(...)         → float, the p_eff where EV crosses zero
    expected_value_plot(...)     → matplotlib Figure of the sweep + breakeven
    replacement_cost_from_salary(...) → float, the SHRM 1.5× helper

Story 3.4 — see `backlog/epic-3-evaluation-rigor.md`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import numpy as np

from retention.evaluation._validation import extract_positive_proba

if TYPE_CHECKING:
    import pandas as pd


# ── Default economics (defaults + citations per Story 3.4.2) ────────────────
# SHRM (2024): the fully-loaded cost to replace an employee — recruiting,
# onboarding, lost productivity ramp — runs roughly 1.5× their annual salary.
REPLACEMENT_COST_MULTIPLIER: float = 1.5
# A representative annual salary, used only to give the module a self-contained
# default. The figure script substitutes the cohort's actual mean salary so the
# reported dollars reflect this workforce, not a placeholder.
_REPRESENTATIVE_ANNUAL_SALARY: float = 60_000.0
DEFAULT_REPLACEMENT_COST: float = REPLACEMENT_COST_MULTIPLIER * _REPRESENTATIVE_ANNUAL_SALARY
# One retention conversation ≈ manager + HRBP preparation and meeting time.
DEFAULT_INTERVENTION_COST: float = 2_000.0
# Effectiveness: fraction of flagged-and-treated genuine exits we actually keep.
# A deliberately modest central assumption — the sensitivity sweep is what makes
# the framing defensible, not this single value.
DEFAULT_P_EFF: float = 0.30


def replacement_cost_from_salary(annual_salary: float) -> float:
    """Fully-loaded replacement cost from a salary via the SHRM 1.5× rule.

    A one-line helper so the figure script (and any caller with a real salary
    figure) derives `replacement_cost` from the same documented multiplier
    rather than hard-coding a dollar number.

    Args:
        annual_salary: Gross annual salary in dollars.

    Returns:
        ``REPLACEMENT_COST_MULTIPLIER × annual_salary`` (SHRM 2024).

    Raises:
        ValueError: if `annual_salary` is negative.
    """
    if annual_salary < 0.0:
        raise ValueError(f"annual_salary must be >= 0, got {annual_salary}.")
    return REPLACEMENT_COST_MULTIPLIER * annual_salary


def _confusion_at_threshold(
    y_true: np.ndarray,  # type: ignore[type-arg]
    proba: np.ndarray,  # type: ignore[type-arg]
    threshold: float,
) -> tuple[int, int, int]:
    """Return (tp, fp, fn) when everyone with ``proba >= threshold`` is flagged.

    Recomputed locally rather than imported from ``models/threshold.py`` so the
    EV module stays self-contained and does not couple to another module's
    private helper. The decision rule is ``>=`` — identical to the threshold
    sweep — so an operating point chosen there means the same thing here.

    EV needs only `tp` and `fp` (the flagged cells); `fn` is returned for the
    separately-reported absolute miss-cost, never for the EV objective itself.
    """
    y_pred = (proba >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, fn


def _validate_threshold(threshold: float) -> None:
    """Reject thresholds outside the probability range [0, 1]."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")


def _validate_costs(
    replacement_cost: float,
    intervention_cost: float,
    *,
    require_positive_replacement: bool = False,
) -> None:
    """Reject negative costs (and a zero replacement cost where it would divide)."""
    if replacement_cost < 0.0:
        raise ValueError(f"replacement_cost must be >= 0, got {replacement_cost}.")
    if require_positive_replacement and replacement_cost == 0.0:
        raise ValueError("replacement_cost must be > 0 to compute breakeven p_eff.")
    if intervention_cost < 0.0:
        raise ValueError(f"intervention_cost must be >= 0, got {intervention_cost}.")


def ev_at_threshold(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    threshold: float,
    *,
    p_eff: float = DEFAULT_P_EFF,
    replacement_cost: float = DEFAULT_REPLACEMENT_COST,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
) -> float:
    """Expected dollar value of the flag-and-intervene program vs. doing nothing.

        EV = p_eff · replacement_cost · TP − intervention_cost · (TP + FP)

    A positive EV means running the model and holding retention conversations
    with everyone it flags saves more (in retained replacement costs) than it
    spends (in conversation time). The figure is measured against the do-nothing
    baseline, so false negatives — exits we never flag — do not appear: they cost
    the same with or without the model and so add nothing to its value.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool). Shape (n_samples,).
        y_proba: Predicted probabilities for the positive class (P(exit=1)).
            Shape (n_samples,) or (n_samples, 2); if 2D, column 1 is used.
        threshold: Operating point — flag every employee with ``proba >= threshold``.
        p_eff: Intervention effectiveness in [0, 1] — fraction of flagged genuine
            exits actually retained. Default 0.30.
        replacement_cost: Dollar cost to backfill one departed employee.
            Default ``1.5 × $60k`` (SHRM 2024 multiplier × representative salary).
        intervention_cost: Dollar cost of one retention conversation. Default $2,000.

    Returns:
        Expected value in dollars (can be negative — a loss-making operating point).

    Raises:
        ValueError: if `y_proba` is non-finite, `threshold` ∉ [0, 1],
            `p_eff` ∉ [0, 1], or a cost is negative.

    Teaching note — only TP and FP enter the sum:
        Because FN and TN both contribute zero value-over-baseline, the EV depends
        solely on the *flagged* population. That is the formula telling you
        something true: this is the expected value of the act of *flagging*, and
        its whole quality is governed by who lands in the flag list.
    """
    proba = extract_positive_proba(y_proba)
    _validate_threshold(threshold)
    if not 0.0 <= p_eff <= 1.0:
        raise ValueError(f"p_eff must be in [0, 1], got {p_eff}.")
    _validate_costs(replacement_cost, intervention_cost)

    y_arr = np.asarray(y_true).astype(int)
    tp, fp, _fn = _confusion_at_threshold(y_arr, proba, threshold)

    benefit = p_eff * replacement_cost * tp  # retained exits we'd otherwise replace
    cost = intervention_cost * (tp + fp)  # every conversation we hold (all flagged)
    return float(benefit - cost)


def p_eff_sensitivity_sweep(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    threshold: float,
    *,
    p_eff_range: tuple[float, float] = (0.1, 0.9),
    n_points: int = 9,
    replacement_cost: float = DEFAULT_REPLACEMENT_COST,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
) -> pd.DataFrame:
    """Tabulate EV across a grid of intervention-effectiveness assumptions.

    p_eff is a forward assumption we cannot read off the data, so instead of
    betting on one value we report EV across the plausible range and let the
    reader locate the breakeven crossing. The operating threshold is held fixed,
    so TP and FP are constant and **EV is linear in p_eff** — the table traces a
    straight line whose slope is ``replacement_cost · TP`` and whose intercept is
    ``−intervention_cost · (TP + FP)``.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class. 1D or (n, 2).
        threshold: Fixed operating point for the whole sweep.
        p_eff_range: (low, high) effectiveness bounds. Default (0.1, 0.9).
        n_points: Number of grid points, inclusive of both ends. Default 9
            → steps of 0.1 across [0.1, 0.9].
        replacement_cost: Dollar cost to backfill one employee. Default ``1.5 × $60k``.
        intervention_cost: Dollar cost of one retention conversation. Default $2,000.

    Returns:
        A DataFrame with columns ``p_eff`` and ``expected_value`` (dollars),
        one row per grid point, ascending in p_eff.

    Raises:
        ValueError: if the range is not ``0 <= low < high <= 1``, `n_points` < 2,
            or any input fails `ev_at_threshold`'s validation.
    """
    import pandas as pd  # noqa: PLC0415

    low, high = p_eff_range
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"p_eff_range must satisfy 0 <= low < high <= 1, got {p_eff_range}.")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")

    # Each grid point reuses ev_at_threshold — one source of truth for the
    # formula, so the sweep, the breakeven, and the headline number can never
    # drift apart. The repeated confusion recompute is negligible at n_points≈9.
    p_eff_grid = np.linspace(low, high, n_points)
    rows: list[dict[str, float]] = [
        {
            "p_eff": float(p_eff),
            "expected_value": ev_at_threshold(
                y_true,
                y_proba,
                threshold,
                p_eff=float(p_eff),
                replacement_cost=replacement_cost,
                intervention_cost=intervention_cost,
            ),
        }
        for p_eff in p_eff_grid
    ]
    return pd.DataFrame(rows)


def breakeven_p_eff(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    threshold: float,
    *,
    replacement_cost: float = DEFAULT_REPLACEMENT_COST,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
) -> float:
    """The intervention effectiveness at which EV crosses zero, at a fixed threshold.

        EV = 0  ⟺  p_eff* = intervention_cost · (TP + FP) / (replacement_cost · TP)
                          = (intervention_cost / replacement_cost) / precision

    Below ``p_eff*`` the program loses money; at or above it, it pays. The closed
    form exposes the lever directly: breakeven is the cost ratio divided by the
    precision at this threshold, so a more precise flag list needs *less* effective
    interventions to be worth funding.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class. 1D or (n, 2).
        threshold: Operating point whose breakeven we want.
        replacement_cost: Dollar cost to backfill one employee. Must be > 0.
        intervention_cost: Dollar cost of one retention conversation. Default $2,000.

    Returns:
        The breakeven ``p_eff`` as a float.
        * A value in (0, 1] is the effectiveness the program needs to pay off.
        * A value > 1.0 means the operating point cannot break even *even with a
          perfect intervention* — its precision is below the cost ratio.
        * ``math.inf`` when TP == 0 (no genuine exits flagged): no effectiveness
          level can make the program profitable.

    Raises:
        ValueError: if `y_proba` is non-finite, `threshold` ∉ [0, 1],
            `replacement_cost` <= 0, or `intervention_cost` < 0.

    Teaching note — why values can exceed 1.0:
        p_eff is a probability, so a "breakeven" above 1.0 is not a reachable
        operating assumption — it is the formula's honest way of saying *this
        threshold is too imprecise to fund at these costs*. We return the raw
        value rather than clamping it, so the caller can see how far past
        feasibility a bad operating point sits.
    """
    proba = extract_positive_proba(y_proba)
    _validate_threshold(threshold)
    _validate_costs(replacement_cost, intervention_cost, require_positive_replacement=True)

    y_arr = np.asarray(y_true).astype(int)
    tp, fp, _fn = _confusion_at_threshold(y_arr, proba, threshold)

    if tp == 0:
        # No true exits in the flag list → the benefit term is identically zero,
        # so EV = −intervention_cost·FP ≤ 0 for every p_eff. Never profitable.
        return math.inf
    return float(intervention_cost * (tp + fp) / (replacement_cost * tp))


def expected_value_plot(
    sweep: pd.DataFrame,
    *,
    breakeven: float | None = None,
    threshold: float | None = None,
    model_name: str = "",
    cohort: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot EV against p_eff, with the breakeven crossing and the profitable region.

    The chart makes the central honesty of the framing visible: a straight EV
    line, the ``EV = 0`` axis, and the breakeven p_eff where the line crosses it.
    Everything to the right of breakeven is shaded as profitable, everything to
    the left as loss-making — so a reader sees at a glance how much of the
    plausible effectiveness range pays off.

    Args:
        sweep: The DataFrame from `p_eff_sensitivity_sweep` (columns
            ``p_eff``, ``expected_value``).
        breakeven: If given and within the plotted range, drawn as a dashed
            vertical line. If below the range, annotated as "profitable across the
            whole range"; if above 1 / infinite, annotated as infeasible. Pass the
            output of `breakeven_p_eff`.
        threshold: Optional operating-point value, for the title only.
        model_name: Optional model label for the title (e.g. 'GBM').
        cohort: Optional cohort label for the title (e.g. 'hybrid').
        ax: Optional matplotlib Axes to draw on. If None, a new Figure and Axes
            are created.

    Returns:
        The matplotlib Figure. The caller owns the lifecycle — save with
        ``fig.savefig(...)`` and free with ``plt.close(fig)`` when looping.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.ticker as mticker  # noqa: PLC0415

    # Validate the sweep up front (feast T2-EV-1): an empty frame makes
    # `p_eff.min()` raise an opaque "zero-size array" error, and a wrong-shaped
    # one raises a bare KeyError. Fail with a message that names the contract.
    if sweep.empty or not {"p_eff", "expected_value"} <= set(sweep.columns):
        raise ValueError(
            "sweep must be a non-empty DataFrame with columns 'p_eff' and "
            "'expected_value' (the output of p_eff_sensitivity_sweep)."
        )

    # Direct if/else (not a stored predicate) so mypy narrows `ax` from
    # `Axes | None` to `Axes` in the else branch — same pattern as calibration.py.
    if ax is None:
        _fig, _ax = plt.subplots(figsize=(7, 5))
        fig: matplotlib.figure.Figure = _fig
        axes: matplotlib.axes.Axes = _ax
    else:
        axes = ax
        _raw_fig = axes.get_figure()
        assert _raw_fig is not None, "ax.get_figure() returned None"
        fig = _raw_fig  # type: ignore[assignment]

    p_eff = sweep["p_eff"].to_numpy()
    ev = sweep["expected_value"].to_numpy()

    # The EV line and the break-even axis it crosses.
    axes.plot(p_eff, ev, color="#4c72b0", linewidth=2.2, label="Expected value", zorder=3)
    axes.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)

    # Shade profitable (EV > 0) green and loss-making (EV <= 0) red. `where`
    # masks the fill to each region; interpolate=True closes the wedge cleanly
    # at the zero crossing.
    axes.fill_between(p_eff, 0.0, ev, where=(ev > 0), color="#55a868", alpha=0.15, interpolate=True)
    axes.fill_between(
        p_eff, ev, 0.0, where=(ev <= 0), color="#c44e52", alpha=0.12, interpolate=True
    )

    # Break-even marker — three honest cases depending on where it falls.
    lo, hi = float(p_eff.min()), float(p_eff.max())
    if breakeven is not None:
        if math.isfinite(breakeven) and lo <= breakeven <= hi:
            axes.axvline(
                breakeven,
                color="black",
                linestyle="--",
                linewidth=1.4,
                zorder=4,
                label=f"Breakeven p_eff = {breakeven:.2f}",
            )
        elif math.isfinite(breakeven) and breakeven < lo:
            axes.text(
                0.04,
                0.94,
                f"Breakeven p_eff = {breakeven:.3f} (below plotted range)\n"
                f"→ profitable across all plausible effectiveness",
                transform=axes.transAxes,
                fontsize=8,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#e8f3ec", "alpha": 0.9},
            )
        else:  # > hi, > 1, or infinite → cannot fund at these costs
            shown = "∞" if not math.isfinite(breakeven) else f"{breakeven:.2f}"
            axes.text(
                0.04,
                0.94,
                f"Breakeven p_eff = {shown} (above plotted range)\n"
                f"→ too imprecise to fund at these costs",
                transform=axes.transAxes,
                fontsize=8,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f6e7e6", "alpha": 0.9},
            )

    axes.set_xlim(lo, hi)
    axes.set_xlabel(
        "Intervention effectiveness  p_eff  (fraction of flagged exits retained)", fontsize=9
    )
    axes.set_ylabel("Expected value vs. do-nothing (USD, validation set)", fontsize=9)

    # Dollar tick labels in $k for legibility.
    axes.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"${v / 1_000:,.0f}k"))

    title_bits = [bit for bit in (model_name, cohort) if bit]
    title = " — ".join(title_bits) if title_bits else "Expected-Value Sensitivity"
    if threshold is not None:
        title += f"  (threshold t={threshold:.2f})"
    axes.set_title(title, fontsize=10)

    axes.legend(fontsize=8, loc="lower right")
    axes.grid(visible=True, alpha=0.2)

    return fig


__all__ = [
    "DEFAULT_INTERVENTION_COST",
    "DEFAULT_P_EFF",
    "DEFAULT_REPLACEMENT_COST",
    "REPLACEMENT_COST_MULTIPLIER",
    "breakeven_p_eff",
    "ev_at_threshold",
    "expected_value_plot",
    "p_eff_sensitivity_sweep",
    "replacement_cost_from_salary",
]
