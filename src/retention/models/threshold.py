"""Threshold calibration as a controlled experiment — Story 3.3.

This module answers a question that calibration (Story 3.2) does not:

    Calibration asks *are the probabilities trustworthy numbers?*
    Thresholding asks *given trustworthy probabilities, where do we draw the
    line between "flag for a retention conversation" and "leave alone"?*

A classifier outputs a probability per employee. To *act* on it, HR needs a
binary decision — flag or don't. That requires an operating threshold. The
default 0.5 is almost never the right one under class imbalance: at a 20 %
base rate, demanding p > 0.5 to flag someone means flagging almost no one, so
real exits slip through. The right threshold is an empirical choice, and this
module makes that choice the way a careful analyst would — by experiment.

────────────────────────────────────────────────────────────────────────────
FRAMING 1 — Threshold selection IS a controlled experiment (the A/B framing)
────────────────────────────────────────────────────────────────────────────
Sweeping thresholds is not hyperparameter tuning and it is not retraining. The
model's *ranking* is frozen — every employee's predicted probability is fixed.
All we vary is the cut-point. So each candidate threshold is a **treatment
arm**: a different decision policy applied to the *same* scored population.

    arm_t:  "flag everyone with p ≥ t"

We evaluate every arm on the *same held-out set*, score each on a *pre-declared*
metric (F1 by default; F2 when missed-exits dominate the cost), and adopt the
empirically-winning arm. That is precisely the discipline of an A/B test:
pre-register the metric, compare arms on common held-out data, pick the winner,
confirm on a fresh sample. The honest difference from a classic A/B test is
that we compare operating points of *one* deployed system rather than two
competing systems — a *within-system* operating-point experiment. Naming it
correctly is the Senior-Product-Analytics signal; the work was always an
experiment, most practitioners just never label it one.

────────────────────────────────────────────────────────────────────────────
FRAMING 2 — Why a threshold, not SMOTE (the R1 [FLIP-RISK] rationale)
────────────────────────────────────────────────────────────────────────────
The textbook reflex for imbalance is to resample (SMOTE). We deliberately do
not, for two reasons that compound:

  1. SMOTE interpolates between training rows to synthesise minority examples.
     Those synthetic rows have no place in time — they straddle the temporal
     split boundary in feature space, quietly breaking the very ordering
     guarantee `temporal_split()` exists to protect.
  2. SMOTE distorts probability calibration. Epic 3's entire downstream chain —
     calibration (3.2), threshold (this story), expected value (3.4) — depends
     on probabilities that mean what they say. Inflating minority density
     breaks the "predicted 0.30 ≈ observed 0.30" property we just verified.

Threshold calibration reaches the same operational goal (catch more exits)
without touching the data distribution at all. We keep the honest, calibrated
probabilities and simply move the decision line. The probability model stays
correct; only the operating point changes.

The **[FLIP-RISK]** is the asymmetric cost: a *missed exit* (false negative — a
regretted-attrition employee we failed to flag) is far more expensive than a
*wasted conversation* (false positive — a retention chat with someone who would
have stayed). The F2 criterion encodes that asymmetry: it weights recall twice
as heavily as precision, so the F2-optimal threshold sits *below* the F1-optimal
one — flagging more people, accepting more false positives, to miss fewer real
exits. This module surfaces both operating points so the cost trade-off is an
explicit decision, not a hidden default.

────────────────────────────────────────────────────────────────────────────
LEAKAGE DISCIPLINE — sweep on validation, confirm on test (once)
────────────────────────────────────────────────────────────────────────────
Choosing an operating point is a *model-selection* decision, so it must use the
**validation** split — never the test split. Optimising the threshold on test
and then reporting test metrics at that threshold is leakage: you tuned on the
data you report on. The clean protocol, which this project follows:

    train  → fit the probability model
    val    → compare threshold arms, pick the winning operating point  (here)
    test   → confirm the chosen arm's performance, ONCE, at champion
             selection (Story 3.6); untouched until then

So the figures and tables this module produces are **validation-set** numbers.
The test set stays pristine.

────────────────────────────────────────────────────────────────────────────
API
────────────────────────────────────────────────────────────────────────────
    threshold_sweep(...)       → DataFrame, one row per arm (the experiment log)
    optimize_threshold(...)    → float, the empirically-winning threshold
    threshold_sweep_plot(...)  → matplotlib Figure of the arm comparison

Story 3.3 — see `backlog/epic-3-evaluation-rigor.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# The criteria `optimize_threshold` knows how to maximise. 'ev' is intentionally
# absent: expected-value optimisation needs cost parameters (replacement cost,
# intervention cost, p_eff) that arrive in Story 3.4 — see expected_value.py.
_VALID_CRITERIA: tuple[str, ...] = ("f1", "f2", "youden")


def _confusion_counts(
    y_true: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for two aligned binary 0/1 arrays.

    Kept as a tiny named helper rather than inlined because the four counts are
    the atoms every metric below is built from — naming them makes the precision
    and recall formulas read like their textbook definitions.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is 0 instead of NaN.

    Why 0.0 and not NaN: at extreme thresholds a confusion-matrix cell empties
    out (e.g. no predicted positives → precision is 0/0). The sklearn convention
    for that degenerate case is 0.0, and a 0.0 keeps `argmax` well-defined — a
    NaN would silently poison the threshold search.
    """
    return float(numerator / denominator) if denominator > 0 else 0.0


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """F-beta score: the harmonic mean of precision and recall, recall-weighted.

        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

    beta = 1 → F1, the symmetric balance (precision and recall weigh equally).
    beta = 2 → F2, recall counts 2x as much as precision — the [FLIP-RISK]
               criterion, because a missed exit costs more than a wasted flag.

    The recall-weighting is visible algebraically: as beta grows, the `beta^2 * P`
    term dominates the denominator, so the score becomes dominated by R. Concretely
    P=1.0, R=0.5 gives F1=0.667 but F2=0.556 (penalised for low recall), whereas
    P=0.5, R=1.0 gives F1=0.667 but F2=0.833 (rewarded for high recall).
    """
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return _safe_divide((1.0 + beta_sq) * precision * recall, denominator)


def threshold_sweep(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    *,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep decision thresholds and tabulate the confusion-derived metrics.

    Each row of the returned table is one **treatment arm** — the decision policy
    "flag everyone with predicted probability ≥ threshold" — scored on the data
    you pass in. The table is the experiment log; `optimize_threshold` reads it to
    pick a winner, and `threshold_sweep_plot` draws it.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool). Shape (n_samples,).
        y_proba: Predicted probabilities for the positive class (P(exit=1)).
            Shape (n_samples,) or (n_samples, 2); if 2D, column 1 is used.
        start: Lowest threshold in the grid. Default 0.05.
        stop: Highest threshold in the grid (inclusive). Default 0.95.
        step: Grid spacing. Default 0.01 → 91 arms across [0.05, 0.95].

    Returns:
        A DataFrame with one row per threshold and columns:
            threshold, tp, fp, fn, tn,
            precision, recall, specificity, f1, f2, youden_j
        Sorted ascending by threshold.

    Raises:
        ValueError: if y_proba contains non-finite values, or if y_true has no
            positive examples (recall and the F-scores would be undefined for
            every arm — threshold optimisation is meaningless without a positive).

    Teaching note — the decision rule is `>=`, not `>`:
        An employee is flagged when their probability is *at or above* the
        threshold. This is the common convention and it makes the lowest grid
        threshold behave intuitively: at t = 0.05, everyone with p ≥ 0.05 is
        flagged, so recall → 1.0 and precision → the base rate.
    """
    import pandas as pd  # noqa: PLC0415

    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]

    if not np.isfinite(proba).all():
        raise ValueError("y_proba contains non-finite values (NaN or Inf).")

    y_arr = np.asarray(y_true).astype(int)
    if y_arr.sum() == 0:
        raise ValueError(
            "y_true has no positive examples; recall and F-scores are undefined "
            "for every threshold. Threshold optimisation requires ≥ 1 positive."
        )

    # Build the grid of candidate thresholds — one treatment arm per value.
    # We derive it from a rounded integer count rather than np.arange(start, stop,
    # step) because np.arange accumulates floating-point error and can drop or add
    # a final element (the classic 0.95-becomes-0.9499999 bug). round() pins each
    # threshold to a clean grid value so downstream equality checks behave.
    n_steps = int(round((stop - start) / step)) + 1
    thresholds = np.round(start + step * np.arange(n_steps), 10)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        # The whole experiment in one line: re-draw the decision boundary, keep
        # the (frozen) probabilities. Only `y_pred` changes between arms.
        y_pred = (proba >= threshold).astype(int)
        tp, fp, fn, tn = _confusion_counts(y_arr, y_pred)

        precision = _safe_divide(tp, tp + fp)  # of those we flagged, how many truly exit
        recall = _safe_divide(tp, tp + fn)  # of all true exits, how many we caught
        specificity = _safe_divide(tn, tn + fp)  # of all stayers, how many we left alone

        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": _fbeta(precision, recall, beta=1.0),
                "f2": _fbeta(precision, recall, beta=2.0),
                # Youden's J = sensitivity + specificity - 1. A prevalence-independent
                # criterion (Youden 1950): it maximises the vertical distance from the
                # ROC chance line, ignoring how rare exits are. Reported as a third
                # arm-ranking lens alongside the F-scores.
                "youden_j": recall + specificity - 1.0,
            }
        )

    return pd.DataFrame(rows)


def optimize_threshold(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    criterion: str = "f1",
    *,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> float:
    """Return the threshold that maximises `criterion` over the sweep grid.

    This is the "pick the empirically-winning arm" step. It runs the full
    `threshold_sweep`, then returns the threshold of the arm with the best score
    on the chosen criterion.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class. 1D or (n, 2).
        criterion: Which arm-ranking metric to maximise.
            'f1'     — symmetric balance of precision and recall (default).
            'f2'     — recall-weighted; the [FLIP-RISK] choice (missed exits hurt).
            'youden' — Youden's J (sensitivity + specificity − 1), prevalence-free.
        start, stop, step: Grid definition, passed through to `threshold_sweep`.

    Returns:
        The winning threshold as a float on the grid, in [start, stop].

    Raises:
        ValueError: if `criterion` is not a recognised name, or for the input
            problems `threshold_sweep` validates (non-finite proba, no positives).
        NotImplementedError: if `criterion='ev'` — expected-value optimisation
            needs cost parameters that arrive in Story 3.4 (expected_value.py).

    Teaching note — how ties are broken:
        When several adjacent thresholds achieve the same best score (a plateau —
        common with discrete confusion counts), `np.argmax` returns the *first*
        index. Because the grid is ascending, that is the *lowest* threshold on
        the plateau — the one that flags the most people, i.e. the highest-recall
        end. Under [FLIP-RISK] that is exactly the safe tie-break: when two
        operating points are otherwise equal, prefer the one that misses fewer
        real exits.
    """
    if criterion == "ev":
        raise NotImplementedError(
            "criterion='ev' is implemented in Story 3.4 (expected_value.py); "
            "it requires replacement_cost, intervention_cost, and p_eff parameters."
        )
    if criterion not in _VALID_CRITERIA:
        raise ValueError(f"criterion must be one of {_VALID_CRITERIA}, got {criterion!r}.")

    sweep = threshold_sweep(y_true, y_proba, start=start, stop=stop, step=step)

    # Map the criterion name to the column it maximises. 'youden' reads the
    # youden_j column; 'f1'/'f2' read their own.
    score_column = "youden_j" if criterion == "youden" else criterion
    scores = sweep[score_column].to_numpy()

    best_idx = int(np.argmax(scores))  # first-occurrence → lowest-threshold tie-break
    return float(sweep["threshold"].iloc[best_idx])


def threshold_sweep_plot(
    sweep: pd.DataFrame,
    *,
    optimal_threshold: float | None = None,
    criterion: str = "f1",
    model_name: str = "",
    cohort: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot precision, recall, and F1 across the threshold sweep.

    This is the "treatment arm comparison" chart: the x-axis is the threshold
    (one arm per value), and the three curves show how precision, recall, and F1
    trade off as the decision line moves. The dashed vertical line marks the
    empirically-winning arm returned by `optimize_threshold`.

    Args:
        sweep: The DataFrame returned by `threshold_sweep`.
        optimal_threshold: If given, drawn as a dashed vertical line and labelled
            as the chosen arm. Pass the output of `optimize_threshold`.
        criterion: Name of the criterion the optimal threshold maximised — used
            only to label the chosen-arm line (e.g. "F1-optimal arm").
        model_name: Optional model label for the title (e.g. 'GBM').
        cohort: Optional cohort label for the title (e.g. 'hybrid').
        ax: Optional matplotlib Axes to draw on. If None, a new Figure and Axes
            are created.

    Returns:
        The matplotlib Figure. The caller owns the lifecycle — save with
        ``fig.savefig(...)`` and free with ``plt.close(fig)`` when looping.

    Teaching note — what the crossing point tells you:
        Precision rises and recall falls as the threshold climbs (flag fewer
        people → cleaner flags but more misses). They cross somewhere in the
        middle; F1 peaks near that crossing because the harmonic mean is dragged
        down by whichever of the two is smaller. The F2 curve (not drawn, to keep
        the chart legible) peaks to the *left* of F1 — that leftward shift is the
        [FLIP-RISK] recall preference made visible.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # Direct if/else (not a stored predicate) so mypy narrows `ax` from
    # `Axes | None` to `Axes` inside the else branch — see calibration.py for
    # the same pattern and the reasoning behind it.
    if ax is None:
        _fig, _ax = plt.subplots(figsize=(7, 5))
        fig: matplotlib.figure.Figure = _fig
        axes: matplotlib.axes.Axes = _ax
    else:
        axes = ax
        _raw_fig = axes.get_figure()
        assert _raw_fig is not None, "ax.get_figure() returned None"
        fig = _raw_fig  # type: ignore[assignment]

    thresholds = sweep["threshold"].to_numpy()
    axes.plot(
        thresholds, sweep["precision"].to_numpy(), color="#c44e52", linewidth=1.6, label="Precision"
    )
    axes.plot(
        thresholds, sweep["recall"].to_numpy(), color="#4c72b0", linewidth=1.6, label="Recall"
    )
    axes.plot(thresholds, sweep["f1"].to_numpy(), color="#55a868", linewidth=2.4, label="F1")

    if optimal_threshold is not None:
        label = f"{criterion.upper()}-optimal arm (t={optimal_threshold:.2f})"
        axes.axvline(optimal_threshold, color="black", linestyle="--", linewidth=1.2, label=label)

    axes.set_xlim(float(thresholds.min()), float(thresholds.max()))
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Decision threshold  —  one treatment arm per value", fontsize=9)
    axes.set_ylabel("Score (validation set)", fontsize=9)

    title_parts = [part for part in (model_name, cohort) if part]
    title = " — ".join(title_parts) if title_parts else "Threshold Sweep"
    axes.set_title(title, fontsize=10)

    axes.legend(fontsize=8, loc="best")
    axes.grid(visible=True, alpha=0.2)

    return fig


__all__ = [
    "optimize_threshold",
    "threshold_sweep",
    "threshold_sweep_plot",
]
