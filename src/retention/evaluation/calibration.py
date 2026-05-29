"""Calibration metrics + reliability diagrams — Story 3.2.

Two questions this module answers:

    1. *Are the predicted probabilities trustworthy numbers?*
       `expected_calibration_error()` (ECE) quantifies this: if the model
       says "30 % attrition risk" for a group of employees, do roughly 30 %
       of them actually leave? A low ECE means yes.

    2. *Where does the model over- or under-estimate confidence?*
       `reliability_diagram()` visualises this: the 45° diagonal is perfect
       calibration; bars above the line mean the model is under-confident
       (predicted lower than observed) and bars below mean over-confident.

**Why calibration matters here:**
The threshold and expected-value work in Stories 3.3–3.4 depends on
calibrated probabilities. If a model's predicted 0.3 actually corresponds
to a 0.5 observed rate, the EV formula gives the wrong answer. Calibration
check comes *first*, before any decision-theoretic work downstream.

**Key findings from Loop 2 (main comparison table):**
    - GBM Brier 0.158 ≈ base-rate Brier 0.162 — nearly calibrated.
    - LR  Brier 0.236 >> base-rate — class_weight='balanced' inflates proba.
    - EBM Brier 0.235 >> base-rate — compute_sample_weight same issue.
    This module operationalises that finding into an actual ECE + diagram per
    model, which the calibration notebook (Story 3.6) will present.

**ECE formula:**
    ECE = Σ_k (|B_k| / n) × |mean_confidence(B_k) - fraction_positive(B_k)|
    where B_k is the k-th equal-width probability bin [k/M, (k+1)/M).
    Source: Guo et al. (2017) "On Calibration of Modern Neural Networks."
    The formula is architecture-agnostic — applies equally to tree ensembles,
    linear models, and GAMs.

**Reliability diagram:**
    10 equal-width probability bins across [0, 1].
    x-axis: mean predicted probability per bin.
    y-axis: observed positive fraction per bin.
    45° diagonal = perfect calibration reference line.
    Bar colour encodes direction: under-confident (blue) vs over-confident (red).
    Empty bins (no examples) are silently skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE is the weighted average absolute difference between a model's
    predicted confidence and the observed positive fraction within each
    probability bin:

        ECE = Σ_k (|B_k| / n) × |mean_conf(B_k) − frac_pos(B_k)|

    A model with ECE = 0 is perfectly calibrated (predicted probabilities
    exactly match observed rates). A model with ECE = 0.10 is on average 10
    percentage points off between confidence and observation.

    **Calibration baseline:** A degenerate model that always predicts the
    base rate β achieves ECE = 0.0 (every bin's confidence equals β and the
    only non-empty bin has frac_pos = β). This means ECE alone is insufficient
    — a model can achieve low ECE by predicting the same constant for everyone.
    Always read ECE alongside AUC-PR: you want both high discrimination and
    good calibration.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool). Shape (n_samples,).
        y_proba: Predicted probabilities for the positive class (P(exit=1)).
            Shape (n_samples,) or (n_samples, 2); if 2D, column 1 is used.
        n_bins: Number of equal-width probability bins in [0, 1].
            Default 10 → bins of width 0.1.

    Returns:
        Float in [0, 1]. Lower is better calibrated.
        Returns NaN if y_true has no variation (all 0 or all 1) —
        calibration is undefined for a degenerate label set.

    Raises:
        ValueError: if n_bins < 2 or y_proba contains non-finite values.

    Teaching note — equal-width vs equal-frequency bins:
        Equal-width bins (used here): fixed probability intervals [0.1×k, 0.1×(k+1)).
        Equal-frequency bins: same number of examples per bin. Equal-frequency
        is preferred when the distribution of predicted probabilities is very
        skewed (most predictions near 0). For this project's 20 % base rate,
        equal-width bins work well — we have examples spread across the range.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")

    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]

    if not np.isfinite(proba).all():
        raise ValueError("y_proba contains non-finite values (NaN or Inf).")

    y_arr = np.asarray(y_true).astype(float)
    n = len(proba)

    # Equal-width bin edges: [0, 0.1, 0.2, ..., 1.0] for n_bins=10.
    # np.digitize assigns each prediction to a bin index 1..n_bins.
    # We use edges[:-1] to get the lower bound of each bin.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bin_edges[1:-1])  # 0-indexed bin assignments

    ece = 0.0
    for k in range(n_bins):
        # Boolean mask for examples in bin k
        mask = bin_ids == k

        if mask.sum() == 0:
            # Empty bin — contributes 0 to ECE (weight term is 0/n = 0)
            continue

        bin_confidence = float(proba[mask].mean())  # mean predicted prob in bin k
        bin_accuracy = float(y_arr[mask].mean())  # observed positive rate in bin k
        bin_weight = float(mask.sum()) / n

        # Weighted absolute calibration error for this bin
        ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return float(ece)


def reliability_diagram(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    n_bins: int = 10,
    *,
    model_name: str = "",
    cohort: str = "",
    show_ece: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot a reliability diagram for a binary classifier.

    A reliability diagram (calibration curve) plots:
      - x-axis: mean predicted probability in each probability bin
      - y-axis: observed positive fraction in each bin
      - 45° diagonal: perfect calibration reference line

    Bars above the diagonal: model is *under-confident* (predicted lower
    than the observed rate — e.g. predicted 0.2 but observed 0.4).
    Bars below the diagonal: model is *over-confident* (predicted higher
    than observed — e.g. predicted 0.7 but observed 0.3).

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2).
        n_bins: Number of equal-width probability bins. Default 10.
        model_name: Optional model label for the title (e.g. 'GBM').
        cohort: Optional cohort label for the title (e.g. 'hybrid').
        show_ece: If True, annotate the plot with the ECE value.
        ax: Optional matplotlib Axes to draw on. If None, a new Figure
            and Axes are created.

    Returns:
        matplotlib.figure.Figure object. The caller is responsible for
        saving (``fig.savefig(...)``) or displaying (``plt.show()``).
        When `ax` is provided, the Figure containing `ax` is returned.

    Teaching note — what to look for:
        Perfect calibration → all bars lie on the diagonal.
        GBM in this project: near-diagonal (Brier ≈ 0.158, close to baseline).
        LR/EBM: bars above the diagonal in the high-confidence range because
        balanced class weights inflate probabilities — the model assigns p=0.6
        to groups where only ~40 % actually exit. Visible as rightward bars
        above the 45° line.

    Teaching note — matplotlib Figure lifecycle:
        This function creates a Figure and returns it. The figure is NOT
        shown or saved automatically — the caller owns the lifecycle.
        In a notebook: ``fig = reliability_diagram(...); plt.show()``.
        In a script: ``fig = reliability_diagram(...); fig.savefig(path)``.
        Always call ``plt.close(fig)`` after saving in a loop to free memory.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]

    y_arr = np.asarray(y_true).astype(float)

    # ── Bin the predictions ────────────────────────────────────────────────
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bin_edges[1:-1])

    mean_predicted: list[float] = []  # x-axis: mean confidence per bin
    fraction_positive: list[float] = []  # y-axis: observed rate per bin
    bin_counts: list[int] = []  # for bar width annotation

    for k in range(n_bins):
        mask = bin_ids == k
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted.append(float(proba[mask].mean()))
        fraction_positive.append(float(y_arr[mask].mean()))
        bin_counts.append(count)

    # ── Plot ───────────────────────────────────────────────────────────────
    # Direct if/else so mypy can narrow `ax` from `Axes | None` to `Axes`
    # inside the else branch (stored-predicate narrowing is not supported).
    if ax is None:
        # plt.subplots() returns (Figure, Axes) for a single subplot.
        _fig, _ax = plt.subplots(figsize=(5, 5))
        fig: matplotlib.figure.Figure = _fig
        axes: matplotlib.axes.Axes = _ax
    else:
        # ax was passed in — extract its parent Figure.
        # get_figure() is typed as Figure | SubFigure | None in newer stubs;
        # we assert it's not None (the Axes must belong to a Figure).
        axes = ax
        _raw_fig = axes.get_figure()
        assert _raw_fig is not None, "ax.get_figure() returned None"
        fig = _raw_fig  # type: ignore[assignment]

    # 45° perfect-calibration reference line
    axes.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")

    # Calibration bars: colour by direction
    # Blue (under-confident): fraction_positive > mean_predicted
    # Red (over-confident):   fraction_positive < mean_predicted
    for x, y, _count in zip(mean_predicted, fraction_positive, bin_counts, strict=True):
        colour = "#4c72b0" if y >= x else "#c44e52"
        bin_width = 1.0 / n_bins
        axes.bar(
            x,
            y,
            width=bin_width * 0.8,
            color=colour,
            alpha=0.7,
            align="center",
            edgecolor="white",
            linewidth=0.5,
        )

    # ECE annotation
    if show_ece:
        ece_val = expected_calibration_error(y_true, y_proba, n_bins=n_bins)
        axes.text(
            0.05,
            0.92,
            f"ECE = {ece_val:.3f}",
            transform=axes.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    # Axis formatting
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("Mean predicted probability", fontsize=9)
    axes.set_ylabel("Observed positive fraction", fontsize=9)

    title_parts = []
    if model_name:
        title_parts.append(model_name)
    if cohort:
        title_parts.append(cohort)
    title = " — ".join(title_parts) if title_parts else "Reliability Diagram"
    axes.set_title(title, fontsize=10)

    axes.legend(fontsize=8, loc="lower right")

    return fig


__all__ = [
    "expected_calibration_error",
    "reliability_diagram",
]
