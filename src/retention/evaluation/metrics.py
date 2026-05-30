"""Evaluation metrics — Story 3.1 (extended in Story 2.5 for comparison table).

AUC-PR (Area Under the Precision-Recall Curve) is the primary metric for
the retention model throughout all loops. Why not AUC-ROC?

- Retention datasets are class-imbalanced (voluntary exits are ~15-25% of
  headcount). AUC-ROC is optimistic under imbalance because true-negatives
  inflate the denominator.
- AUC-PR scores the model only on how well it recovers positive examples,
  which is exactly the business question: "can you rank the people most
  likely to leave so HR can intervene?"
- The Rung 1 caption is a deliberate epistemic flag — the model is
  associational, not causal. It tells an HR analyst what the prediction IS
  and what it ISN'T, keeping the model card honest from day one.

**Story 2.5 additions:** `auc_roc`, `precision_at_k`, `recall_at_k`, `brier_score`
are added here to power the 6-cell comparison table (LR/GBM/EBM × hris_only/hybrid).
These will be further wrapped by calibration, threshold, and EV logic in Epic 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from retention.evaluation._validation import extract_positive_proba

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def auc_pr(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Compute the Area Under the Precision-Recall Curve.

    Uses sklearn's `average_precision_score`, which computes the weighted
    mean of precisions at each threshold, with the increase in recall from
    the previous threshold as the weight. This equals the area under the
    step-interpolated PR curve.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class (P(exit=1)).
            Shape (n_samples,) or (n_samples, 2) — if 2-column, column 1
            is used automatically.

    Returns:
        Float in [0, 1]. A no-skill classifier (constant P(y=1)) scores
        equal to the class prevalence. 1.0 = perfect ranking.

    Raises:
        ValueError: if y_proba is non-finite (NaN/Inf), empty, or a 2-D array
            whose width is not exactly 2 (see ``_validation``).
    """
    proba = extract_positive_proba(y_proba)
    return float(average_precision_score(y_true, proba))


def auc_roc(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Compute AUC-ROC (Area Under the Receiver Operating Characteristic Curve).

    Reported alongside AUC-PR for completeness — but AUC-PR is the primary
    metric because AUC-ROC is optimistic under class imbalance. A model that
    predicts the majority class well can still achieve a high AUC-ROC even if
    it rarely identifies exits. AUC-PR forces the model to rank exits well.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2) — col 1 if 2D.

    Returns:
        Float in [0.5, 1.0] for a useful model. 0.5 = random, 1.0 = perfect.

    Raises:
        ValueError: if y_proba is non-finite (NaN/Inf), empty, or a 2-D array
            whose width is not exactly 2 (see ``_validation``).
    """
    proba = extract_positive_proba(y_proba)
    return float(roc_auc_score(y_true, proba))


def precision_at_k(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    k: float = 0.10,
) -> float:
    """Precision in the top-k fraction of employees ranked by predicted exit risk.

    The operational HR question: "If we flag the top k% highest-risk employees
    for a retention conversation, what fraction of those are genuine exit risks?"
    This is the metric HR managers care about — it controls wasted interventions.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2).
        k: Fraction of the population to flag (0.10 = top 10%).

    Returns:
        Precision in the top-k bucket. 0.0 = all flagged employees are false
        positives; 1.0 = all flagged employees actually exit.

    Raises:
        ValueError: if k is not in (0, 1], or if y_proba is non-finite (NaN/Inf),
            empty, or a 2-D array whose width is not exactly 2 (see ``_validation``).
    """
    import numpy as np  # noqa: PLC0415

    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")

    proba = extract_positive_proba(y_proba)
    y_arr = np.asarray(y_true)
    n = len(proba)
    n_top = max(1, int(np.ceil(n * k)))

    # Descending sort by predicted probability, take top n_top
    top_indices = np.argsort(proba)[::-1][:n_top]
    return float(y_arr[top_indices].mean())


def recall_at_k(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    k: float = 0.10,
) -> float:
    """Recall in the top-k fraction of employees ranked by predicted exit risk.

    The operational HR question: "Of all employees who will actually exit, what
    fraction are captured in our top-k flag list?" This controls missed exits
    (the FLIP-RISK: intervening with no-one who would have left).

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2).
        k: Fraction of the population to flag (0.10 = top 10%).

    Returns:
        Recall in the top-k bucket. 0.0 = no real exits flagged; 1.0 = all
        real exits captured in the top-k slice. Returns 0.0 if no positives
        in y_true (degenerate label set).

    Raises:
        ValueError: if k is not in (0, 1], or if y_proba is non-finite (NaN/Inf),
            empty, or a 2-D array whose width is not exactly 2 (see ``_validation``).
    """
    import numpy as np  # noqa: PLC0415

    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")

    proba = extract_positive_proba(y_proba)
    y_arr = np.asarray(y_true)
    total_positives = y_arr.sum()
    if total_positives == 0:
        return 0.0

    n = len(proba)
    n_top = max(1, int(np.ceil(n * k)))
    top_indices = np.argsort(proba)[::-1][:n_top]
    positives_captured = y_arr[top_indices].sum()
    return float(positives_captured / total_positives)


def brier_score(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Compute the Brier score (mean squared error of predicted probabilities).

    A calibration-aware metric that penalises confident wrong predictions more
    than uncertain wrong predictions. Lower is better:
      - Perfect calibration + prediction: 0.0
      - Random (P(exit) = base rate): ~base_rate * (1 - base_rate)
      - Constant P=0: base_rate (predicts no one exits)
      - Constant P=1: 1 - base_rate (predicts everyone exits)

    At 18.8% base rate, random baseline Brier ≈ 0.153. A model with Brier
    above 0.153 is worse than predicting the base rate for everyone.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2).

    Returns:
        Float in [0, 1]. Lower = better calibration + discrimination.

    Raises:
        ValueError: if y_proba is non-finite (NaN/Inf), empty, or a 2-D array
            whose width is not exactly 2 (see ``_validation``).
    """
    proba = extract_positive_proba(y_proba)
    return float(brier_score_loss(y_true, proba))


def lift_at_k(
    y_true: pd.Series | np.ndarray,  # type: ignore[type-arg]
    y_proba: np.ndarray,  # type: ignore[type-arg]
    k: float = 0.10,
) -> float:
    """Lift in the top-k fraction of employees ranked by predicted exit risk.

    Lift = precision_at_k / base_rate.

    The operational translation: "If the model flags the top k% of employees,
    how many times more exits does that group contain than a random sample of
    the same size?" A lift of 2.0 means the model is 2x better than random
    at surfacing real exits in the top-k slice.

    Lift is the dimensionless complement to precision@k — it normalises out the
    base rate, making models comparable across datasets with different prevalence.
    A model with precision@10% = 0.35 on a 20% base rate (lift = 1.75) is doing
    less well than one with precision@10% = 0.25 on a 10% base rate (lift = 2.5).

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_proba: Predicted probabilities for the positive class.
            Shape (n_samples,) or (n_samples, 2).
        k: Fraction of the population to flag (0.10 = top 10 %).

    Returns:
        Lift in the top-k bucket. 1.0 = no better than random; > 1.0 = model
        identifies exits more efficiently than random. Returns 0.0 if
        base_rate is 0 (degenerate label set with no positives).

    Raises:
        ValueError: if k is not in (0, 1].
    """
    import numpy as np  # noqa: PLC0415

    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")

    base_rate = float(np.asarray(y_true).mean())
    if base_rate == 0.0:
        return 0.0

    # Lift is the ratio of precision@k to the base rate.
    # precision_at_k handles the top-k slicing and binary averaging.
    prec = precision_at_k(y_true, y_proba, k=k)
    return prec / base_rate


def format_rung1_caption(value: float) -> str:
    """Return a Rung 1 captioned string for use in notebooks and model cards.

    The Rung 1 tag (Pearl's Ladder of Causation) is a permanent reminder
    that this model measures association, not causation. It prevents anyone
    reading the output from confusing a high AUC-PR with a proof that
    any listed feature *causes* attrition.

    Args:
        value: AUC-PR score, typically from `auc_pr()`.

    Returns:
        Human-readable string, e.g.:
        "AUC-PR = 0.712 — associational, not causal (Rung 1)"
    """
    return f"AUC-PR = {value:.3f} — associational, not causal (Rung 1)"
