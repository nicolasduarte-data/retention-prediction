"""Evaluation metrics — Story 3.1.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import average_precision_score

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
        ValueError: if y_proba contains NaNs or is the wrong shape.
    """
    import numpy as np  # noqa: PLC0415

    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        if proba.shape[1] != 2:
            raise ValueError(
                f"y_proba with 2D shape must have exactly 2 columns; got shape {proba.shape}."
            )
        proba = proba[:, 1]

    if np.isnan(proba).any():
        raise ValueError("y_proba contains NaN values. Check model output.")

    return float(average_precision_score(y_true, proba))


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
