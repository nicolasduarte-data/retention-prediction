"""Shared input validation for the evaluation package — feast T2-1 / Story 3.7.

Every metric, calibration, and expected-value function in this package accepts
predicted probabilities in one of two shapes — a 1-D positive-class vector or
the sklearn ``(n, 2)`` ``predict_proba`` matrix — and every one of them must
reject the same degenerate inputs: a wrong 2-D width, non-finite values, and an
empty array.

Before this module those checks were duplicated inconsistently across
``metrics.py``, ``calibration.py``, and ``expected_value.py``. Most dangerously,
the top-k metrics did **not** reject NaN: ``np.argsort`` sorts NaN to the end, so
the ``[::-1]`` descending reversal placed a NaN-scored employee at the **top** of
the risk ranking — a silent wrong answer (feast finding T2-1). ``calibration.py``
and ``expected_value.py`` already used ``np.isfinite``; ``metrics.py`` did not.

Centralising the contract here gives one definition, one test surface, and one
guarantee: any probability that reaches a computation is finite, correctly
shaped, and non-empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def extract_positive_proba(
    y_proba: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Validate and reduce predicted probabilities to a 1-D positive-class vector.

    Accepts a 1-D probability vector or a 2-column ``predict_proba`` output
    (column 1 is the positive class). Rejects, with a clear ``ValueError``:

    - a 2-D array whose width is not exactly 2;
    - any non-finite value (NaN or Inf) — these silently corrupt rankings
      (top-k) and dollar figures (expected value) downstream;
    - an empty array — every metric in this package is undefined on zero samples.

    Args:
        y_proba: Predicted probabilities. Shape ``(n,)`` or ``(n, 2)``.

    Returns:
        A 1-D array of positive-class probabilities.

    Raises:
        ValueError: on a wrong 2-D width, a non-finite value, or an empty array.
    """
    import numpy as np  # noqa: PLC0415 — lazy import keeps the package import light

    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        if proba.shape[1] != 2:
            raise ValueError(f"2D y_proba must have exactly 2 columns; got shape {proba.shape}.")
        proba = proba[:, 1]
    if proba.size == 0:
        raise ValueError("y_proba is empty; metrics are undefined on zero samples.")
    if not np.isfinite(proba).all():
        raise ValueError("y_proba contains non-finite values (NaN or Inf).")
    return proba
