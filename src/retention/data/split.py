"""Temporal train/val/test split with no-future-leak enforcement.

Story 1.4 — Risk 3 mitigation. HR attrition data exhibits seasonality and
secular trends (org growth, market shifts) that random shuffling silently
leaks across train/test. The canonical failure mode: random shuffle →
test set contains rows from BEFORE train rows → model "predicts" past
events from future ones → AUC looks great → deploys to noise.

This module enforces the senior-craft pattern: **sort by temporal anchor,
slice oldest 70% → train, next 15% → val, newest 15% → test.** No random
shuffling. Ever. `assert_no_temporal_leak()` is the gate that fails CI if
that invariant breaks.

Split indices are saved to `data/processed/splits.parquet` so:
- A subsequent run uses the SAME train/val/test rows (cross-run reproducibility).
- Loop 2's nested CV (Story 3.5) can reference the splits.
- Loop 4's prediction writeback knows which test-set rows to score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from retention import config
from retention.features.catalog import get_temporal_anchor

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    *,
    snapshot_col: str | None = None,
    tiebreak_col: str = "employee_id",
    val_size: float = 0.15,
    test_size: float = 0.15,
    save_indices: bool = True,
    indices_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train, val, test) by ascending `snapshot_col`.

    Args:
        df: Source DataFrame. Must contain `snapshot_col` and `tiebreak_col`.
        snapshot_col: Temporal anchor column. Defaults to the
            FEATURE_CATALOG `temporal_anchor` (currently `snapshot_date`).
        tiebreak_col: Secondary sort key for deterministic ordering when many
            rows share the same `snapshot_col` value. Defaults to `employee_id`.
        val_size: Fraction of rows for the validation set. Default 0.15.
        test_size: Fraction of rows for the test set. Default 0.15.
        save_indices: If True (default), persist the split assignment to
            `data/processed/splits.parquet` (employee_id → split label).
        indices_path: Override the persistence path.

    Returns:
        (train_df, val_df, test_df) tuple. Each DataFrame retains the full
        column set of the input; only the row subsets differ.

    Raises:
        ValueError: val_size + test_size >= 1.0, or required columns missing.
        AssertionError: invariant broken (caught by `assert_no_temporal_leak`).
    """
    import pandas as pd  # noqa: PLC0415 — lazy import to keep retention.config cheap

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must leave a "
            f"positive training fraction; got train_size = "
            f"{1.0 - val_size - test_size}"
        )

    anchor = snapshot_col or get_temporal_anchor()
    if anchor not in df.columns:
        raise ValueError(
            f"snapshot column '{anchor}' not in DataFrame columns "
            f"({list(df.columns)}). The FEATURE_CATALOG temporal_anchor "
            f"must be present in the loaded mart."
        )
    if tiebreak_col not in df.columns:
        raise ValueError(
            f"tiebreak column '{tiebreak_col}' not in DataFrame columns. "
            f"Either pass a different `tiebreak_col` or ensure the loader "
            f"includes it."
        )

    # Deterministic sort: temporal first, then identifier for tie-break.
    df_sorted = df.sort_values(by=[anchor, tiebreak_col], kind="stable").reset_index(drop=True)

    n = len(df_sorted)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"After computing val ({n_val}) + test ({n_test}) sizes, no rows "
            f"remain for training (n={n}). Lower val_size/test_size or get more data."
        )

    train = df_sorted.iloc[:n_train].copy()
    val = df_sorted.iloc[n_train : n_train + n_val].copy()
    test = df_sorted.iloc[n_train + n_val :].copy()

    logger.info(
        "temporal_split: n=%d → train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%); "
        "anchor=%s [%s → %s]",
        n,
        len(train),
        100 * len(train) / n,
        len(val),
        100 * len(val) / n,
        len(test),
        100 * len(test) / n,
        anchor,
        df_sorted[anchor].iloc[0],
        df_sorted[anchor].iloc[-1],
    )

    # Hard invariant — no row in train may be temporally after any row in val/test, etc.
    assert_no_temporal_leak(train, val, test, snapshot_col=anchor)

    if save_indices:
        path = indices_path or (config.DATA_DIR / "processed" / "splits.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        split_labels = pd.DataFrame(
            {
                tiebreak_col: pd.concat(
                    [train[tiebreak_col], val[tiebreak_col], test[tiebreak_col]],
                    ignore_index=True,
                ),
                "split": (["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)),
            }
        )
        split_labels.to_parquet(path, index=False)
        logger.info("temporal_split: indices written to %s", path)

    return train, val, test


def assert_no_temporal_leak(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    snapshot_col: str = "snapshot_date",
) -> None:
    """Story 1.4.5 — Risk 3 invariant.

    All train timestamps must be ≤ all val timestamps, and all val timestamps
    must be ≤ all test timestamps. Any violation means the future has leaked
    into the past and the model's test-set AUC is fiction.

    Raises:
        AssertionError: with a descriptive message if any split is out of order.
    """
    if train.empty or val.empty or test.empty:
        return  # vacuously safe — edge case for tiny inputs

    train_max = train[snapshot_col].max()
    val_min = val[snapshot_col].min()
    val_max = val[snapshot_col].max()
    test_min = test[snapshot_col].min()

    if train_max > val_min:
        raise AssertionError(
            f"Temporal leak: max train {snapshot_col}={train_max!r} > "
            f"min val {snapshot_col}={val_min!r}. "
            f"Train must precede val chronologically."
        )
    if val_max > test_min:
        raise AssertionError(
            f"Temporal leak: max val {snapshot_col}={val_max!r} > "
            f"min test {snapshot_col}={test_min!r}. "
            f"Val must precede test chronologically."
        )
