"""Story 1.4.5 — Tests for temporal_split.

The temporal split is the single most important Risk 3 mitigation. These
tests are the canaries for any future change that might re-introduce a
random shuffle or otherwise leak the future into the past.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retention.data.split import assert_no_temporal_leak, temporal_split


# --------------------------------------------------------------------- #
# Fixture                                                                #
# --------------------------------------------------------------------- #


def _synthetic_df(n: int = 100) -> pd.DataFrame:
    """Deterministic synthetic DataFrame for split tests.

    `snapshot_date` advances monotonically so we can verify ordering.
    """
    return pd.DataFrame(
        {
            "employee_id": [f"EMP_{i:05d}" for i in range(n)],
            "snapshot_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "voluntary_exit_label": [bool(i % 5 == 0) for i in range(n)],
        }
    )


# --------------------------------------------------------------------- #
# Happy paths                                                            #
# --------------------------------------------------------------------- #


def test_temporal_split_returns_three_dataframes(tmp_path: Path) -> None:
    """Basic shape: (train, val, test) tuple, all DataFrames."""
    df = _synthetic_df(100)
    train, val, test = temporal_split(
        df, save_indices=False, indices_path=tmp_path / "splits.parquet"
    )
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_temporal_split_sizes_match_input(tmp_path: Path) -> None:
    """Sum of split sizes must equal input row count."""
    df = _synthetic_df(100)
    train, val, test = temporal_split(
        df,
        val_size=0.15,
        test_size=0.15,
        save_indices=False,
        indices_path=tmp_path / "splits.parquet",
    )
    assert len(train) + len(val) + len(test) == len(df)
    # 15% of 100 = 15 by round; 100 - 15 - 15 = 70 train
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_temporal_split_no_leak_invariant(tmp_path: Path) -> None:
    """The Risk-3 canary: max(train) ≤ min(val) ≤ max(val) ≤ min(test)."""
    df = _synthetic_df(100)
    train, val, test = temporal_split(
        df, save_indices=False, indices_path=tmp_path / "splits.parquet"
    )
    # Internal assertion is already invoked by temporal_split; verify it externally too.
    assert train["snapshot_date"].max() <= val["snapshot_date"].min()
    assert val["snapshot_date"].max() <= test["snapshot_date"].min()


def test_temporal_split_assigns_oldest_to_train(tmp_path: Path) -> None:
    """Oldest rows go to train, newest to test — never the reverse."""
    df = _synthetic_df(100)
    train, val, test = temporal_split(
        df, save_indices=False, indices_path=tmp_path / "splits.parquet"
    )
    assert train["snapshot_date"].min() == df["snapshot_date"].min()
    assert test["snapshot_date"].max() == df["snapshot_date"].max()


def test_temporal_split_saves_indices_when_requested(tmp_path: Path) -> None:
    """`save_indices=True` writes a parquet with employee_id → split label."""
    df = _synthetic_df(20)
    indices_path = tmp_path / "splits.parquet"
    temporal_split(df, save_indices=True, indices_path=indices_path)
    assert indices_path.exists()

    splits = pd.read_parquet(indices_path)
    assert set(splits.columns) == {"employee_id", "split"}
    assert set(splits["split"].unique()) == {"train", "val", "test"}
    assert len(splits) == len(df)


def test_temporal_split_deterministic_across_runs(tmp_path: Path) -> None:
    """Same input → same split assignment. Risk-6 reproducibility check."""
    df = _synthetic_df(50)
    train1, val1, test1 = temporal_split(
        df, save_indices=False, indices_path=tmp_path / "1.parquet"
    )
    train2, val2, test2 = temporal_split(
        df, save_indices=False, indices_path=tmp_path / "2.parquet"
    )
    assert list(train1["employee_id"]) == list(train2["employee_id"])
    assert list(val1["employee_id"]) == list(val2["employee_id"])
    assert list(test1["employee_id"]) == list(test2["employee_id"])


# --------------------------------------------------------------------- #
# Edge cases                                                             #
# --------------------------------------------------------------------- #


def test_temporal_split_raises_on_oversized_val_plus_test(tmp_path: Path) -> None:
    """val_size + test_size ≥ 1.0 leaves no rows for training."""
    df = _synthetic_df(10)
    with pytest.raises(ValueError, match="must leave a positive training fraction"):
        temporal_split(
            df,
            val_size=0.5,
            test_size=0.6,
            save_indices=False,
            indices_path=tmp_path / "x.parquet",
        )


def test_temporal_split_raises_on_missing_snapshot_col(tmp_path: Path) -> None:
    """Clear error when the temporal anchor column is missing."""
    df = pd.DataFrame({"employee_id": ["a", "b"], "x": [1, 2]})
    with pytest.raises(ValueError, match="snapshot column"):
        temporal_split(df, save_indices=False, indices_path=tmp_path / "x.parquet")


def test_temporal_split_raises_on_missing_tiebreak_col(tmp_path: Path) -> None:
    """Clear error when the tie-break column is missing."""
    df = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2025-01-01", periods=3, freq="D"),
            "voluntary_exit_label": [True, False, True],
        }
    )
    with pytest.raises(ValueError, match="tiebreak column"):
        temporal_split(df, save_indices=False, indices_path=tmp_path / "x.parquet")


# --------------------------------------------------------------------- #
# assert_no_temporal_leak directly                                       #
# --------------------------------------------------------------------- #


def test_assert_no_temporal_leak_catches_train_after_val() -> None:
    """If train contains rows newer than val, the assertion must fail."""
    train = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-12-31"])})
    val = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-01-01"])})
    test = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-06-01"])})
    with pytest.raises(AssertionError, match="Temporal leak"):
        assert_no_temporal_leak(train, val, test, snapshot_col="snapshot_date")


def test_assert_no_temporal_leak_catches_val_after_test() -> None:
    train = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-01-01"])})
    val = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-12-31"])})
    test = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-06-01"])})
    with pytest.raises(AssertionError, match="Temporal leak"):
        assert_no_temporal_leak(train, val, test, snapshot_col="snapshot_date")


def test_assert_no_temporal_leak_passes_when_ordered() -> None:
    train = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-01-01", "2025-03-01"])})
    val = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-04-01", "2025-06-01"])})
    test = pd.DataFrame({"snapshot_date": pd.to_datetime(["2025-07-01", "2025-12-31"])})
    # Should not raise.
    assert_no_temporal_leak(train, val, test, snapshot_col="snapshot_date")


def test_assert_no_temporal_leak_empty_splits_pass() -> None:
    """Vacuous safety for tiny inputs — empty splits skip the check."""
    train = pd.DataFrame({"snapshot_date": pd.to_datetime([])})
    val = pd.DataFrame({"snapshot_date": pd.to_datetime([])})
    test = pd.DataFrame({"snapshot_date": pd.to_datetime([])})
    # Should not raise.
    assert_no_temporal_leak(train, val, test, snapshot_col="snapshot_date")
