"""Story 1.1.6 + 1.1.8 — Tests for the BigQuery loader + contract enforcement.

Five layers of coverage:
1. Contract parser unit tests — `load_contract()` parses the markdown table.
2. DataFrame schema assertion — happy path + drift detection with diff details.
3. CSV fallback path — `load_attrition_features_local()` works without BQ.
4. Live BigQuery contract test (Story 1.1.8) — skipif no SA key. The Risk-2
   canary that fails CI if pa-warehouse mart schema diverges from the contract.
5. Loader dry-run — exercises the cost-estimation path without scanning data
   (also skipif no SA key).

The BQ-dependent tests use `pytest.mark.skipif` so CI without credentials
still runs cleanly while a local developer with creds gets full coverage.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from retention import config
from retention.data import (
    SchemaContractError,
    assert_df_columns_match_contract,
    load_attrition_features_local,
    load_contract,
)


# --------------------------------------------------------------------- #
# Skip helper for BQ-credential-dependent tests                          #
# --------------------------------------------------------------------- #


def _sa_key_path() -> Path:
    """Return the resolved SA key path using the same precedence as the loader."""
    env_value = os.getenv("PA_WAREHOUSE_SA_KEY")
    if env_value:
        return Path(env_value)
    return Path.home() / ".gcp" / "pa-warehouse-sa.json"


_HAS_BQ_CREDS = _sa_key_path().exists()
_SKIP_NO_CREDS = pytest.mark.skipif(
    not _HAS_BQ_CREDS,
    reason="No BigQuery service-account key — skipping live-contract test.",
)


# --------------------------------------------------------------------- #
# Layer 1 — Contract parser                                              #
# --------------------------------------------------------------------- #


def test_load_contract_parses_v_attrition_features() -> None:
    """`load_contract()` must extract the canonical 10 columns from the live
    integration contract file."""
    contract = load_contract()

    expected_cols = {
        "employee_id",
        "snapshot_date",
        "tenure_months",
        "performance_tier",
        "compa_ratio",
        "is_critical_role",
        "successor_count",
        "age_at_window_close",
        "gender",
        "voluntary_exit_label",
    }
    assert set(contract.keys()) == expected_cols, (
        f"Contract parsed unexpected columns: got {set(contract.keys())}, "
        f"expected {expected_cols}. Check docs/integration_contract.md format."
    )
    # Spot-check structure of a representative row.
    assert contract["employee_id"]["data_type"] == "STRING"
    assert contract["voluntary_exit_label"]["data_type"] == "BOOL"
    assert all(v["is_nullable"] in {"YES", "NO"} for v in contract.values())


def test_load_contract_missing_file_raises(tmp_path: Path) -> None:
    """`load_contract()` raises FileNotFoundError when the contract file is missing."""
    bogus = tmp_path / "does-not-exist.md"
    with pytest.raises(FileNotFoundError, match="Integration contract not found"):
        load_contract(bogus)


# --------------------------------------------------------------------- #
# Layer 2 — DataFrame schema assertion                                   #
# --------------------------------------------------------------------- #


def _make_compliant_df() -> pd.DataFrame:
    """Build a DataFrame with the canonical 10 columns for happy-path tests."""
    contract = load_contract()
    return pd.DataFrame({col: [] for col in contract})


def test_assert_df_columns_match_contract_happy_path() -> None:
    """A DataFrame with exactly the contract columns passes silently."""
    df = _make_compliant_df()
    # Should not raise.
    assert_df_columns_match_contract(df)


def test_assert_df_columns_match_contract_detects_missing_column() -> None:
    """Drift surfaces with a clear missing-column message + remediation hint."""
    df = _make_compliant_df().drop(columns=["gender"])
    with pytest.raises(SchemaContractError) as exc_info:
        assert_df_columns_match_contract(df)
    msg = str(exc_info.value)
    assert "Missing columns" in msg
    assert "gender" in msg
    assert "_schema_dump.py" in msg, (
        "Drift message should cite the regeneration script as remediation."
    )


def test_assert_df_columns_match_contract_detects_extra_column() -> None:
    """Drift surfaces with a clear extra-column message."""
    df = _make_compliant_df()
    df["new_pa_warehouse_column"] = []
    with pytest.raises(SchemaContractError) as exc_info:
        assert_df_columns_match_contract(df)
    msg = str(exc_info.value)
    assert "Extra columns" in msg
    assert "new_pa_warehouse_column" in msg


# --------------------------------------------------------------------- #
# Layer 3 — CSV fallback path                                            #
# --------------------------------------------------------------------- #


def test_load_attrition_features_local_roundtrip(tmp_path: Path) -> None:
    """`load_attrition_features_local()` reads a CSV snapshot and validates schema."""
    snap_path = tmp_path / "snapshot.csv"
    df = _make_compliant_df()
    # Add at least one row so to_csv produces a non-trivial file.
    contract = load_contract()
    row = {col: ("x" if spec["data_type"] == "STRING" else 1) for col, spec in contract.items()}
    df = pd.DataFrame([row])
    df.to_csv(snap_path, index=False)

    loaded = load_attrition_features_local(snap_path)
    assert set(loaded.columns) == set(contract.keys())
    assert len(loaded) == 1


def test_load_attrition_features_local_missing_file_raises(tmp_path: Path) -> None:
    """Clear remediation when no CSV snapshot exists yet."""
    with pytest.raises(FileNotFoundError, match="Snapshot CSV not found"):
        load_attrition_features_local(tmp_path / "missing.csv")


def test_load_attrition_features_local_drift_raises(tmp_path: Path) -> None:
    """If a CSV has stale columns, the loader's contract check catches it."""
    snap_path = tmp_path / "stale.csv"
    pd.DataFrame({"old_col": [1, 2, 3]}).to_csv(snap_path, index=False)
    with pytest.raises(SchemaContractError):
        load_attrition_features_local(snap_path)


# --------------------------------------------------------------------- #
# Layer 4 — Live BigQuery contract test (Story 1.1.8 — Risk 2 canary)    #
# --------------------------------------------------------------------- #


@_SKIP_NO_CREDS
def test_v_attrition_features_schema_against_live_bq() -> None:
    """Story 1.1.8 — the contract test that catches pa-warehouse schema drift.

    Compares the LIVE BigQuery table schema (via the free `tables.get` API,
    zero bytes scanned) against the documented contract. Fails CI loudly if
    pa-warehouse changes the mart without updating the contract.

    Skipped automatically when no SA key is configured.
    """
    from retention.data.contract import assert_bq_schema_matches_contract
    from retention.data.load import _get_bq_client, _resolve_sa_key

    client = _get_bq_client(_resolve_sa_key())
    assert_bq_schema_matches_contract(
        client=client,
        project_id=config.BQ_PROJECT_ID,
        dataset=config.BQ_DATASET_MARTS,
        table=config.BQ_TABLE_ATTRITION_FEATURES,
    )


# --------------------------------------------------------------------- #
# Layer 5 — Loader dry-run                                               #
# --------------------------------------------------------------------- #


@_SKIP_NO_CREDS
def test_load_attrition_features_dry_run_returns_empty_df() -> None:
    """`dry_run=True` skips execution but still emits the cost estimate.

    This is the cost-discipline canary — a developer can dry-run before
    committing to a real scan and confirm the bytes estimate is reasonable.
    """
    from retention.data.load import load_attrition_features

    df = load_attrition_features(dry_run=True, snapshot_to_csv=False)
    assert df.empty, "dry_run=True must return an empty DataFrame (no execution)."
