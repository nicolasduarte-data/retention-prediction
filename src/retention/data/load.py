"""BigQuery data loader for `marts.v_attrition_features`.

The single entry point for getting attrition features into the project. Two
loading paths:

- `load_attrition_features()` — live BigQuery via `pandas-gbq`. Cost-capped,
  dry-run-capable, optionally snapshots to CSV for offline reproducibility.
- `load_attrition_features_local()` — load from a previously-snapshotted CSV.
  Lets reviewers without GCP credentials still run the pipeline (Story 1.1.5).

Both paths return a DataFrame that has been validated against the schema
contract at `docs/integration_contract.md` (Story 1.1.8 — Risk 2 mitigation).

**Cost discipline (Story 1.1.9 — Risk 8):**
- `maximum_bytes_billed` is hard-capped (default 50 MB). The full mart fits
  in well under 1 MB, so this is paranoia-level safe.
- `dry_run=True` returns the BigQuery cost estimate WITHOUT running the query,
  letting callers verify cost expectations before paying.
- All cost parameters are env-overridable so a reviewer running against
  a paid GCP project can tighten or loosen the cap.

**Reproducibility (Story 1.1.2 + Risk 6 mitigation):**
- The query is `SELECT * FROM ... ORDER BY employee_id`. Deterministic row
  order across re-runs even when BigQuery's default sort differs.
- Snapshots saved to `data/processed/v_attrition_features_YYYY-MM-DD.csv`
  use UTC date so cross-machine reproducibility holds.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from retention import config
from retention.data.contract import (
    SchemaContractError,
    assert_df_columns_match_contract,
    load_contract,
)

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

# Default hard cap: 50 MB. The full mart scans <1 MB today; this leaves
# room for future column growth without runaway cost.
DEFAULT_MAX_BYTES_BILLED = 50 * 1024 * 1024


def _resolve_sa_key(override: Path | None = None) -> Path:
    """Cross-platform service-account key path resolution.

    Order of precedence:
        1. Explicit `override` argument
        2. `PA_WAREHOUSE_SA_KEY` env var
        3. `~/.gcp/pa-warehouse-sa.json` default
    """
    if override is not None:
        return override
    env_value = os.getenv("PA_WAREHOUSE_SA_KEY")
    if env_value:
        return Path(env_value)
    return Path.home() / ".gcp" / "pa-warehouse-sa.json"


def _get_bq_client(sa_key: Path) -> bigquery.Client:
    """Build an authenticated BigQuery client from a service-account JSON."""
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if not sa_key.exists():
        raise FileNotFoundError(
            f"BigQuery service-account key not found at {sa_key}. "
            f"Set PA_WAREHOUSE_SA_KEY or place the SA JSON at the default path. "
            f"See `.env.example` for the documented contract."
        )

    # google-auth ships no type stubs for `from_service_account_file`; safe
    # call, documented usage. See `/feast r02` Tier 1 #8.
    creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
        str(sa_key)
    )
    return bigquery.Client(project=creds.project_id, credentials=creds)


def _estimate_bytes(client: bigquery.Client, query: str, project_id: str) -> int:
    """Run a BigQuery dry-run and return the estimated bytes processed.

    Story 1.1.9 — Risk 8 mitigation. Zero cost — BigQuery's `dry_run=True`
    returns the cost estimate without scanning any bytes.
    """
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config, project=project_id)
    return int(job.total_bytes_processed or 0)


def load_attrition_features(
    *,
    project_id: str | None = None,
    dataset: str | None = None,
    table: str | None = None,
    snapshot_to_csv: bool = True,
    snapshot_dir: Path | None = None,
    max_bytes_billed: int | None = None,
    dry_run: bool = False,
    sa_key: Path | None = None,
    enforce_contract: bool = True,
) -> pd.DataFrame:
    """Load `marts.v_attrition_features` from BigQuery.

    Args:
        project_id: GCP project. Defaults to `config.BQ_PROJECT_ID`.
        dataset: BQ dataset. Defaults to `config.BQ_DATASET_MARTS`.
        table: BQ table/view name. Defaults to `config.BQ_TABLE_ATTRITION_FEATURES`.
        snapshot_to_csv: If True (default), save the result as
            `<snapshot_dir>/<table>_<YYYY-MM-DD>.csv` for offline replay.
        snapshot_dir: Where to write the CSV snapshot. Defaults to
            `DATA_DIR / "processed"`.
        max_bytes_billed: Hard cap on bytes BigQuery is allowed to bill.
            Defaults to 50 MB. None disables the cap (NOT recommended for
            reviewer machines).
        dry_run: If True, return an empty DataFrame after logging the estimated
            bytes. Does NOT execute the query. Useful for cost confirmation.
        sa_key: Override the service-account JSON path.
        enforce_contract: If True (default), assert the loaded DataFrame's
            columns match `docs/integration_contract.md`. Raises
            `SchemaContractError` on drift.

    Returns:
        DataFrame with the configured columns (10 per the v0.1 contract),
        ordered by `employee_id`. Empty if `dry_run=True`.

    Raises:
        FileNotFoundError: SA key not at expected path.
        SchemaContractError: loaded columns disagree with the contract
            (only when `enforce_contract=True`).
    """
    import pandas_gbq

    project = project_id or config.BQ_PROJECT_ID
    dset = dataset or config.BQ_DATASET_MARTS
    tbl = table or config.BQ_TABLE_ATTRITION_FEATURES
    snap_dir = snapshot_dir or (config.DATA_DIR / "processed")
    cap = max_bytes_billed if max_bytes_billed is not None else DEFAULT_MAX_BYTES_BILLED

    # ORDER BY employee_id for deterministic row order (Story 1.1.2 — Risk 6).
    query = f"SELECT * FROM `{project}.{dset}.{tbl}` ORDER BY employee_id"

    key_path = _resolve_sa_key(sa_key)
    client = _get_bq_client(key_path)

    # Cost discipline — log the estimate before any real bytes are scanned.
    estimate_bytes = _estimate_bytes(client, query, project)
    logger.info(
        "load_attrition_features: dry-run estimate = %d bytes (~%.2f MB); "
        "max_bytes_billed cap = %d bytes (~%.2f MB)",
        estimate_bytes,
        estimate_bytes / (1024 * 1024),
        cap,
        cap / (1024 * 1024),
    )

    if dry_run:
        logger.info("load_attrition_features: dry_run=True — skipping execution.")
        return pd.DataFrame()

    if estimate_bytes > cap:
        raise RuntimeError(
            f"Query would scan {estimate_bytes:,} bytes (~{estimate_bytes / 1e6:.1f} MB), "
            f"exceeding max_bytes_billed cap of {cap:,} bytes "
            f"(~{cap / 1e6:.1f} MB). Pass max_bytes_billed=<higher> or investigate "
            f"why the mart grew."
        )

    # Use credentials directly with read_gbq — pandas-gbq picks them up.
    df = pandas_gbq.read_gbq(
        query,
        project_id=project,
        credentials=client._credentials,  # noqa: SLF001 — re-use authenticated creds
        configuration={
            "query": {
                "maximumBytesBilled": str(cap),
            }
        },
        progress_bar_type=None,
    )

    if enforce_contract:
        assert_df_columns_match_contract(df)

    if snapshot_to_csv:
        snap_dir.mkdir(parents=True, exist_ok=True)
        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        snap_path = snap_dir / f"{tbl}_{today}.csv"
        df.to_csv(snap_path, index=False)
        logger.info(
            "load_attrition_features: snapshot written to %s (%d rows × %d cols)",
            snap_path,
            len(df),
            df.shape[1],
        )

    return df


def load_attrition_features_local(
    csv_path: Path,
    *,
    enforce_contract: bool = True,
) -> pd.DataFrame:
    """Load a previously-snapshotted CSV without touching BigQuery.

    Story 1.1.5 — reviewer reproducibility path. Anyone with the repo + a
    committed CSV snapshot can run the pipeline without GCP credentials.

    Args:
        csv_path: Path to the snapshot CSV (typically
            `data/processed/v_attrition_features_YYYY-MM-DD.csv`).
        enforce_contract: If True (default), assert columns match the contract.

    Returns:
        DataFrame loaded from CSV.

    Raises:
        FileNotFoundError: CSV missing.
        SchemaContractError: columns disagree with the contract.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Snapshot CSV not found at {csv_path}. "
            f"Run `load_attrition_features(snapshot_to_csv=True)` once with "
            f"BigQuery credentials to generate it."
        )

    df = pd.read_csv(csv_path)

    if enforce_contract:
        assert_df_columns_match_contract(df)

    return df


__all__ = [
    "DEFAULT_MAX_BYTES_BILLED",
    "SchemaContractError",
    "load_attrition_features",
    "load_attrition_features_local",
    "load_contract",
]
