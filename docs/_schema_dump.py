"""Story 0.5.0 — Schema discovery spike.

Dumps the live `INFORMATION_SCHEMA.COLUMNS` for `pa-warehouse-prod.marts.v_attrition_features`
and writes a markdown integration contract. One-shot script; safe to re-run.

Timestamp semantic: the file's timestamp reflects the *actual moment of last
regeneration in UTC*. Future re-runs intentionally produce different timestamps
so schema drift surfaces in `git status` / PR review. Story 1.1.8 (cross-project
contract test) automates this comparison in CI once Story 1.1 loader ships.

Configuration (cross-platform, env-overridable):
- `PA_WAREHOUSE_SA_KEY` — absolute path to the BigQuery service-account JSON.
  Defaults to `~/.gcp/pa-warehouse-sa.json` (resolves to the running user's home
  on Windows / macOS / Linux). See `.env.example` (Story 0.4) for the documented
  contract.
- `BQ_PROJECT_ID` / `BQ_DATASET_MARTS` — see `retention.config`. This script does
  NOT import `retention.config` because it needs to run before the package is
  editable-installed.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

# --- Cross-platform SA key resolution ---
_DEFAULT_SA_KEY = Path.home() / ".gcp" / "pa-warehouse-sa.json"
SA_KEY = Path(os.getenv("PA_WAREHOUSE_SA_KEY", str(_DEFAULT_SA_KEY)))

PROJECT = os.getenv("BQ_PROJECT_ID", "pa-warehouse-prod")
DATASET = os.getenv("BQ_DATASET_MARTS", "marts")
TABLE = os.getenv("BQ_TABLE_ATTRITION_FEATURES", "v_attrition_features")
OUT = Path(__file__).resolve().parent / "integration_contract.md"


def main() -> None:
    """Dump the live schema and write the contract. See module docstring."""
    if not SA_KEY.exists():
        raise SystemExit(
            f"SA key not found at {SA_KEY}. "
            f"Set PA_WAREHOUSE_SA_KEY env var or place key at the default path."
        )

    # google-auth ships no type stubs for `from_service_account_file`; safe call,
    # documented usage. mypy strict otherwise rejects untyped-call in typed context.
    creds = service_account.Credentials.from_service_account_file(str(SA_KEY))  # type: ignore[no-untyped-call]
    client = bigquery.Client(project=PROJECT, credentials=creds)

    # Cost discipline: cap at 10 MB just for INFORMATION_SCHEMA — trivial query.
    # Table name is parameterized to keep the pattern injection-safe (see r02 #10);
    # project + dataset go in backticks because BQ identifiers can't be parameterized.
    query = f"""
    SELECT
      ordinal_position,
      column_name,
      data_type,
      is_nullable
    FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @table_name
    ORDER BY ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=10 * 1024 * 1024,
        query_parameters=[
            bigquery.ScalarQueryParameter("table_name", "STRING", TABLE),
        ],
    )
    rows = list(client.query(query, job_config=job_config).result())

    if not rows:
        raise SystemExit(f"Empty result for {PROJECT}.{DATASET}.{TABLE} — does the table exist?")

    # Timezone-aware UTC timestamp — see module docstring.
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Integration Contract — `{DATASET}.{TABLE}`",
        "",
        f"Schema dumped from `{PROJECT}.{DATASET}.{TABLE}` at **{now}** via BigQuery "
        "`INFORMATION_SCHEMA` (`docs/_schema_dump.py`). Ground-truth contract for the "
        "retention-prediction loader (Story 1.1.8).",
        "",
        f"**Row count in schema dump:** {len(rows)} columns",
        "",
        "| # | column_name | data_type | is_nullable |",
        "|---|-------------|-----------|-------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.ordinal_position} | `{r.column_name}` | `{r.data_type}` | {r.is_nullable} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "_Timestamp reflects the actual moment of last regeneration (UTC). To verify "
        "the current mart schema matches this contract, re-run `python docs/_schema_dump.py` "
        "— any column-level diff surfaces in `git status`. Story 1.1.8 (cross-project "
        "contract test) automates this comparison in CI once the Story 1.1 loader ships._"
    )
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} — {len(rows)} columns @ {now}")


if __name__ == "__main__":
    main()
