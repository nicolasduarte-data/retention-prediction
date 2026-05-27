"""Integration contract parser + schema-assertion helpers.

`docs/integration_contract.md` is the source of truth for the schema of
`pa-warehouse-prod.marts.v_attrition_features`. This module:

1. Parses that file into a Python dict (`load_contract()`).
2. Provides two assertion helpers:
   - `assert_bq_schema_matches_contract()` — rigorous live-BigQuery check used
     by the Story 1.1.8 contract test. Compares actual mart schema (via
     `bigquery.Client.get_table().schema`) against the contract column-by-column
     including type-level checks.
   - `assert_df_columns_match_contract()` — fast column-set check used by the
     loader at runtime. No BigQuery roundtrip; works on CSV-loaded DataFrames too.

Story 1.1.6 + 1.1.8 (Risk 2 — pa-warehouse contract drift) — see prey notes
and `BACKLOG.md → ## ⚠️ Risk Register`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from google.cloud import bigquery


# --------------------------------------------------------------------- #
# Custom exception                                                       #
# --------------------------------------------------------------------- #


class SchemaContractError(AssertionError):
    """Raised when a DataFrame or BigQuery table schema diverges from the
    documented contract in `docs/integration_contract.md`.

    Subclass of AssertionError so pytest reports it as a clean test failure
    (not an unexpected exception), while still being catchable as a real
    error type if calling code wants to handle drift gracefully.
    """


# --------------------------------------------------------------------- #
# Contract markdown parser                                               #
# --------------------------------------------------------------------- #


_TABLE_ROW = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(YES|NO)\s*\|\s*$")

# BigQuery exposes two parallel type-naming conventions:
#   - INFORMATION_SCHEMA.COLUMNS returns Standard SQL names (INT64, FLOAT64, BOOL)
#   - bigquery.SchemaField.field_type returns Legacy SQL names (INTEGER, FLOAT, BOOLEAN)
# The contract file is generated from INFORMATION_SCHEMA (standard names). Normalize
# legacy → standard before any cross-API comparison so the contract test reflects
# actual drift, not naming-convention differences.
_LEGACY_TO_STANDARD = {
    "INTEGER": "INT64",
    "FLOAT": "FLOAT64",
    "BOOLEAN": "BOOL",
    # STRING, DATE, BYTES, TIMESTAMP, DATETIME, TIME, NUMERIC, BIGNUMERIC, GEOGRAPHY,
    # RECORD/STRUCT, JSON all use the same name in both conventions — pass-through.
}


def _normalize_bq_type(field_type: str) -> str:
    """Map a BigQuery legacy SQL type name to its Standard SQL equivalent."""
    return _LEGACY_TO_STANDARD.get(field_type.upper(), field_type.upper())


def load_contract(contract_path: Path | None = None) -> dict[str, dict[str, str]]:
    """Parse `docs/integration_contract.md` into a column-keyed dict.

    Returns:
        {column_name: {"data_type": "STRING|INT64|FLOAT64|BOOL|DATE",
                       "is_nullable": "YES|NO"}, ...}

    Args:
        contract_path: Override the contract file location. Defaults to
            `docs/integration_contract.md` resolved relative to the package's
            project root (works for both editable install + source tree).

    Raises:
        FileNotFoundError: contract file missing.
        SchemaContractError: contract file present but table is empty / unparseable.
    """
    if contract_path is None:
        from retention import config

        contract_path = config.PROJECT_ROOT / "docs" / "integration_contract.md"

    if not contract_path.exists():
        raise FileNotFoundError(
            f"Integration contract not found at {contract_path}. "
            f"Regenerate via `python docs/_schema_dump.py`."
        )

    schema: dict[str, dict[str, str]] = {}
    for line in contract_path.read_text(encoding="utf-8").splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            column, data_type, is_nullable = match.groups()
            schema[column] = {"data_type": data_type, "is_nullable": is_nullable}

    if not schema:
        raise SchemaContractError(
            f"No schema rows parsed from {contract_path}. "
            f"The file's table format may have changed — expected rows like "
            f"`| 1 | `employee_id` | `STRING` | YES |`."
        )

    return schema


# --------------------------------------------------------------------- #
# Assertion helpers                                                      #
# --------------------------------------------------------------------- #


def _format_drift_message(
    missing: set[str],
    extra: set[str],
    type_mismatches: list[tuple[str, str, str]] | None = None,
) -> str:
    """Build a uniform error message for any contract drift."""
    lines = ["Schema drift detected between live data and integration contract:"]
    if missing:
        lines.append(f"  Missing columns (in contract, not in data): {sorted(missing)}")
    if extra:
        lines.append(f"  Extra columns (in data, not in contract):   {sorted(extra)}")
    if type_mismatches:
        lines.append("  Type mismatches (column: contract → actual):")
        for col, expected, actual in sorted(type_mismatches):
            lines.append(f"    {col}: {expected} → {actual}")
    lines.append(
        "Update `docs/integration_contract.md` by re-running "
        "`python docs/_schema_dump.py` and review the diff before committing."
    )
    return "\n".join(lines)


def assert_df_columns_match_contract(
    df: pd.DataFrame,
    contract: dict[str, dict[str, str]] | None = None,
) -> None:
    """Assert that a loaded DataFrame's columns match the contract column set.

    Fast path used by the loader at runtime — column-set check only, no type
    checks. Works regardless of data source (BigQuery, CSV, parquet).

    Args:
        df: DataFrame to validate (typically the loader's output).
        contract: Pre-parsed contract dict. If None, loads fresh from disk.

    Raises:
        SchemaContractError: if columns differ.
    """
    if contract is None:
        contract = load_contract()

    df_cols = set(df.columns)
    expected_cols = set(contract.keys())

    missing = expected_cols - df_cols
    extra = df_cols - expected_cols

    if missing or extra:
        raise SchemaContractError(_format_drift_message(missing, extra))


def assert_bq_schema_matches_contract(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    table: str,
    contract: dict[str, dict[str, str]] | None = None,
) -> None:
    """Assert that a live BigQuery table's schema matches the contract.

    Rigorous check used by the Story 1.1.8 contract test. Compares both
    column membership AND BigQuery type strings (STRING, INT64, etc.).
    Costs zero bytes — uses the `tables.get` metadata API, not a SELECT.

    Args:
        client: An authenticated bigquery.Client.
        project_id, dataset, table: Identify the mart to check.
        contract: Pre-parsed contract dict. If None, loads fresh.

    Raises:
        SchemaContractError: on any drift (columns or types).
    """
    if contract is None:
        contract = load_contract()

    table_ref = f"{project_id}.{dataset}.{table}"
    bq_table = client.get_table(table_ref)
    # Normalize legacy names (INTEGER/FLOAT/BOOLEAN) to standard (INT64/FLOAT64/BOOL)
    # so the comparison reflects real drift, not API naming-convention drift.
    actual_schema: dict[str, str] = {
        field.name: _normalize_bq_type(field.field_type) for field in bq_table.schema
    }

    actual_cols = set(actual_schema.keys())
    expected_cols = set(contract.keys())

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    type_mismatches: list[tuple[str, str, str]] = []
    for col in expected_cols & actual_cols:
        expected_type = contract[col]["data_type"]
        actual_type = actual_schema[col]
        if expected_type != actual_type:
            type_mismatches.append((col, expected_type, actual_type))

    if missing or extra or type_mismatches:
        raise SchemaContractError(_format_drift_message(missing, extra, type_mismatches))
