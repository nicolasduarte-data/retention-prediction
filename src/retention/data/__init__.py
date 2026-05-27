"""data subpackage — BigQuery loader + integration-contract enforcement.

Public API surface (re-exported here so callers can `from retention.data
import load_attrition_features` rather than the longer path).
"""

from retention.data.contract import (
    SchemaContractError,
    assert_bq_schema_matches_contract,
    assert_df_columns_match_contract,
    load_contract,
)
from retention.data.load import (
    DEFAULT_MAX_BYTES_BILLED,
    load_attrition_features,
    load_attrition_features_local,
)
from retention.data.split import assert_no_temporal_leak, temporal_split

__all__ = [
    "DEFAULT_MAX_BYTES_BILLED",
    "SchemaContractError",
    "assert_bq_schema_matches_contract",
    "assert_df_columns_match_contract",
    "assert_no_temporal_leak",
    "load_attrition_features",
    "load_attrition_features_local",
    "load_contract",
    "temporal_split",
]
