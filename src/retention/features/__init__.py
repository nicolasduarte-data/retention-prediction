"""features subpackage — FeatureSpec catalog + cohort filtering + preprocessing.

Public API surface — re-exports so callers can use the short
`from retention.features import FEATURE_CATALOG` form.
"""

from retention.features.catalog import (
    FEATURE_CATALOG,
    FeatureSpec,
    get_feature_names,
    get_identifier_names,
    get_label_name,
    get_mutable_features,
    get_protected_features,
    get_spec,
    get_temporal_anchor,
)

__all__ = [
    "FEATURE_CATALOG",
    "FeatureSpec",
    "get_feature_names",
    "get_identifier_names",
    "get_label_name",
    "get_mutable_features",
    "get_protected_features",
    "get_spec",
    "get_temporal_anchor",
]
