"""Dual-cohort splitting — Story 1.3.

Implements the controlled experiment at the heart of Loop 2:
HRIS-only feature set vs HRIS+survey hybrid feature set.

Both cohorts use IDENTICAL ROWS (all employees, same label, same temporal
split). The ONLY difference is which columns are given to the model. This
makes the comparison apples-to-apples: any AUC-PR difference is attributable
to the survey signal, not to a different employee population.

    hris_only: all rows, only columns where cohort in ('both',)
    hybrid:    all rows, columns where cohort in ('both', 'hybrid_only')

Why not split by who HAS survey data?
    That would leave only ~104 rows in HRIS-only (the 8.2% with no survey
    response), creating a severe class-imbalance artifact between cohorts.
    The model would be comparing different populations, not different
    feature sets. We want to isolate the signal, not the population.

The preprocessor handles missing survey values in the hybrid cohort:
    - ~8.2% of employees have no survey response -> survey columns are NaN
    - SimpleImputer(strategy='median') fills these at transform time
    - The imputation is logged in docs/data_card.md -> Known Limitations

Story 1.3: see `backlog/epic-1-data-layer.md -> Story 1.3`.
"""

from __future__ import annotations

import pandas as pd

from retention.features.catalog import FEATURE_CATALOG


def _cohort_columns(cohort: str) -> list[str]:
    """Return the column names that belong to a given cohort.

    Args:
        cohort: One of 'hris_only' or 'hybrid'.

    Returns:
        Ordered list of column names: identifiers + temporal anchor
        + cohort-appropriate features + label. Non-feature columns
        (identifier, temporal_anchor, label) are always included
        regardless of cohort — the model training code is responsible
        for separating X from y.

    Raises:
        ValueError: cohort is not one of the two known values.

    Why include identifiers and label here?
        split_cohorts() returns raw DataFrames, not X/y pairs.
        The caller (train script, notebook) performs the X/y split.
        Keeping all roles together avoids partial-read bugs where
        a caller forgets to re-attach the label after cohort filtering.
    """
    if cohort not in ("hris_only", "hybrid"):
        raise ValueError(
            f"cohort must be 'hris_only' or 'hybrid', got {cohort!r}. "
            "No other cohort values are defined in this project."
        )

    # Non-feature columns always travel with the data regardless of cohort.
    # These are identifiers, temporal anchors, and the label — they have
    # no 'cohort' concept because they're not model inputs.
    non_feature_roles = {"identifier", "temporal_anchor", "label"}

    allowed_feature_cohorts: set[str]
    if cohort == "hris_only":
        # HRIS-only: only features shared across both cohorts (no survey signal)
        allowed_feature_cohorts = {"both"}
    else:
        # hybrid: all features — HRIS + survey signal
        allowed_feature_cohorts = {"both", "hybrid_only"}

    cols: list[str] = []
    for spec in FEATURE_CATALOG:
        if spec.role in non_feature_roles:
            cols.append(spec.name)
        elif spec.role == "feature" and spec.cohort in allowed_feature_cohorts:
            cols.append(spec.name)
        # else: feature belonging to a different cohort -> silently excluded

    return cols


def split_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a raw attrition DataFrame into two cohort DataFrames.

    Both returned DataFrames have IDENTICAL row indices. The only
    difference is which columns are present:

        hris_only  ->  10 HRIS columns + identifier + temporal anchor + label
        hybrid     ->  13 columns (HRIS + 3 survey signal) + identifier + ...

    Args:
        df: Raw DataFrame loaded from `marts.v_attrition_features`.
            Expected to have passed `assert_df_columns_match_contract`.
            All 13 contract columns must be present. Row order is
            preserved from the input; index is not reset.

    Returns:
        dict with keys 'hris_only' and 'hybrid'. Each value is a
        pd.DataFrame slice — a VIEW of the input (no copy), so do
        not mutate them without calling .copy() first.

    Raises:
        KeyError: A required column is missing from `df`. This means
            the contract has drifted and the mart hasn't been refreshed,
            or `assert_df_columns_match_contract` was skipped.

    Example::

        df = load_attrition_features_local(csv_path)
        cohorts = split_cohorts(df)
        X_hris = cohorts['hris_only'].drop(columns=['voluntary_exit_label', ...])
        X_hybrid = cohorts['hybrid'].drop(columns=['voluntary_exit_label', ...])
        # cohorts['hris_only'].index == cohorts['hybrid'].index  <- always True

    Teaching note — why identical indices matter:
        Downstream, temporal_split() divides employees into train/val/test
        by index. If both cohorts use the same index, the SAME employees
        land in each split — model comparison is never contaminated by
        different employees in different splits.
    """
    hris_cols = _cohort_columns("hris_only")
    hybrid_cols = _cohort_columns("hybrid")

    # Validate that all required columns exist in the input.
    # KeyError on missing column is intentional — it surfaces contract drift.
    return {
        "hris_only": df[hris_cols],
        "hybrid": df[hybrid_cols],
    }


def get_cohort_feature_names(cohort: str) -> list[str]:
    """Return feature-only column names for a cohort (no identifiers, no label).

    Use this when you need the X column list before you have a DataFrame,
    e.g. to pre-configure a ColumnTransformer for a specific cohort.

    Args:
        cohort: 'hris_only' or 'hybrid'.

    Returns:
        List of column names with role == 'feature' for the cohort.
        Order matches FEATURE_CATALOG insertion order.
    """
    if cohort not in ("hris_only", "hybrid"):
        raise ValueError(f"cohort must be 'hris_only' or 'hybrid', got {cohort!r}.")

    allowed: set[str]
    if cohort == "hris_only":
        allowed = {"both"}
    else:
        allowed = {"both", "hybrid_only"}

    return [
        spec.name for spec in FEATURE_CATALOG if spec.role == "feature" and spec.cohort in allowed
    ]


__all__ = [
    "split_cohorts",
    "get_cohort_feature_names",
]
