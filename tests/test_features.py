"""Story 1.2.4 + 1.5.5 Layer 1 + 1.3 — Tests for FEATURE_CATALOG and cohort splitting.

Four responsibilities:

1. **Catalog coverage** — every column in `docs/integration_contract.md` must
   have a `FeatureSpec`. Catches drift between the live mart contract and
   the project's documented understanding of it.

2. **Helper sanity** — `get_feature_names()`, `get_protected_features()`, etc.
   return the correct names per the catalog. Regression-protects refactors.

3. **Leakage rationale validation (Story 1.5.5 Layer 1)** — every feature's
   `leakage_rationale` MUST cite a concrete pa-warehouse dbt model path or
   `_core.yml` metadata. Generic rationales like "looks safe" are rejected.
   This forces the catalog author to be specific, which is what makes the
   audit defensible to a reviewer.

4. **Cohort splitting (Story 1.3)** — `split_cohorts()` returns two DataFrames
   with identical row indices. The ONLY difference is which feature columns are
   present. This is the apples-to-apples requirement for the Loop 2 controlled
   experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from retention.data.contract import load_contract
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


# --------------------------------------------------------------------- #
# Catalog coverage (Story 1.2.4)                                         #
# --------------------------------------------------------------------- #


def test_catalog_covers_all_contract_columns() -> None:
    """Every column in `docs/integration_contract.md` must have a FeatureSpec.

    This is the schema-drift detector. When pa-warehouse adds a new column,
    `_schema_dump.py` regenerates the contract; this test fails until the
    column gets a catalog entry. Forces documentation discipline.
    """
    contract = load_contract()
    catalog_names = {s.name for s in FEATURE_CATALOG}
    contract_cols = set(contract.keys())

    missing_from_catalog = contract_cols - catalog_names
    extra_in_catalog = catalog_names - contract_cols

    assert not missing_from_catalog, (
        f"Contract columns missing from FEATURE_CATALOG: {sorted(missing_from_catalog)}. "
        f"Add FeatureSpec entries to src/retention/features/catalog.py."
    )
    assert not extra_in_catalog, (
        f"FEATURE_CATALOG has columns not in the integration contract: "
        f"{sorted(extra_in_catalog)}. Either remove them from the catalog or "
        f"regenerate the contract via `python docs/_schema_dump.py`."
    )


def test_catalog_has_exactly_one_label_and_one_temporal_anchor() -> None:
    """A well-formed catalog has one label column and one temporal anchor.

    More than one of either means downstream code (`temporal_split`,
    `model.fit(y=...)`) has ambiguous targets — likely a typo in role values.
    """
    labels = [s for s in FEATURE_CATALOG if s.role == "label"]
    anchors = [s for s in FEATURE_CATALOG if s.role == "temporal_anchor"]
    assert len(labels) == 1, f"Expected exactly 1 label; got {[s.name for s in labels]}"
    assert len(anchors) == 1, f"Expected exactly 1 temporal_anchor; got {[s.name for s in anchors]}"


def test_catalog_features_have_known_dtypes() -> None:
    """dtype must be one of the documented values — guards against typos."""
    allowed = {"numeric", "categorical", "boolean", "date", "string"}
    for spec in FEATURE_CATALOG:
        assert spec.dtype in allowed, (
            f"FeatureSpec({spec.name!r}).dtype={spec.dtype!r} not in {allowed}"
        )


# --------------------------------------------------------------------- #
# Helper sanity (Story 1.2.3)                                            #
# --------------------------------------------------------------------- #


def test_get_feature_names_excludes_id_label_and_anchor() -> None:
    """The list returned by get_feature_names() must NOT contain employee_id,
    snapshot_date, or voluntary_exit_label — those are role-distinct."""
    features = get_feature_names()
    assert "employee_id" not in features
    assert "snapshot_date" not in features
    assert "voluntary_exit_label" not in features
    # And there's at least one real feature
    assert len(features) > 0


def test_get_protected_features_contains_demographics() -> None:
    """Per paw-prey-005's promise, age + gender are the v0.1 protected attrs."""
    protected = set(get_protected_features())
    assert {"age_at_window_close", "gender"}.issubset(protected), (
        f"Expected {{age_at_window_close, gender}} ⊆ {protected}"
    )


def test_get_mutable_features_excludes_immutable_protected() -> None:
    """Age + gender must NEVER show up as mutable (DiCE actionability constraint)."""
    mutable = set(get_mutable_features())
    assert "age_at_window_close" not in mutable
    assert "gender" not in mutable
    # employee_id is also not mutable (and not a feature anyway).
    assert "employee_id" not in mutable


def test_get_identifier_names_returns_employee_id() -> None:
    assert get_identifier_names() == ["employee_id"]


def test_get_label_name_returns_voluntary_exit_label() -> None:
    assert get_label_name() == "voluntary_exit_label"


def test_get_temporal_anchor_returns_snapshot_date() -> None:
    assert get_temporal_anchor() == "snapshot_date"


def test_get_spec_returns_correct_dataclass() -> None:
    spec = get_spec("compa_ratio")
    assert isinstance(spec, FeatureSpec)
    assert spec.name == "compa_ratio"
    assert spec.role == "feature"
    assert spec.mutable is True
    assert spec.protected is False


def test_get_spec_unknown_column_raises_keyerror() -> None:
    import pytest

    with pytest.raises(KeyError, match="No FeatureSpec for column"):
        get_spec("nonexistent_column")


# --------------------------------------------------------------------- #
# Leakage rationale validation (Story 1.5.5 Layer 1)                     #
# --------------------------------------------------------------------- #


def test_all_specs_marked_leakage_audited() -> None:
    """v0.1 invariant — every catalog entry must be explicitly audited.

    `leakage_audited=False` blocks Epic 2 modeling. Loop 2 will tighten the
    leakage gate further with semantic + correlation + pattern layers
    (Story 1.5.5 Layers 2-4); Layer 1 here is the explicit allowlist gate.
    """
    unaudited = [s.name for s in FEATURE_CATALOG if not s.leakage_audited]
    assert not unaudited, (
        f"FEATURE_CATALOG has unaudited entries: {unaudited}. "
        f"Set leakage_audited=True with a specific dbt-citing rationale."
    )


def test_leakage_rationales_cite_pa_warehouse_paths() -> None:
    """Story 1.5.5 Layer 1 — generic rationales fail this test.

    Every rationale must include either `dbt/models/` (specific dbt model
    citation) OR `_core.yml` (pa-warehouse metadata layer reference). Strings
    like 'looks safe' or 'computed before label' without source pointers are
    not defensible to a reviewer.
    """
    failures: list[str] = []
    for spec in FEATURE_CATALOG:
        rationale = spec.leakage_rationale.lower()
        if "dbt/models/" not in rationale and "_core.yml" not in rationale:
            failures.append(
                f"  {spec.name}: rationale missing dbt/models/ or _core.yml citation "
                f"— got {spec.leakage_rationale[:80]!r}"
            )
    assert not failures, (
        "Layer 1 leakage gate — rationales lack pa-warehouse source citation:\n"
        + "\n".join(failures)
    )


def test_leakage_rationales_non_trivial_length() -> None:
    """A rationale shorter than 40 chars is not defensible — force detail."""
    too_short = [
        (s.name, len(s.leakage_rationale)) for s in FEATURE_CATALOG if len(s.leakage_rationale) < 40
    ]
    assert not too_short, (
        f"Some leakage_rationales are too short (< 40 chars): {too_short}. "
        f"Expand to explain WHY the feature precedes the label."
    )


# --------------------------------------------------------------------- #
# Leakage gate — Layer 2: Semantic temporal check (Story 1.5.5)         #
# --------------------------------------------------------------------- #
#
# snapshot_date_offset_months encodes when each feature is measured
# relative to the observation window end. Rules:
#   - Must be non-negative (negative = using future data)
#   - Label must be 0 (the thing we predict is not a lookback)
#   - Offsets > 12 are suspect — flag them explicitly (stale feature risk)
#
# Layer 2 is catalog-level: purely structural, no data needed.
# --------------------------------------------------------------------- #


def test_layer2_no_negative_snapshot_offsets() -> None:
    """Layer 2 — no feature is measured AFTER the observation window.

    A negative snapshot_date_offset_months would mean the feature value
    was captured in the future relative to snapshot_date — a direct temporal
    leak. This should never appear in the catalog.
    """
    violations = [
        (s.name, s.snapshot_date_offset_months)
        for s in FEATURE_CATALOG
        if s.snapshot_date_offset_months < 0
    ]
    assert not violations, (
        f"Layer 2 gate: features with negative offset (future data): {violations}. "
        "snapshot_date_offset_months must be >= 0 for all catalog entries."
    )


def test_layer2_label_has_zero_offset() -> None:
    """Layer 2 — the label must not be treated as a lookback.

    voluntary_exit_label is the event we're predicting. Its
    snapshot_date_offset_months must be 0 — it is not measured before
    snapshot_date, it is the outcome that FOLLOWS it.
    """
    for spec in FEATURE_CATALOG:
        if spec.role == "label":
            assert spec.snapshot_date_offset_months == 0, (
                f"Label column '{spec.name}' has offset "
                f"{spec.snapshot_date_offset_months} — must be 0. "
                "The label is the future event, not a lookback."
            )


def test_layer2_no_implausibly_stale_offsets() -> None:
    """Layer 2 — offsets > 12 months require scrutiny.

    A feature measured 13+ months before snapshot_date is likely stale:
    an annual review system would need a 12-month offset at most. Anything
    beyond 12 months should be explicitly reviewed — add a comment in
    FEATURE_CATALOG explaining why the stale value is still predictive.

    Current catalog maximum: 3 months (quarterly review / survey cadence).
    This test fails if a new entry accidentally uses a large offset without
    human review.
    """
    MAX_PLAUSIBLE_OFFSET = 12  # months
    suspect = [
        (s.name, s.snapshot_date_offset_months)
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.snapshot_date_offset_months > MAX_PLAUSIBLE_OFFSET
    ]
    assert not suspect, (
        f"Layer 2 gate: features with offset > {MAX_PLAUSIBLE_OFFSET} months: {suspect}. "
        "Offsets this large suggest stale data. Review and add explicit justification "
        "to the catalog entry's leakage_rationale before proceeding."
    )


def test_layer2_feature_offsets_match_source_cadence() -> None:
    """Layer 2 — survey features must have non-zero offset; HRIS snapshots may be 0.

    Survey signal is never measured at exactly snapshot_date — pulse surveys
    run on a cadence (quarterly at minimum). A survey feature with offset=0
    would imply it was captured at the exact moment of window close, which
    is implausible and should be reviewed.

    HRIS features (tenure, compa_ratio, etc.) are point-in-time at
    snapshot_date, so offset=0 is correct for them.
    """
    survey_at_zero = [
        s.name
        for s in FEATURE_CATALOG
        if s.role == "feature" and s.source == "survey" and s.snapshot_date_offset_months == 0
    ]
    assert not survey_at_zero, (
        f"Layer 2 gate: survey features with offset=0: {survey_at_zero}. "
        "Survey signal cannot be measured at exactly snapshot_date (no real-time "
        "pulse survey); set snapshot_date_offset_months >= 1."
    )


# --------------------------------------------------------------------- #
# Leakage gate — Layer 3: Empirical correlation check (Story 1.5.5)     #
# --------------------------------------------------------------------- #
#
# Load the most recent CSV snapshot. For every feature column, compute
# abs(Spearman rank correlation) vs the label. Correlations >= 0.70
# are a red flag for data leakage — they suggest the feature encodes
# label information directly (e.g. tenure computed AFTER exit date).
#
# Threshold: 0.70 is deliberately permissive. Genuine retention signals
# (e.g. compa_ratio) might reach 0.4-0.5 in a well-designed dataset;
# anything above 0.70 in a cross-sectional dataset is suspicious.
#
# Skip behaviour: if no CSV snapshot is found, skip with a clear message.
# This keeps CI green for reviewers without BQ credentials.
# --------------------------------------------------------------------- #


def _find_latest_snapshot() -> Path | None:
    """Return the most recently modified v_attrition_features CSV, or None."""
    snapshot_dir = Path(__file__).parents[1] / "data" / "raw"
    candidates = sorted(
        snapshot_dir.glob("v_attrition_features_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def test_layer3_no_high_correlation_with_label() -> None:
    """Layer 3 — empirical Spearman gate: abs(rho) < 0.70 for all features.

    Runs against the most recent CSV snapshot in data/raw/.
    Skipped if no snapshot is present (reviewer without BQ creds).

    What 0.70 means in context:
        compa_ratio genuinely predicts exits (expected rho ~0.2-0.5).
        A feature with rho > 0.70 likely post-dates the exit event
        (e.g. "months_since_last_review" computed after the exit date),
        which is a data leak. Flag it for manual investigation.

    This test does NOT catch subtle leakage (rho=0.50 on a leaked feature
    still passes). It catches catastrophic leakage — the gross sanity check.
    Layer 1 rationale + Layer 2 temporal offset are the subtle guards.
    """
    import pytest
    from scipy.stats import spearmanr

    snapshot = _find_latest_snapshot()
    if snapshot is None:
        pytest.skip(
            "No v_attrition_features_*.csv found in data/raw/ — "
            "Layer 3 empirical gate skipped. Run load_attrition_features() "
            "with BigQuery credentials to generate the snapshot."
        )

    df = pd.read_csv(snapshot)
    label = df["voluntary_exit_label"].astype(float)

    feature_specs = [s for s in FEATURE_CATALOG if s.role == "feature"]
    violations: list[str] = []

    for spec in feature_specs:
        if spec.name not in df.columns:
            continue  # snapshot may predate a new column; Layer 1 catches drift
        col = df[spec.name]
        # Drop rows where the feature is null (survey non-response is expected)
        mask = col.notna()
        if mask.sum() < 50:  # noqa: PLR2004 — minimum sample for correlation
            continue  # too few non-null values to compute meaningful correlation
        rho, _ = spearmanr(col[mask], label[mask])
        if abs(rho) >= 0.70:  # noqa: PLR2004
            violations.append(
                f"  {spec.name}: abs(Spearman)={abs(rho):.3f} >= 0.70 — "
                f"possible data leakage. Investigate before training."
            )

    assert not violations, (
        "Layer 3 gate — high feature-label correlations detected:\n"
        + "\n".join(violations)
        + "\nInvestigate whether these features encode post-exit information."
    )


# --------------------------------------------------------------------- #
# Leakage gate — Layer 4: Pattern-based safety net (Story 1.5.5)        #
# --------------------------------------------------------------------- #
#
# Static scan of feature names and descriptions for patterns that
# indicate the feature is computed from post-label events. This is a
# blunt instrument — it catches naming convention violations and
# obvious mistakes (a column named 'exit_date' getting into features).
#
# Does NOT catch subtle leakage (a benign-looking name can still leak).
# It is a final mechanical sweep, not a substitute for Layers 1-3.
# --------------------------------------------------------------------- #

# Patterns that must NOT appear in feature names or descriptions.
# These indicate the value was computed from post-exit events or
# identifies an exit directly (which would be a circular feature).
_FORBIDDEN_PATTERNS = (
    "exit_date",
    "termination_date",
    "last_day",
    "separation_date",
    "offboard",
    "post_exit",
    "after_termination",
    "post_termination",
    "days_since_exit",
    "time_to_exit",
    "exit_reason",
    "termination_reason",
    "rehire",
)


def test_layer4_no_termination_patterns_in_feature_names() -> None:
    """Layer 4 — feature names must not contain post-exit or termination patterns.

    A column named 'exit_date_daysago' or 'post_termination_tenure' would
    encode the label event directly. This test catches naming-convention
    violations before they reach training.
    """
    violations: list[str] = []
    for spec in FEATURE_CATALOG:
        if spec.role != "feature":
            continue
        lower_name = spec.name.lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in lower_name:
                violations.append(f"  {spec.name}: name contains forbidden pattern '{pattern}'")
    assert not violations, (
        "Layer 4 gate — termination/post-exit patterns in feature names:\n" + "\n".join(violations)
    )


def test_layer4_no_termination_patterns_in_descriptions() -> None:
    """Layer 4 — feature descriptions must not reference post-exit aggregations.

    A description like 'average salary after termination' or 'days between
    exit and last review' signals the value was derived post-event, even if
    the name is innocuous.
    """
    violations: list[str] = []
    for spec in FEATURE_CATALOG:
        if spec.role != "feature":
            continue
        lower_desc = spec.description.lower()
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in lower_desc:
                violations.append(
                    f"  {spec.name}: description contains forbidden pattern '{pattern}' "
                    f"— description: {spec.description[:80]!r}"
                )
    assert not violations, (
        "Layer 4 gate — termination/post-exit patterns in feature descriptions:\n"
        + "\n".join(violations)
    )


def test_layer4_feature_names_are_lowercase_snake_case() -> None:
    """Layer 4 — naming convention guard.

    Camel-case or uppercase column names often indicate a paste from a
    report or BI tool where calculated fields (possibly post-event) were
    copy-pasted in. Snake_case is the pa-warehouse + this project's
    convention for all mart columns.
    """
    import re

    violations = [
        spec.name for spec in FEATURE_CATALOG if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name)
    ]
    assert not violations, (
        f"Layer 4 gate — non-snake_case feature names: {violations}. "
        "All column names must be lowercase snake_case to match the mart convention."
    )


# --------------------------------------------------------------------- #
# Cohort splitting (Story 1.3)                                           #
# --------------------------------------------------------------------- #


def _make_full_df(n: int = 10) -> pd.DataFrame:
    """Minimal fixture with all 13 contract columns for cohort-split tests.

    Uses deterministic values so tests are reproducible across machines.
    Survey columns include NaN rows to simulate the 8.2% non-response rate.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "employee_id": [f"EMP_{i:03d}" for i in range(n)],
            "snapshot_date": pd.to_datetime(["2025-05-27"] * n),
            "tenure_months": rng.uniform(1, 60, n),
            "compa_ratio": rng.uniform(0.7, 1.3, n),
            "successor_count": rng.integers(0, 4, n).astype(float),
            "age_at_window_close": rng.uniform(22, 60, n),
            "performance_tier": rng.choice(["2", "3", "4", "5"], n),
            "gender": rng.choice(["M", "F", "NB"], n),
            "is_critical_role": rng.choice([True, False], n),
            "enps": rng.uniform(-100, 100, n),
            "engagement_score": rng.uniform(1, 5, n),
            "manager_relationship_score": rng.uniform(1, 5, n),
            "voluntary_exit_label": rng.choice([True, False], n),
        }
    )
    # Simulate ~20% non-response (NaN survey values)
    nan_mask = rng.random(n) < 0.2
    df.loc[nan_mask, ["enps", "engagement_score", "manager_relationship_score"]] = np.nan
    return df


def test_cohort_row_alignment() -> None:
    """Story 1.3 core requirement — identical row indices across cohorts.

    This is the apples-to-apples invariant: same employees, same labels,
    same temporal split boundaries in both arms. Any deviation means the
    model comparison is contaminated by population differences.
    """
    from retention.features.cohorts import split_cohorts

    df = _make_full_df(n=50)
    cohorts = split_cohorts(df)

    assert set(cohorts.keys()) == {"hris_only", "hybrid"}, (
        "split_cohorts must return exactly {'hris_only', 'hybrid'}"
    )
    assert cohorts["hris_only"].index.equals(cohorts["hybrid"].index), (
        "Row indices must be identical between hris_only and hybrid cohorts. "
        "Violation means model comparison is comparing different populations."
    )


def test_hris_cohort_excludes_survey_columns() -> None:
    """HRIS-only cohort must not contain survey signal columns."""
    from retention.features.cohorts import split_cohorts

    df = _make_full_df(n=20)
    cohorts = split_cohorts(df)

    survey_cols = {"enps", "engagement_score", "manager_relationship_score"}
    hris_cols = set(cohorts["hris_only"].columns)
    assert hris_cols.isdisjoint(survey_cols), (
        f"hris_only cohort contains survey columns: {hris_cols & survey_cols}. "
        "Survey signal must be absent for the HRIS-only arm."
    )


def test_hybrid_cohort_includes_survey_columns() -> None:
    """Hybrid cohort must include all three survey signal columns."""
    from retention.features.cohorts import split_cohorts

    df = _make_full_df(n=20)
    cohorts = split_cohorts(df)

    survey_cols = {"enps", "engagement_score", "manager_relationship_score"}
    hybrid_cols = set(cohorts["hybrid"].columns)
    assert survey_cols.issubset(hybrid_cols), (
        f"hybrid cohort is missing survey columns: {survey_cols - hybrid_cols}."
    )


def test_both_cohorts_preserve_row_count() -> None:
    """Neither cohort should gain or lose rows relative to the input."""
    from retention.features.cohorts import split_cohorts

    n = 30
    df = _make_full_df(n=n)
    cohorts = split_cohorts(df)

    assert len(cohorts["hris_only"]) == n, (
        f"hris_only row count changed: expected {n}, got {len(cohorts['hris_only'])}"
    )
    assert len(cohorts["hybrid"]) == n, (
        f"hybrid row count changed: expected {n}, got {len(cohorts['hybrid'])}"
    )


def test_both_cohorts_contain_label_column() -> None:
    """The label column must survive the cohort split — it's the y target."""
    from retention.features.cohorts import split_cohorts

    df = _make_full_df(n=10)
    cohorts = split_cohorts(df)

    assert "voluntary_exit_label" in cohorts["hris_only"].columns
    assert "voluntary_exit_label" in cohorts["hybrid"].columns


def test_invalid_cohort_raises_valueerror() -> None:
    """Passing an unknown cohort name to get_cohort_feature_names raises."""
    import pytest

    from retention.features.cohorts import get_cohort_feature_names

    with pytest.raises(ValueError, match="cohort must be"):
        get_cohort_feature_names("invalid_cohort")


def test_get_cohort_feature_names_hris_only_excludes_survey() -> None:
    """get_cohort_feature_names('hris_only') must not return survey columns."""
    from retention.features.cohorts import get_cohort_feature_names

    hris_features = get_cohort_feature_names("hris_only")
    survey_cols = {"enps", "engagement_score", "manager_relationship_score"}
    assert set(hris_features).isdisjoint(survey_cols), (
        f"hris_only feature names contain survey columns: {set(hris_features) & survey_cols}"
    )


def test_get_cohort_feature_names_hybrid_includes_survey() -> None:
    """get_cohort_feature_names('hybrid') must include all survey columns."""
    from retention.features.cohorts import get_cohort_feature_names

    hybrid_features = get_cohort_feature_names("hybrid")
    survey_cols = {"enps", "engagement_score", "manager_relationship_score"}
    assert survey_cols.issubset(set(hybrid_features)), (
        f"hybrid feature names missing survey columns: {survey_cols - set(hybrid_features)}"
    )
