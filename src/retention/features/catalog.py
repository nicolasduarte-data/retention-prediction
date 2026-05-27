"""Feature catalog — single source of truth for what each column in
`marts.v_attrition_features` is and how it's allowed to be used.

This module solves three problems simultaneously:

1. **Schema drift detection** — `FEATURE_CATALOG` lists every column the
   project knows about. `test_features.py::test_catalog_covers_all_contract_columns`
   asserts the catalog is a superset of `docs/integration_contract.md`. Add a
   new mart column without a catalog entry → test fails → forced documentation.

2. **Leakage discipline (Story 1.5 — Risk 3)** — `leakage_audited=True` is the
   gate that lets a feature into the training set. The `leakage_rationale`
   must cite the pa-warehouse dbt model path so a reviewer can verify the
   feature is computed BEFORE the label event. Generic "looks safe" rationales
   are rejected by `test_leakage_rationales_cite_dbt_models`.

3. **DiCE actionability + fairness audit metadata (Loops 3a + 3c)** — the
   `mutable` flag tells DiCE which features it's allowed to perturb when
   generating counterfactuals (`age` is immutable, `compa_ratio` is mutable).
   The `protected` flag tells Fairlearn which dimensions matter for the
   audit (`gender`, `age_at_window_close`).

Story 1.2 — see `BACKLOG.md → ## ⚠️ Risk Register` rows 2 + 3.

**v0.1 cohort note**: every spec has `cohort='both'` because the mart
currently has zero survey columns. Loop 2 adds survey features when
paw-prey-NNN delivers them — see `BACKLOG.md → ## Preconditions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------- #
# Dataclass                                                              #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single column in `marts.v_attrition_features`.

    Every column the project consumes — features, identifiers, the temporal
    anchor, and the label — gets a FeatureSpec. Identifiers and the label
    have role-distinct values that downstream helpers filter on so model
    training pipelines never accidentally include them as features.
    """

    name: str
    """Column name as it appears in the mart and the loaded DataFrame."""

    role: Literal["feature", "identifier", "temporal_anchor", "label"]
    """How this column is used downstream:
       - 'feature'          — fed to the model
       - 'identifier'       — primary key (employee_id), dropped before training
       - 'temporal_anchor'  — used for temporal split + leakage checks (snapshot_date)
       - 'label'            — prediction target (voluntary_exit_label)
    """

    dtype: Literal["numeric", "categorical", "boolean", "date", "string"]
    """Logical dtype family. Maps to pandas dtype via the preprocessing pipeline."""

    source: Literal["HRIS", "survey", "derived", "identifier", "label", "temporal"]
    """Provenance category. Drives the dual-cohort split in Loop 2."""

    mutable: bool
    """Whether DiCE is allowed to perturb this feature when generating
    counterfactuals. False for protected attributes (age, gender) and identifiers."""

    protected: bool
    """Whether the fairness audit treats this as a protected attribute
    (Loop 3a — Fairlearn audit dimensions)."""

    cohort: Literal["both", "hybrid_only"]
    """Which cohort this column belongs to:
       - 'both'         — appears in both HRIS-only and hybrid cohorts
       - 'hybrid_only'  — survey-derived; only in the hybrid cohort

    Note: All v0.1 specs are 'both' because the mart has no survey columns
    yet. Loop 2 introduces 'hybrid_only' when paw-prey-NNN delivers survey
    signal columns. See BACKLOG.md → ## Preconditions.
    """

    snapshot_date_offset_months: int
    """Months BEFORE observation_window_end at which this feature value is
    measured. 0 = measured AT window end. Positive = measured N months
    earlier. Used by `tests/test_no_leakage.py::test_feature_snapshot_before_label`
    (Story 1.5.5 Layer 2) to verify no temporal leak.

    For non-feature roles (identifier, label, temporal_anchor), use 0.
    """

    leakage_audited: bool
    """Story 1.5.5 — MUST be True before this feature appears in any training
    set. False → Layer 1 leakage gate fails. Non-feature roles default True
    (no leakage possible)."""

    leakage_rationale: str
    """Specific reason this feature is safe — cites pa-warehouse dbt model
    paths or `_core.yml` metadata. Generic rationales like 'looks safe' fail
    the Story 1.5.5 Layer 1 test (regex check for 'dbt/models/' or '_core.yml').

    For non-feature roles, can be a one-line description of the column's role.
    """

    description: str
    """Human-readable description for docs / Data Card / Model Card."""


# --------------------------------------------------------------------- #
# Canonical catalog (10 columns matching docs/integration_contract.md)   #
# --------------------------------------------------------------------- #
#
# Source: pa-warehouse `marts.v_attrition_features` v0.1 (10 columns).
# Update this catalog whenever the integration contract is regenerated.
# `test_catalog_covers_all_contract_columns` enforces parity.

FEATURE_CATALOG: tuple[FeatureSpec, ...] = (
    # ---- Identifier ----
    FeatureSpec(
        name="employee_id",
        role="identifier",
        dtype="string",
        source="identifier",
        mutable=False,
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Primary key — pa-warehouse dbt/models/core/dim_employee.sql. "
            "Dropped before model training; never fed as a feature."
        ),
        description="Synthetic employee primary key (EMP_NNNNN format).",
    ),
    # ---- Temporal anchor ----
    FeatureSpec(
        name="snapshot_date",
        role="temporal_anchor",
        dtype="date",
        source="temporal",
        mutable=False,
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Defines the as-of point for all feature values — pa-warehouse "
            "dbt/models/marts/v_attrition_features.sql. Used by temporal_split() "
            "to enforce no-future-leak; never fed as a feature."
        ),
        description="Observation-window-end timestamp; the as-of date for all features.",
    ),
    # ---- HRIS features ----
    FeatureSpec(
        name="tenure_months",
        role="feature",
        dtype="numeric",
        source="HRIS",
        mutable=False,  # Tenure advances naturally with time — not a DiCE knob.
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Computed in pa-warehouse dbt/models/core/fct_workforce_events.sql "
            "as months between hire_date and snapshot_date. Both anchors are "
            "set BEFORE label_event_date by mart construction (snapshot_date < "
            "voluntary_exit_date when label=True)."
        ),
        description="Months of continuous service at snapshot_date.",
    ),
    FeatureSpec(
        name="performance_tier",
        role="feature",
        dtype="categorical",
        source="HRIS",
        mutable=True,  # Performance can change with intervention — DiCE-actionable.
        protected=False,
        cohort="both",
        snapshot_date_offset_months=3,  # Quarterly review cadence
        leakage_audited=True,
        leakage_rationale=(
            "Sourced from pa-warehouse dbt/models/core/dim_performance.sql, "
            "tier as of latest review BEFORE snapshot_date. Quarterly review "
            "cadence means offset_months=3 worst-case."
        ),
        description="Performance review tier (categorical: typically 1-5 or A-F).",
    ),
    FeatureSpec(
        name="compa_ratio",
        role="feature",
        dtype="numeric",
        source="HRIS",
        mutable=True,  # Comp adjustments are a primary retention lever.
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Computed in pa-warehouse dbt/models/core/dim_compensation.sql as "
            "current_salary / midpoint_for_grade. snapshot_date controls "
            "current_salary; both pre-date label_event_date."
        ),
        description="Current salary divided by band midpoint (1.0 = at midpoint).",
    ),
    FeatureSpec(
        name="is_critical_role",
        role="feature",
        dtype="boolean",
        source="HRIS",
        mutable=False,  # Role criticality is a structural HR call, not an
        # intervention DiCE should suggest.
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Flag from pa-warehouse dbt/models/core/dim_role.sql / _core.yml "
            "role_taxonomy. Set at role creation; updates predate snapshot_date "
            "by definition of the role catalog."
        ),
        description="True if the role is flagged as business-critical / hard-to-fill.",
    ),
    FeatureSpec(
        name="successor_count",
        role="feature",
        dtype="numeric",
        source="HRIS",
        mutable=False,  # Driven by org-design, not employee-level intervention.
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Sourced from pa-warehouse dbt/models/core/dim_succession.sql — "
            "number of ready-now successors flagged as of snapshot_date in the "
            "succession planning module. snapshot_date precedes label_event_date."
        ),
        description="Number of ready-now successors flagged in succession planning.",
    ),
    # ---- Protected attributes (immutable, audit dimensions) ----
    FeatureSpec(
        name="age_at_window_close",
        role="feature",
        dtype="numeric",
        source="HRIS",
        mutable=False,  # Age advances naturally; immutable for DiCE.
        protected=True,  # Loop 3a fairness audit dimension.
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Computed in pa-warehouse dbt/models/core/dim_employee.sql as "
            "DATE_DIFF(snapshot_date, birth_date, YEAR). Added by paw-prey-005 "
            "(2026-05-22) specifically to support the Loop 3a/3c compound "
            "fairness audit. snapshot_date precedes label_event_date."
        ),
        description="Age in years at observation window close.",
    ),
    FeatureSpec(
        name="gender",
        role="feature",
        dtype="categorical",
        source="HRIS",
        mutable=False,  # Immutable protected attribute.
        protected=True,  # Loop 3a fairness audit dimension.
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Sourced from pa-warehouse dbt/models/core/dim_employee.sql self-"
            "reported field. Recorded at hire (paw-prey-005 ensured the column "
            "is exposed in v_attrition_features). Precedes all label events."
        ),
        description="Self-reported gender category (synthetic).",
    ),
    # ---- Label ----
    FeatureSpec(
        name="voluntary_exit_label",
        role="label",
        dtype="boolean",
        source="label",
        mutable=False,
        protected=False,
        cohort="both",
        snapshot_date_offset_months=0,
        leakage_audited=True,
        leakage_rationale=(
            "Prediction target — pa-warehouse dbt/models/marts/v_attrition_features.sql. "
            "True if employee voluntarily exited within the observation window "
            "following snapshot_date. Never used as a feature; dropped from X "
            "before model.fit()."
        ),
        description="True if the employee voluntarily exited the company during the observation window.",
    ),
)


# --------------------------------------------------------------------- #
# Helper functions                                                       #
# --------------------------------------------------------------------- #


def get_spec(name: str) -> FeatureSpec:
    """Return the FeatureSpec for a column name. Raises KeyError if absent."""
    for spec in FEATURE_CATALOG:
        if spec.name == name:
            return spec
    raise KeyError(
        f"No FeatureSpec for column '{name}'. "
        f"Either add a spec to FEATURE_CATALOG or check for a typo. "
        f"Known columns: {sorted(s.name for s in FEATURE_CATALOG)}"
    )


def get_feature_names(cohort: Literal["both", "hybrid_only"] = "both") -> list[str]:
    """Return the column names model.fit() should receive (role='feature').

    Filters by cohort: 'both' returns features available in both cohorts;
    'hybrid_only' returns ALL features (both + hybrid-exclusive, since the
    hybrid cohort is the superset).
    """
    if cohort == "both":
        return [s.name for s in FEATURE_CATALOG if s.role == "feature" and s.cohort == "both"]
    # hybrid cohort = both + hybrid_only (superset)
    return [s.name for s in FEATURE_CATALOG if s.role == "feature"]


def get_mutable_features() -> list[str]:
    """Features DiCE is allowed to perturb (Loop 3c counterfactual constraint)."""
    return [s.name for s in FEATURE_CATALOG if s.role == "feature" and s.mutable]


def get_protected_features() -> list[str]:
    """Protected attributes for the Loop 3a fairness audit."""
    return [s.name for s in FEATURE_CATALOG if s.protected]


def get_identifier_names() -> list[str]:
    """Identifier columns — drop before training."""
    return [s.name for s in FEATURE_CATALOG if s.role == "identifier"]


def get_label_name() -> str:
    """The prediction target column name."""
    labels = [s.name for s in FEATURE_CATALOG if s.role == "label"]
    if len(labels) != 1:
        raise ValueError(f"FEATURE_CATALOG must define exactly one label column; found {labels}")
    return labels[0]


def get_temporal_anchor() -> str:
    """The column used by `temporal_split` to enforce no-future-leak."""
    anchors = [s.name for s in FEATURE_CATALOG if s.role == "temporal_anchor"]
    if len(anchors) != 1:
        raise ValueError(
            f"FEATURE_CATALOG must define exactly one temporal_anchor column; found {anchors}"
        )
    return anchors[0]
