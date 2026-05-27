"""Story 1.2.4 + 1.5.5 Layer 1 — Tests for FEATURE_CATALOG.

Three responsibilities:

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
"""

from __future__ import annotations

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
