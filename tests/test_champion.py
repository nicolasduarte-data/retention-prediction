"""Tests for src/retention/evaluation/champion.py — Story 3.6.

Coverage scope:
    _base_rate_brier   — the β·(1−β) gate-bar formula (the math the whole rule
                         hinges on; tested as a pure function so a mutation to
                         ``*``/``-`` is caught immediately)
    select_champion    — the calibration-gated rank: gate filters BEFORE the
                         AUC-PR sort, boundary is inclusive, honest fallback when
                         no cell clears the bar, input-validation guards
    ChampionSelection  — frozen dataclass, summary() PASS/FAIL rendering
    ChampionArtifact   — frozen dataclass, predict_proba shape normalisation
                         (2-D → 1-D, 1-D passthrough), threshold decision rule,
                         summary() provenance block
    persist / load     — pickle roundtrip, parent-dir creation, return value,
                         missing-file and wrong-type guards

Design principles (mirrors test_nested_cv.py / test_expected_value.py):
    * No real data or model fits are needed — ``select_champion`` is a pure
      function of a metrics table, so tests build small ``summary`` DataFrames
      inline, and the artifact tests wrap tiny *module-level* fake estimators.
    * The fakes live at module scope on purpose: pickle serialises a class by
      *reference* (its import path), so a class defined inside a test function
      cannot be unpickled — the persist/load roundtrip would fail. Module scope
      is the fix.
    * Gate-logic tests are written as the properties a reviewer checks by hand
      to trust the headline pick: "the higher-AUC-PR cell that *fails* the gate
      is correctly skipped" is the single most important assertion in the file,
      because it proves selection is a gated rank and not a sort.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from retention.evaluation.champion import (
    REQUIRED_SUMMARY_COLUMNS,
    ChampionArtifact,
    ChampionSelection,
    SupportsPredictProba,
    _base_rate_brier,
    load_champion,
    persist_champion,
    select_champion,
)

# (model, cohort, auc_pr, brier, ece, threshold) — one summary-table row.
_SummaryRow = tuple[str, str, float, float, float, float]


# ------------------------------------------------------------------ #
# Module-level fakes (picklable) + small builders                      #
# ------------------------------------------------------------------ #


class _FakeProbaModel:
    """Fake estimator returning the sklearn ``(n, 2)`` probability matrix.

    Stores a positive-class vector and reconstructs ``[1−p, p]`` columns, so it
    exercises the ChampionArtifact shape-normalisation path (2-D → column 1).
    Ignores ``X`` entirely — the tests control the output directly.

    Helpers are fully annotated (not relying on the ``tests.*`` untyped-def
    relaxation) because strict mode's ``disallow_untyped_calls`` still flags
    calls to *untyped* helpers from the checked test bodies. The annotated
    ``predict_proba`` also makes the structural match to ``SupportsPredictProba``
    explicit.
    """

    def __init__(self, positive_proba: Sequence[float]) -> None:
        self.positive_proba = np.asarray(positive_proba, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        p = self.positive_proba
        return np.column_stack([1.0 - p, p])


class _Fake1DProbaModel:
    """Fake estimator returning a *1-D* positive-class vector.

    Exercises the passthrough branch of ChampionArtifact.predict_proba — some
    estimators (and Loop 4 wrappers) already hand back a 1-D P(exit) vector, and
    the artifact must not index ``[:, 1]`` into it.
    """

    def __init__(self, positive_proba: Sequence[float]) -> None:
        self.positive_proba = np.asarray(positive_proba, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        return self.positive_proba


def _make_summary(rows: Sequence[_SummaryRow]) -> pd.DataFrame:
    """Build a summary table from (model, cohort, auc_pr, brier, ece, threshold) tuples."""
    return pd.DataFrame(list(rows), columns=list(REQUIRED_SUMMARY_COLUMNS))


def _make_selection(*, passed_gate: bool = True) -> ChampionSelection:
    """A minimal valid ChampionSelection for embedding in artifact tests."""
    return ChampionSelection(
        model_name="gbm",
        cohort="hybrid",
        auc_pr=0.27,
        brier=0.158,
        ece=0.05,
        threshold=0.40,
        criterion="f2",
        passed_calibration_gate=passed_gate,
        base_rate_brier=0.16,
        rationale="test selection",
    )


def _make_artifact(model: SupportsPredictProba, *, threshold: float = 0.5) -> ChampionArtifact:
    """Wrap ``model`` in a ChampionArtifact with otherwise-fixed provenance."""
    return ChampionArtifact(
        model=model,
        model_name="gbm",
        cohort="hybrid",
        threshold=threshold,
        feature_names=["tenure_months", "enps"],
        seed=42,
        selection=_make_selection(),
        val_metrics={"auc_pr": 0.27, "brier": 0.158},
        test_metrics={"auc_pr": 0.25, "brier": 0.161},
        created_utc="2026-05-29T00:00:00Z",
    )


# ------------------------------------------------------------------ #
# _base_rate_brier — the gate-bar formula                              #
# ------------------------------------------------------------------ #


def test_base_rate_brier_at_half_is_quarter():
    """β = 0.5 is the maximum-variance Bernoulli → Brier 0.25."""
    assert _base_rate_brier(0.5) == pytest.approx(0.25)


def test_base_rate_brier_known_value():
    """β = 0.2 → 0.2 · 0.8 = 0.16 (kills ``*``→``+`` and ``1−``→``1+`` mutants)."""
    assert _base_rate_brier(0.2) == pytest.approx(0.16)


def test_base_rate_brier_is_symmetric():
    """β·(1−β) is symmetric about 0.5: β and 1−β give the same bar."""
    assert _base_rate_brier(0.1) == pytest.approx(_base_rate_brier(0.9))


# ------------------------------------------------------------------ #
# select_champion — the gated rank                                     #
# ------------------------------------------------------------------ #


def test_gate_filters_before_rank():
    """THE core property: a higher-AUC-PR cell that FAILS the gate is skipped.

    base_rate 0.2 → gate 0.16. ``lr`` has the highest AUC-PR (0.30) but a Brier
    of 0.235 (fails); ``gbm`` clears the gate (0.158) at a lower AUC-PR. The
    champion must be ``gbm`` — proving selection gates first, then ranks.
    """
    summary = _make_summary(
        [
            ("lr", "hybrid", 0.30, 0.235, 0.12, 0.30),
            ("gbm", "hybrid", 0.27, 0.158, 0.05, 0.40),
            ("ebm", "hybrid", 0.25, 0.240, 0.14, 0.35),
        ]
    )
    result = select_champion(summary, base_rate=0.2)
    assert result.model_name == "gbm"
    assert result.passed_calibration_gate is True


def test_highest_auc_pr_among_multiple_eligible():
    """When several cells clear the gate, the highest AUC-PR among them wins."""
    summary = _make_summary(
        [
            ("gbm", "hybrid", 0.27, 0.150, 0.05, 0.40),
            ("lr", "hris_only", 0.29, 0.155, 0.06, 0.45),  # eligible AND higher
            ("ebm", "hybrid", 0.25, 0.240, 0.14, 0.35),  # fails gate
        ]
    )
    result = select_champion(summary, base_rate=0.2)
    assert result.model_name == "lr"
    assert result.cohort == "hris_only"
    assert result.passed_calibration_gate is True


def test_gate_boundary_is_inclusive():
    """A Brier exactly equal to the gate is eligible (``<=``, not ``<``).

    Kills the ``<=``→``<`` mutant: with brier == gate == 0.16, the cell must be
    treated as passing.
    """
    summary = _make_summary(
        [
            ("gbm", "hybrid", 0.27, 0.16, 0.05, 0.40),  # brier == gate
            ("lr", "hybrid", 0.30, 0.50, 0.20, 0.30),  # fails badly
        ]
    )
    result = select_champion(summary, base_rate=0.2)
    assert result.model_name == "gbm"
    assert result.passed_calibration_gate is True


def test_fallback_when_no_cell_clears_gate():
    """If every Brier is above the gate, pick highest AUC-PR overall, flagged."""
    summary = _make_summary(
        [
            ("lr", "hybrid", 0.30, 0.235, 0.12, 0.30),  # highest AUC-PR
            ("gbm", "hybrid", 0.27, 0.180, 0.05, 0.40),
            ("ebm", "hybrid", 0.25, 0.240, 0.14, 0.35),
        ]
    )
    result = select_champion(summary, base_rate=0.2)
    assert result.model_name == "lr"
    assert result.passed_calibration_gate is False
    assert "calibrate" in result.rationale.lower()


def test_metadata_reflects_winning_cell():
    """Every scalar field on the result is copied from the chosen row."""
    summary = _make_summary(
        [
            ("gbm", "hybrid", 0.271, 0.158, 0.051, 0.42),
            ("ebm", "hybrid", 0.240, 0.240, 0.140, 0.35),
        ]
    )
    result = select_champion(summary, base_rate=0.2)
    assert result.model_name == "gbm"
    assert result.cohort == "hybrid"
    assert result.auc_pr == pytest.approx(0.271)
    assert result.brier == pytest.approx(0.158)
    assert result.ece == pytest.approx(0.051)
    assert result.threshold == pytest.approx(0.42)
    assert result.base_rate_brier == pytest.approx(0.16)


def test_criterion_is_stored():
    """The criterion arg is carried onto the result for provenance."""
    summary = _make_summary([("gbm", "hybrid", 0.27, 0.158, 0.05, 0.40)])
    assert select_champion(summary, base_rate=0.2).criterion == "f2"  # default
    assert select_champion(summary, base_rate=0.2, criterion="f1").criterion == "f1"


def test_raises_on_missing_column():
    """Dropping a required column raises ValueError naming the contract."""
    summary = _make_summary([("gbm", "hybrid", 0.27, 0.158, 0.05, 0.40)]).drop(columns=["brier"])
    with pytest.raises(ValueError, match="missing required column"):
        select_champion(summary, base_rate=0.2)


def test_raises_on_empty_summary():
    """An all-columns-but-no-rows table raises ValueError."""
    empty = pd.DataFrame(columns=list(REQUIRED_SUMMARY_COLUMNS))
    with pytest.raises(ValueError, match="no rows"):
        select_champion(empty, base_rate=0.2)


@pytest.mark.parametrize("bad_base_rate", [0.0, 1.0, -0.1, 1.5])
def test_raises_on_base_rate_out_of_range(bad_base_rate):
    """base_rate must be strictly inside (0, 1) — boundaries included raise.

    Testing 0.0 and 1.0 kills a ``<``→``<=`` mutant on the bounds check.
    """
    summary = _make_summary([("gbm", "hybrid", 0.27, 0.158, 0.05, 0.40)])
    with pytest.raises(ValueError, match="base_rate"):
        select_champion(summary, base_rate=bad_base_rate)


# ------------------------------------------------------------------ #
# ChampionSelection — dataclass behaviour                              #
# ------------------------------------------------------------------ #


def test_selection_is_frozen():
    """ChampionSelection is frozen — assignment raises."""
    selection = _make_selection()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        setattr(selection, "auc_pr", 0.0)


def test_selection_summary_shows_pass():
    """summary() renders the gate as PASS when it was cleared."""
    s = _make_selection(passed_gate=True).summary()
    assert "gbm" in s
    assert "hybrid" in s
    assert "PASS" in s


def test_selection_summary_shows_fail():
    """summary() renders the gate as FAIL on the fallback path."""
    assert "FAIL" in _make_selection(passed_gate=False).summary()


# ------------------------------------------------------------------ #
# ChampionArtifact — predict + provenance                              #
# ------------------------------------------------------------------ #


def test_artifact_is_frozen():
    """ChampionArtifact is frozen — a shipped artifact is a fixed point."""
    artifact = _make_artifact(_FakeProbaModel([0.5]))
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        setattr(artifact, "threshold", 0.99)


def test_predict_proba_reduces_2d_to_positive_column():
    """A 2-D (n, 2) matrix is reduced to column 1 (P(exit)) as a 1-D array."""
    artifact = _make_artifact(_FakeProbaModel([0.1, 0.6, 0.9]))
    X = pd.DataFrame({"tenure_months": [1, 2, 3], "enps": [4, 5, 6]})
    proba = artifact.predict_proba(X)
    assert proba.shape == (3,)
    assert proba == pytest.approx([0.1, 0.6, 0.9])


def test_predict_proba_passes_through_1d():
    """A 1-D positive-class vector passes through unindexed."""
    artifact = _make_artifact(_Fake1DProbaModel([0.2, 0.8]))
    X = pd.DataFrame({"tenure_months": [1, 2], "enps": [3, 4]})
    proba = artifact.predict_proba(X)
    assert proba.shape == (2,)
    assert proba == pytest.approx([0.2, 0.8])


def test_predict_applies_threshold():
    """predict() flags exactly the rows with P(exit) ≥ threshold."""
    artifact = _make_artifact(_FakeProbaModel([0.1, 0.6, 0.9]), threshold=0.5)
    X = pd.DataFrame({"tenure_months": [1, 2, 3], "enps": [4, 5, 6]})
    assert artifact.predict(X).tolist() == [0, 1, 1]


def test_predict_threshold_is_inclusive():
    """A probability exactly at the threshold is flagged (``>=``)."""
    artifact = _make_artifact(_FakeProbaModel([0.5]), threshold=0.5)
    X = pd.DataFrame({"tenure_months": [1], "enps": [2]})
    assert artifact.predict(X).tolist() == [1]


def test_predict_returns_integers():
    """predict() returns an integer array (np int width is platform-dependent)."""
    artifact = _make_artifact(_FakeProbaModel([0.1, 0.9]), threshold=0.5)
    X = pd.DataFrame({"tenure_months": [1, 2], "enps": [3, 4]})
    assert np.issubdtype(artifact.predict(X).dtype, np.integer)


def test_artifact_summary_contains_provenance():
    """summary() shows the model, both metric splits, and the feature count."""
    s = _make_artifact(_FakeProbaModel([0.5])).summary()
    assert "gbm" in s
    assert "validation" in s
    assert "test" in s
    assert "2" in s  # two feature names


# ------------------------------------------------------------------ #
# persist_champion / load_champion — pickle roundtrip                  #
# ------------------------------------------------------------------ #


def test_persist_returns_written_path(tmp_path):
    """persist_champion returns the path it wrote (for logging/echo)."""
    artifact = _make_artifact(_FakeProbaModel([0.5]))
    path = tmp_path / "champion.pkl"
    assert persist_champion(artifact, path) == path
    assert path.exists()


def test_persist_creates_parent_dirs(tmp_path):
    """Missing parent directories are created (reports/models/ won't exist yet)."""
    artifact = _make_artifact(_FakeProbaModel([0.5]))
    path = tmp_path / "nested" / "deeper" / "champion.pkl"
    persist_champion(artifact, path)
    assert path.exists()


def test_persist_load_roundtrip(tmp_path):
    """A persisted champion loads back with provenance and a working model."""
    artifact = _make_artifact(_FakeProbaModel([0.1, 0.6, 0.9]), threshold=0.4)
    path = tmp_path / "champion.pkl"
    persist_champion(artifact, path)

    loaded = load_champion(path)
    assert loaded.model_name == artifact.model_name
    assert loaded.cohort == artifact.cohort
    assert loaded.threshold == pytest.approx(artifact.threshold)
    assert loaded.feature_names == artifact.feature_names
    assert loaded.seed == artifact.seed
    assert loaded.val_metrics == artifact.val_metrics
    assert loaded.test_metrics == artifact.test_metrics
    assert loaded.selection.model_name == artifact.selection.model_name

    # The wrapped estimator survives the roundtrip and still predicts.
    X = pd.DataFrame({"tenure_months": [1, 2, 3], "enps": [4, 5, 6]})
    assert loaded.predict_proba(X) == pytest.approx([0.1, 0.6, 0.9])
    assert loaded.predict(X).tolist() == [0, 1, 1]  # threshold 0.4


def test_load_missing_file_raises(tmp_path):
    """Loading an absent path raises FileNotFoundError with a how-to hint."""
    with pytest.raises(FileNotFoundError, match="No champion"):
        load_champion(tmp_path / "absent.pkl")


def test_load_wrong_type_raises(tmp_path):
    """A pickle that isn't a ChampionArtifact raises TypeError."""
    path = tmp_path / "champion.pkl"
    with path.open("wb") as fh:
        pickle.dump({"not": "an artifact"}, fh)
    with pytest.raises(TypeError, match="ChampionArtifact"):
        load_champion(path)
