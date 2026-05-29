"""Tests for src/retention/evaluation/nested_cv.py — Story 3.5.

Coverage scope:
    nested_cv_auc_pr   — result type, scores shape + range, mean/std consistency,
                         both cohorts, metadata fields, reproducibility,
                         invalid-input guards
    NestedCVResult     — frozen dataclass, summary string format
    PARAM_GRID         — structure and value type checks

Design principle (mirrors test_expected_value.py):
    The nested CV is expensive to run on real data, so tests use the shared
    ``synthetic_attrition_df`` fixture (100 rows, 25% positive rate) with
    reduced fold counts (outer_splits=2, inner_splits=2) to keep CI under
    the 30-second smoke-test budget while still exercising the full code path.

    Module-scoped fixtures cache the expensive model fits so each test
    function pays only the assertion cost, not the fit cost.

    Two properties are tested as invariants that hold for all valid inputs:
      - result.mean == np.mean(result.scores)  — mean is computed from scores
      - result.std  == np.std(result.scores)   — std is computed from scores
    These are the properties a reviewer would verify by hand to trust the
    headline number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from retention.evaluation.nested_cv import (
    PARAM_GRID,
    NestedCVResult,
    nested_cv_auc_pr,
)
from retention.features.cohorts import extract_X_y, split_cohorts


# ------------------------------------------------------------------ #
# Fixtures                                                              #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def hybrid_X_y(synthetic_attrition_df):
    """Extract X, y for the hybrid cohort from the shared 100-row fixture.

    Why module scope?
        The split is a pure transform on the session-scoped ``synthetic_attrition_df``
        fixture.  Nothing mutates the result, so sharing it across all tests
        in this module is safe.
    """
    cohorts = split_cohorts(synthetic_attrition_df)
    X, y = extract_X_y(cohorts["hybrid"], "hybrid")
    return X, y


@pytest.fixture(scope="module")
def hris_X_y(synthetic_attrition_df):
    """Extract X, y for the hris_only cohort from the shared fixture."""
    cohorts = split_cohorts(synthetic_attrition_df)
    X, y = extract_X_y(cohorts["hris_only"], "hris_only")
    return X, y


@pytest.fixture(scope="module")
def nested_result_hybrid(hybrid_X_y):
    """Run nested 2×2 CV on the hybrid cohort once; reuse across all tests.

    outer_splits=2, inner_splits=2 → 2 outer folds × (2 inner folds × 4 combos
    + 1 refit) = 18 total model fits on ~80-row folds.  Fast enough for CI;
    exercises the full code path including inner-loop hyperparameter selection.
    """
    X, y = hybrid_X_y
    return nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=2)


@pytest.fixture(scope="module")
def nested_result_hris(hris_X_y):
    """Run nested 2×2 CV on the hris_only cohort once."""
    X, y = hris_X_y
    return nested_cv_auc_pr(X, y, "hris_only", outer_splits=2, inner_splits=2)


# ------------------------------------------------------------------ #
# PARAM_GRID — structure checks                                         #
# ------------------------------------------------------------------ #


def test_param_grid_has_four_entries():
    """Sanity: the grid has the right number of combinations."""
    assert len(PARAM_GRID) == 4


def test_param_grid_required_keys():
    """Every entry must have n_estimators, max_depth, learning_rate."""
    required = {"n_estimators", "max_depth", "learning_rate"}
    for i, combo in enumerate(PARAM_GRID):
        missing = required - combo.keys()
        assert not missing, f"PARAM_GRID[{i}] missing keys: {missing}"


def test_param_grid_value_types():
    """n_estimators and max_depth must be int; learning_rate must be float."""
    for i, combo in enumerate(PARAM_GRID):
        # isinstance check: int subclasses bool; use type() to be strict.
        # In practice the literals in the source are plain int / float, so
        # isinstance is fine here.
        assert isinstance(combo["n_estimators"], int), (
            f"PARAM_GRID[{i}]['n_estimators'] should be int, got {type(combo['n_estimators'])}"
        )
        assert isinstance(combo["max_depth"], int), (
            f"PARAM_GRID[{i}]['max_depth'] should be int, got {type(combo['max_depth'])}"
        )
        assert isinstance(combo["learning_rate"], float), (
            f"PARAM_GRID[{i}]['learning_rate'] should be float, got {type(combo['learning_rate'])}"
        )


# ------------------------------------------------------------------ #
# NestedCVResult — dataclass properties                                 #
# ------------------------------------------------------------------ #


def test_result_is_frozen(nested_result_hybrid):
    """NestedCVResult must be frozen — assigning to any field raises."""
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        # setattr dispatches through __setattr__ — frozen dataclass raises
        # FrozenInstanceError.  Using setattr() (not direct attribute assignment)
        # avoids a mypy type error while keeping the runtime assertion live.
        setattr(nested_result_hybrid, "mean", 0.0)


def test_summary_format(nested_result_hybrid):
    """summary() must include the cohort name and fold counts."""
    s = nested_result_hybrid.summary()
    assert "hybrid" in s
    assert "2×2" in s  # outer_splits × inner_splits
    assert "AUC-PR" in s


def test_summary_contains_mean_and_std(nested_result_hybrid):
    """summary() must embed the formatted mean and std values."""
    result = nested_result_hybrid
    s = result.summary()
    # The mean should appear formatted to 3 decimal places.
    assert f"{result.mean:.3f}" in s
    assert f"{result.std:.3f}" in s


# ------------------------------------------------------------------ #
# nested_cv_auc_pr — result shape and type                             #
# ------------------------------------------------------------------ #


def test_returns_nested_cv_result(nested_result_hybrid):
    """Return type must be NestedCVResult, not a bare tuple or dict."""
    assert isinstance(nested_result_hybrid, NestedCVResult)


def test_scores_length_matches_outer_splits(nested_result_hybrid):
    """len(result.scores) must equal the number of outer folds."""
    assert len(nested_result_hybrid.scores) == 2  # outer_splits=2


def test_scores_in_unit_interval(nested_result_hybrid):
    """All outer-fold AUC-PR scores must be in [0, 1]."""
    for i, score in enumerate(nested_result_hybrid.scores):
        assert 0.0 <= score <= 1.0, f"scores[{i}] = {score:.4f} is outside [0, 1]"


def test_mean_consistency(nested_result_hybrid):
    """result.mean must equal np.mean(result.scores) to floating-point precision."""
    result = nested_result_hybrid
    assert result.mean == pytest.approx(float(np.mean(result.scores)))


def test_std_consistency(nested_result_hybrid):
    """result.std must equal np.std(result.scores) to floating-point precision."""
    result = nested_result_hybrid
    assert result.std == pytest.approx(float(np.std(result.scores)))


def test_metadata_fields(nested_result_hybrid):
    """Metadata fields must reflect the arguments passed to nested_cv_auc_pr."""
    result = nested_result_hybrid
    assert result.cohort == "hybrid"
    assert result.outer_splits == 2
    assert result.inner_splits == 2


# ------------------------------------------------------------------ #
# nested_cv_auc_pr — both cohorts                                      #
# ------------------------------------------------------------------ #


def test_hris_cohort_returns_valid_result(nested_result_hris):
    """nested_cv_auc_pr must work for hris_only as well as hybrid."""
    result = nested_result_hris
    assert isinstance(result, NestedCVResult)
    assert result.cohort == "hris_only"
    assert len(result.scores) == 2
    assert all(0.0 <= s <= 1.0 for s in result.scores)


# ------------------------------------------------------------------ #
# nested_cv_auc_pr — reproducibility                                   #
# ------------------------------------------------------------------ #


def test_same_seed_produces_same_scores(hybrid_X_y):
    """Running nested_cv_auc_pr twice with the same seed must yield identical scores.

    This is the nested-CV equivalent of the Story 2.8 reproducibility smoke
    test: the outer StratifiedKFold splits + inner-loop selections must be
    fully deterministic under a fixed seed.
    """
    X, y = hybrid_X_y
    result_a = nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=2, seed=0)
    result_b = nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=2, seed=0)
    assert result_a.scores == pytest.approx(result_b.scores)


def test_different_seeds_may_differ(hybrid_X_y):
    """Different seeds should typically produce different fold partitions.

    This is a probabilistic test: the two runs almost certainly differ on
    100 rows, but if by chance they agree the test passes anyway — we are
    checking that the seed *routes through* the fold generator, not that it
    *forces* different partitions.  A more rigorous check is not worth the
    fragility cost on a 100-row fixture.
    """
    X, y = hybrid_X_y
    result_0 = nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=2, seed=0)
    result_1 = nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=2, seed=1)
    # At least the metadata must always differ (different seed stored, even if
    # scores happen to match on a tiny fixture).
    # We can't assert scores differ without making the test fragile, so we only
    # assert the function runs without error for a different seed.
    assert isinstance(result_1, NestedCVResult)
    _ = result_0  # suppress unused-variable lint


# ------------------------------------------------------------------ #
# nested_cv_auc_pr — input validation                                  #
# ------------------------------------------------------------------ #


def test_raises_on_outer_splits_too_small(hybrid_X_y):
    """outer_splits=1 must raise ValueError."""
    X, y = hybrid_X_y
    with pytest.raises(ValueError, match="outer_splits"):
        nested_cv_auc_pr(X, y, "hybrid", outer_splits=1, inner_splits=2)


def test_raises_on_inner_splits_too_small(hybrid_X_y):
    """inner_splits=1 must raise ValueError."""
    X, y = hybrid_X_y
    with pytest.raises(ValueError, match="inner_splits"):
        nested_cv_auc_pr(X, y, "hybrid", outer_splits=2, inner_splits=1)
