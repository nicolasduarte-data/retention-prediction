"""Story 2.6.5 — End-to-end pipeline smoke test.

Verifies that the full Loop 2 pipeline executes without error from a raw
DataFrame all the way through to predicted probabilities:

    synthetic DataFrame
        → temporal_split()
        → split_cohorts()
        → extract_X_y()
        → train_lr()               (both cohorts)
        → pipeline.predict_proba()
        → metric assertions

Why LR only (not GBM or EBM)?

    EBM trains in ~4s even on 30 rows (see test_models.py: 26s for 16 tests).
    On the 70-row training split EBM would consume ~8s × 2 cohorts = 16s,
    leaving only 14s for LR + GBM before the 30s CI budget is exceeded. LR
    covers the pipeline contract: if split_cohorts, extract_X_y, and
    predict_proba work for LR they work for all models — the pipeline is
    model-agnostic. Model-specific correctness (GBM, EBM) is covered by
    test_models.py (Stories 2.6.1–2.6.4).

Why save_indices=False?

    temporal_split() by default persists a splits.parquet file to
    data/processed/. In tests we never want that side-effect — it would
    overwrite the production split record and create a non-reproducible
    fixture dependency. Always pass save_indices=False in test code.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from retention.data.split import temporal_split
from retention.features.cohorts import extract_X_y, split_cohorts, get_cohort_feature_names
from retention.models.lr import train_lr


# ------------------------------------------------------------------ #
# Story 2.6.5 — Pipeline smoke test                                    #
# ------------------------------------------------------------------ #


def test_pipeline_smoke(
    synthetic_attrition_df: pd.DataFrame,
) -> None:
    """Full pipeline from raw DataFrame to LR predictions runs in < 30s.

    This is the integration contract: every layer of the pipeline must
    compose without error. If any link breaks (schema drift, split
    boundary, extract_X_y regression, pipeline fit failure), this test
    catches it before the notebook does.

    Assertions (per cohort):
      - Predicted probability array has shape (n_val_rows, 2).
      - All predicted probabilities are in [0, 1].
      - Row probabilities sum to 1.0 within floating-point tolerance.
      - Validation feature columns match training feature columns (no drift).
    """
    t_start = time.monotonic()

    df = synthetic_attrition_df

    # ── Step 1: Temporal split ────────────────────────────────────────
    # save_indices=False: don't write splits.parquet during tests.
    # With 100 rows and 70/15/15 defaults: train=70, val=15, test=15.
    train_df, val_df, test_df = temporal_split(df, save_indices=False)

    # Shape sanity — confirm split sizes are as expected for 100 rows
    assert len(train_df) == 70, f"Expected 70 train rows, got {len(train_df)}"
    assert len(val_df) == 15, f"Expected 15 val rows, got {len(val_df)}"
    assert len(test_df) == 15, f"Expected 15 test rows, got {len(test_df)}"

    # ── Step 2: Cohort split ──────────────────────────────────────────
    # split_cohorts returns two views: hris_only (10 cols) and hybrid (13 cols).
    # Both views have IDENTICAL row indices — same employees, different columns.
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    assert set(train_cohorts.keys()) == {"hris_only", "hybrid"}, (
        "split_cohorts must return keys 'hris_only' and 'hybrid'"
    )
    # Identical index invariant — the core dual-cohort guarantee
    assert list(train_cohorts["hris_only"].index) == list(train_cohorts["hybrid"].index), (
        "hris_only and hybrid cohorts must have identical row indices"
    )

    # ── Step 3: Train LR on each cohort and evaluate on validation ───
    for cohort in ("hris_only", "hybrid"):
        # extract_X_y separates feature matrix from label — tested in detail
        # by test_no_leakage_at_predict_time in test_data_quality.py.
        X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)
        X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)

        # Feature count sanity: catalog defines 7 hris_only features, 10 hybrid.
        expected_n_features = len(get_cohort_feature_names(cohort))
        assert X_tr.shape[1] == expected_n_features, (
            f"[{cohort}] Expected {expected_n_features} features, got {X_tr.shape[1]}"
        )

        # Column consistency: same features at train and predict time.
        # A mismatch here would cause a silent leakage or KeyError in production.
        assert list(X_val.columns) == list(X_tr.columns), (
            f"[{cohort}] Train and val feature columns differ — schema drift."
        )

        # train_lr returns a fitted sklearn Pipeline (preprocessor → LogisticRegression)
        pipeline = train_lr(X_tr, y_tr, cohort=cohort)

        # ── Step 4: Predict on validation set ────────────────────────
        proba = pipeline.predict_proba(X_val)

        # Shape: (n_val_rows, 2) — one column per class
        assert proba.shape == (len(X_val), 2), (
            f"[{cohort}] predict_proba shape mismatch: expected ({len(X_val)}, 2), "
            f"got {proba.shape}"
        )

        # All values in [0, 1] — foundational probability axiom
        assert (proba >= 0.0).all(), (
            f"[{cohort}] Negative probability value encountered: min={proba.min():.6f}"
        )
        assert (proba <= 1.0).all(), (
            f"[{cohort}] Probability > 1.0 encountered: max={proba.max():.6f}"
        )

        # Each row's P(0) + P(1) = 1.0 — probability mass conservation
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-6,
            err_msg=f"[{cohort}] Row probabilities do not sum to 1.0",
        )

    # ── Step 5: Timing guard ──────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    assert elapsed < 30.0, (
        f"Pipeline smoke test took {elapsed:.1f}s — exceeds the 30s CI budget. "
        "If EBM was accidentally added to this test, remove it (it belongs in "
        "test_models.py, not the smoke test). If LR is genuinely slow, profile "
        "the preprocessor for unexpected complexity."
    )


def test_pipeline_smoke_no_data_leaks_across_splits(
    synthetic_attrition_df: pd.DataFrame,
) -> None:
    """Employee indices must be disjoint across train / val / test splits.

    This is the temporal integrity invariant: an employee who appears in the
    training set must not appear in validation or test (and vice versa). If
    this breaks, eval metrics become optimistic because the model has already
    'seen' those employees during training.

    Implementation note: the fixture uses unique employee_ids (EMP001–EMP100)
    so index overlap directly implies employee overlap.
    """
    train_df, val_df, test_df = temporal_split(synthetic_attrition_df, save_indices=False)

    train_ids = set(train_df["employee_id"])
    val_ids = set(val_df["employee_id"])
    test_ids = set(test_df["employee_id"])

    assert train_ids.isdisjoint(val_ids), (
        f"Employee overlap between train and val: {train_ids & val_ids}"
    )
    assert train_ids.isdisjoint(test_ids), (
        f"Employee overlap between train and test: {train_ids & test_ids}"
    )
    assert val_ids.isdisjoint(test_ids), (
        f"Employee overlap between val and test: {val_ids & test_ids}"
    )
    # All 100 employees accounted for — no employee dropped silently
    assert len(train_ids | val_ids | test_ids) == len(synthetic_attrition_df), (
        "Not all employees appear in exactly one split — possible data loss in temporal_split."
    )
