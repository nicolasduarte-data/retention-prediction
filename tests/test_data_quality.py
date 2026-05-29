"""Stories 2.6.6 + 2.6.7 — Prediction calibration sanity + leakage check.

Two classes of production failure this file guards against:

    2.6.6  test_prediction_calibration_sanity
        WHAT: GBM's predicted probabilities must be finite, in [0, 1],
              and non-degenerate (not all the same value).
        WHY:  A degenerate model — one that predicts 0.0 for every
              employee, or NaN after a preprocessing edge case — would
              pass test_models.py (which fits on 30 rows and only checks
              shape/sum). This test uses the 100-row integration fixture
              and validates the output distribution, not just the shape.

    2.6.7  test_no_leakage_at_predict_time
        WHAT: The label column and non-feature identifier columns
              (employee_id, snapshot_date) must NOT appear in the
              feature matrix X at predict time.
        WHY:  Target leakage is the #1 silent validity killer in
              people analytics models. If extract_X_y() accidentally
              returns the label in X, the model trains on the answer —
              AUC-PR looks perfect, but the model is useless in
              production where the label is unknown.

Both tests use the session-scoped `synthetic_attrition_df` fixture from
conftest.py — identical rows for both, so failures are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention.data.split import temporal_split
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.models.xgb import RetentionModel


# ------------------------------------------------------------------ #
# Story 2.6.6 — Calibration sanity                                     #
# ------------------------------------------------------------------ #


def test_prediction_calibration_sanity(
    synthetic_attrition_df: pd.DataFrame,
) -> None:
    """GBM's validation-set predictions must be finite, bounded, and non-constant.

    This is a *sanity* check, not a statistical calibration test. It catches
    four catastrophic failure modes:

        1. NaN / Inf output — preprocessing edge case (e.g. all-zero column
           fed to a log transform, or missing imputer for an unexpected dtype).
        2. Probability outside [0, 1] — shouldn't happen with XGBoost's
           logistic output, but guards against future objective changes.
        3. Constant output — all predictions identical, meaning the model
           has collapsed and ignores all features.
        4. Mean completely outside reasonable range — model is systematically
           predicting near-0 or near-1 for every employee.

    Why GBM specifically?

        GBM (XGBoost) is the best-calibrated of the three Loop 2 models —
        Brier score 0.158 on the real data, vs 0.236 for LR and EBM (which
        use balanced class weights that inflate probabilities). GBM's output
        is closest to a true probability, so the sanity thresholds are tighter
        without being brittle.

    Why not check calibration against the base rate?

        Statistical calibration (mean(proba) ≈ base_rate) requires > 1,000
        samples for a stable estimate. On 15 validation rows this test would
        be noisy and prone to false positives. Quantitative calibration testing
        uses the real 191-row validation set in the comparison notebook. This
        test only catches degenerate outputs.
    """
    df = synthetic_attrition_df

    # Temporal split — save_indices=False avoids writing splits.parquet in tests
    train_df, val_df, _ = temporal_split(df, save_indices=False)

    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    # Use the hybrid cohort — the most feature-rich, most likely to surface
    # preprocessing issues (survey imputation, mixed-type columns, etc.)
    X_tr, y_tr = extract_X_y(train_cohorts["hybrid"], "hybrid")
    X_val, _ = extract_X_y(val_cohorts["hybrid"], "hybrid")

    # Fit the GBM — uses default hyperparams matching the comparison notebook.
    # cohort='hybrid' controls which feature columns the preprocessor expects.
    model = RetentionModel(cohort="hybrid")
    model.fit(X_tr, y_tr)

    # predict_proba returns shape (n_rows, 2): [P(class=0), P(class=1)]
    # We evaluate the positive-class column [:, 1] throughout.
    proba_full = model.predict_proba(X_val)
    proba = proba_full[:, 1]

    # ── Check 1: No NaN or Inf ────────────────────────────────────────
    # A NaN output means a preprocessing step silently failed — e.g., median
    # imputation was skipped on a column, or XGBoost received an unexpected
    # dtype. This is the most dangerous failure mode because it propagates
    # silently through pandas downstream operations.
    assert np.isfinite(proba).all(), (
        f"GBM predict_proba returned non-finite values: "
        f"NaN count={np.isnan(proba).sum()}, Inf count={np.isinf(proba).sum()}. "
        "Check the preprocessor for missing imputation steps."
    )

    # ── Check 2: All values in [0, 1] ────────────────────────────────
    # XGBoost with objective='binary:logistic' (default) always outputs
    # values in [0, 1]. This assertion guards against objective changes
    # (e.g., switching to 'binary:hinge') that would break downstream
    # calibration and threshold code.
    assert (proba >= 0.0).all(), (
        f"GBM returned negative probabilities: min={proba.min():.6f}. "
        "The model output is not a valid probability distribution."
    )
    assert (proba <= 1.0).all(), (
        f"GBM returned probabilities > 1.0: max={proba.max():.6f}. "
        "The model output is not a valid probability distribution."
    )

    # ── Check 3: Non-degenerate output ───────────────────────────────
    # If all predicted probabilities are identical, the model ignores all
    # features and just predicts the prior. This would show as perfect Brier
    # score equality across cohorts — a clear signal of collapse.
    # We use nunique() rather than std() because std can be zero even with
    # 2 distinct values if the sample is small.
    n_unique = len(np.unique(np.round(proba, decimals=6)))
    assert n_unique >= 2, (
        f"GBM predict_proba returned only {n_unique} unique value(s) across "
        f"{len(proba)} validation rows — model output is constant. "
        "This indicates model collapse. Check preprocessing or training data "
        "for degenerate inputs (all-zero features, zero-variance columns, etc.)."
    )

    # ── Check 4: Mean prediction within a very generous range ────────
    # This catches extreme collapse: a model that predicts 0.0 for every
    # employee (all negative class) or 1.0 for every employee (all positive).
    # The bounds [0.01, 0.99] are intentionally loose — we're not testing
    # calibration quality, just verifying the model is not fully degenerate.
    mean_proba = float(proba.mean())
    assert 0.01 <= mean_proba <= 0.99, (
        f"GBM mean predicted probability {mean_proba:.4f} is outside [0.01, 0.99]. "
        f"Training base rate was {float(y_tr.mean()):.2%}. "
        "This suggests the model has collapsed to always predicting one class."
    )


# ------------------------------------------------------------------ #
# Story 2.6.7 — No leakage at predict time                             #
# ------------------------------------------------------------------ #


def test_no_leakage_at_predict_time(
    synthetic_attrition_df: pd.DataFrame,
) -> None:
    """Label and identifier columns must not appear in X at train or predict time.

    Three leakage vectors this test guards against:

    1. **Target leakage** — 'voluntary_exit_label' in X.
       If extract_X_y() fails to drop the label, the model trains on the
       answer. AUC-PR jumps to ~1.0, masking a broken pipeline.

    2. **Identifier leakage** — 'employee_id' in X.
       employee_id is an arbitrary string ID with no causal relationship
       to exit risk. If included as a feature, tree models would memorize
       individual employee IDs and fail to generalise.

    3. **Temporal anchor leakage** — 'snapshot_date' in X.
       snapshot_date drives the temporal split boundary. Including it as a
       feature lets the model learn which split it's in (train vs val),
       causing inflated val-set metrics. The temporal anchor is a control
       variable, not a predictor.

    The test runs both cohorts because the feature column lists differ
    (hris_only: 7 features, hybrid: 10 features). A bug in _cohort_columns()
    that accidentally includes the label or an identifier column for one
    cohort but not the other would only be caught by testing both.
    """
    df = synthetic_attrition_df

    train_df, val_df, _ = temporal_split(df, save_indices=False)

    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    # Columns that must NEVER appear in a feature matrix X
    forbidden_cols = {
        "voluntary_exit_label",  # target — leakage vector 1
        "employee_id",  # primary key — leakage vector 2
        "snapshot_date",  # temporal anchor — leakage vector 3
    }

    for cohort in ("hris_only", "hybrid"):
        X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)
        X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)

        # ── Leakage check on training X ──────────────────────────────
        leaked_train = forbidden_cols & set(X_tr.columns)
        assert not leaked_train, (
            f"[{cohort}] Forbidden column(s) found in training X: {leaked_train}. "
            "extract_X_y() must return only feature columns — no label, identifier, "
            "or temporal anchor."
        )

        # ── Leakage check on validation X ────────────────────────────
        # Predict-time leakage: the same forbidden columns must be absent
        # from the validation feature matrix. This is distinct from training
        # leakage — it's possible for a bug to accidentally include the label
        # in X_val even when X_tr is clean (e.g. if val_df wasn't label-stripped).
        leaked_val = forbidden_cols & set(X_val.columns)
        assert not leaked_val, (
            f"[{cohort}] Forbidden column(s) found in validation X: {leaked_val}. "
            "The feature matrix at predict time must not include the label, "
            "employee_id, or snapshot_date."
        )

        # ── Schema consistency: train and val columns must match ──────
        # A column present in X_tr but absent from X_val (or vice versa)
        # causes sklearn to raise or silently produce wrong predictions
        # (depending on whether ColumnTransformer uses column names or
        # positional indices). Either outcome is wrong.
        assert list(X_tr.columns) == list(X_val.columns), (
            f"[{cohort}] Feature column list differs between train and val.\n"
            f"  In train but not val: {set(X_tr.columns) - set(X_val.columns)}\n"
            f"  In val but not train: {set(X_val.columns) - set(X_tr.columns)}\n"
            "Schema drift between splits would cause silent prediction errors."
        )

        # ── y is the only place the label lives ──────────────────────
        # Belt-and-suspenders: confirm y is the label column, and that
        # it's a Series (not a DataFrame with extra columns smuggled in).
        assert isinstance(y_tr, pd.Series), (
            f"[{cohort}] y_train is {type(y_tr).__name__}, expected pd.Series. "
            "extract_X_y() must return a 1-D label Series, not a DataFrame."
        )
        assert y_tr.name == "voluntary_exit_label", (
            f"[{cohort}] y_train.name = '{y_tr.name}', expected 'voluntary_exit_label'. "
            "The label Series should carry its column name for downstream logging."
        )
        assert isinstance(y_val, pd.Series), (
            f"[{cohort}] y_val is {type(y_val).__name__}, expected pd.Series."
        )

        # ── Label values are binary int ───────────────────────────────
        # Both 0 and 1 should appear (we need positives to compute AUC-PR).
        # The synthetic fixture is designed for ~25% positive rate so this
        # holds on the 70-row training split.
        unique_labels = set(y_tr.unique())
        assert unique_labels <= {0, 1}, (
            f"[{cohort}] y_train contains non-binary values: {unique_labels}. "
            "voluntary_exit_label must be 0 or 1."
        )
