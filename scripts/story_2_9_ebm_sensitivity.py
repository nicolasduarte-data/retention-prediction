"""Story 2.9 — EBM OHE sensitivity check.

Trains two EBM variants on the same real data and compares all Loop 2 metrics:

    EBM-native  — uses build_ebm_preprocessor() (no OHE; native categorical
                  binning via set_output("pandas") passthrough)
    EBM-OHE     — uses build_preprocessor() (same OHE pipeline as LR + GBM;
                  categorical columns one-hot encoded before EBM sees them)

**Why this script exists:**
The cross-model comparison in Story 2.5 uses a different preprocessor for EBM
than for LR/GBM. A staff IC reviewer will ask: "did you isolate model
architecture from preprocessing choice?" This script answers that question
by running the sensitivity check and recording the AUC-PR delta.

**Expected result (from InterpretML documentation + prior literature):**
Native categorical handling should be equivalent-or-slightly-better than OHE
for EBM on low-cardinality categoricals. With two low-cardinality columns
(performance_tier: 4 levels, gender: 3 levels) on 892 training rows, we
expect a small delta (|ΔAUC-PR| < 0.010) — confirming preprocessing is not
the driver of EBM's trailing position in the main comparison table.

Usage:
    uv run python scripts/story_2_9_ebm_sensitivity.py

Output: a Markdown comparison table printed to stdout, ready for
pasting into docs/methodology.md → EBM Preprocessing Sensitivity.
"""

from __future__ import annotations

# ruff: noqa: E402 — sys.path.insert() before imports is intentional for scripts/
# that live outside src/ and need to import the retention package without a
# full editable install. All module-level imports intentionally follow sys.path setup.

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# This script lives in scripts/ but imports from src/. We need to add
# src/ to sys.path so Python finds the retention package without requiring
# `uv run` with the package installed in editable mode.
# Alternative: rely on pyproject.toml having `tool.setuptools.packages.find`
# with `where = ["src"]` AND the package installed via `uv pip install -e .`.
# Both work; we use sys.path manipulation as the more portable fallback.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from retention import config
from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.evaluation.metrics import (
    auc_pr,
    auc_roc,
    brier_score,
    precision_at_k,
    recall_at_k,
)
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.features.preprocessing import build_preprocessor
from retention.models.ebm import train_ebm


# ── EBM-OHE helper ────────────────────────────────────────────────────────────
# This is NOT added to train_ebm() — it's a one-off sensitivity check.
# The production EBM is always EBM-native. This variant exists only to
# answer the "did you test both preprocessors?" question.


def _train_ebm_ohe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cohort: str,
    *,
    interactions: int = 10,
    max_bins: int = 256,
    outer_bags: int = 8,
    random_state: int = config.SEED,
) -> Pipeline:
    """Train EBM with OHE preprocessing — sensitivity check only.

    Uses the SAME build_preprocessor() as LR and GBM (OneHotEncoder for
    categoricals, StandardScaler for numerics). EBM therefore sees:
      - numeric columns: float64 (scaled)
      - categorical columns: one-hot binary columns (float64)
      - boolean columns: float64

    This destroys EBM's native categorical advantage — each level of
    performance_tier and gender becomes a separate binary feature instead
    of one GAM shape function term. The expected effect: slightly lower
    AUC-PR and less interpretable shape functions.

    Teaching note — what happens to EBM with OHE input:
        EBM's native categorical handler groups a column's values into bins.
        Given "performance_tier" it learns one shape function: a curve across
        tiers 2/3/4/5. With OHE, it instead gets four binary columns
        (performance_tier_2, ..._3, ..._4, ..._5) and learns four independent
        shape functions — one per binary indicator. Logically equivalent but
        harder to read, and EBM's interaction detector (FAST algorithm) may
        detect spurious interactions between the OHE columns that the native
        handler would merge into a single term.
    """
    # build_preprocessor() applies OHE to categoricals (same as LR/GBM).
    # The output is a numpy array, not a DataFrame — EBM receives float64
    # everywhere and treats all features as continuous.
    preprocessor = build_preprocessor(cohort=cohort)  # type: ignore[arg-type]
    X_train_t = preprocessor.fit_transform(X_train)

    # Same balanced weighting as EBM-native — FLIP-RISK mitigation.
    y_train_arr = y_train.to_numpy()
    sample_weights = compute_sample_weight("balanced", y_train_arr)

    # Identical hyperparameters to EBM-native — we're isolating preprocessing,
    # not tuning the model. Any performance difference is attributable to
    # preprocessing alone.
    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        max_bins=max_bins,
        outer_bags=outer_bags,
        random_state=random_state,
    )
    ebm.fit(X_train_t, y_train_arr, sample_weight=sample_weights)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ebm),
        ]
    )


# ── Metric helper ─────────────────────────────────────────────────────────────


def _compute_all_metrics(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label: str,
) -> dict[str, object]:
    """Run predict_proba and compute all 6 Loop 2 metrics.

    Teaching note — predict_proba output shape:
        sklearn classifiers return shape (n, 2): column 0 = P(class=0),
        column 1 = P(class=1). We always evaluate column 1 — the
        positive-class (exit) probability. The metric functions accept
        either (n,) or (n, 2) and extract column 1 automatically.
    """
    proba = pipeline.predict_proba(X_val)
    return {
        "variant": label,
        "AUC-PR": round(auc_pr(y_val, proba), 3),
        "AUC-ROC": round(auc_roc(y_val, proba), 3),
        "Prec@10%": round(precision_at_k(y_val, proba, k=0.10), 3),
        "Prec@20%": round(precision_at_k(y_val, proba, k=0.20), 3),
        "Rec@10%": round(recall_at_k(y_val, proba, k=0.10), 3),
        "Brier": round(brier_score(y_val, proba), 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run EBM sensitivity check and print results."""
    # ── Step 1: Load data ────────────────────────────────────────────────
    # Use the same canonical CSV that the main comparison notebook uses.
    # load_attrition_features_local() validates the schema contract and
    # parses snapshot_date — ready for temporal_split().
    # CSVs live in data/raw/ (gitignored BigQuery snapshots).
    # Use the 2026-05-28 snapshot — the same one the main comparison table
    # was generated on — so the EBM-native numbers are directly comparable
    # to the 6-cell table in docs/methodology.md.
    csv_path = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-28.csv"
    if not csv_path.exists():
        csv_path = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-29.csv"
    if not csv_path.exists():
        csv_path = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-27.csv"
    print(f"Loading: {csv_path.name}")
    df = load_attrition_features_local(csv_path)
    print(f"  {len(df):,} rows loaded")

    # ── Step 2: Temporal split ───────────────────────────────────────────
    # save_indices=False — we're not overwriting the production split record.
    # This mirrors the main comparison notebook setup exactly.
    config.set_global_seed()
    train_df, val_df, _test_df = temporal_split(df, save_indices=False)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(_test_df):,}")

    # ── Step 3: Cohort split ─────────────────────────────────────────────
    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    # ── Step 4: Train both EBM variants per cohort ───────────────────────
    results: list[dict[str, object]] = []

    for cohort in ("hris_only", "hybrid"):
        print(f"\n  Training EBM variants for cohort={cohort!r} …")

        X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)
        X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)

        # EBM-native — production variant (same as main comparison table)
        config.set_global_seed()
        ebm_native = train_ebm(X_tr, y_tr, cohort=cohort)  # type: ignore[arg-type]
        print(f"    EBM-native ({cohort}) done")

        # EBM-OHE — sensitivity variant (OHE preprocessing, no native bins)
        config.set_global_seed()
        ebm_ohe = _train_ebm_ohe(X_tr, y_tr, cohort=cohort)
        print(f"    EBM-OHE   ({cohort}) done")

        # Evaluate both on the same validation set
        results.append(
            _compute_all_metrics(
                ebm_native,
                X_val,
                y_val,
                label=f"EBM-native ({cohort})",
            )
        )
        results.append(
            _compute_all_metrics(
                ebm_ohe,
                X_val,
                y_val,
                label=f"EBM-OHE    ({cohort})",
            )
        )

    # ── Step 5: Print results ─────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("variant")

    print("\n\n" + "=" * 72)
    print("STORY 2.9 — EBM Preprocessing Sensitivity Check")
    print("=" * 72)
    print("\nData:", csv_path.name)
    print(
        f"Val set: {len(val_df):,} rows | {int(y_val.sum())} positives ({y_val.mean():.1%} base rate)"
    )
    print()
    print(results_df.to_string())

    # AUC-PR delta summary
    print("\n--- AUC-PR delta (EBM-native minus EBM-OHE) ---")
    for cohort in ("hris_only", "hybrid"):
        native_auc = float(
            results_df.loc[f"EBM-native ({cohort})", "AUC-PR"]  # type: ignore[arg-type]
        )
        ohe_auc = float(
            results_df.loc[f"EBM-OHE    ({cohort})", "AUC-PR"]  # type: ignore[arg-type]
        )
        delta = native_auc - ohe_auc
        direction = "native wins" if delta > 0 else "OHE wins" if delta < 0 else "tied"
        print(
            f"  {cohort:10s}: native={native_auc:.3f}  OHE={ohe_auc:.3f}  "
            f"delta={delta:+.3f}  ({direction})"
        )

    print("\n--- Interpretation ---")
    print(
        "If |delta| < 0.010: preprocessing choice is negligible -- the comparison table is robust.\n"
        "If |delta| > 0.020: preprocessing meaningfully affects EBM rank -- caveat needed in write-up."
    )
    print("=" * 72)

    # ── Step 6: Emit Markdown table for methodology.md paste ────────────
    print("\n\n--- MARKDOWN TABLE (paste into methodology.md) ---")
    print()
    print("| Variant | AUC-PR | AUC-ROC | Prec@10% | Prec@20% | Rec@10% | Brier |")
    print("|---|---|---|---|---|---|---|")
    for row in results:
        variant = str(row["variant"])
        print(
            f"| {variant} | {row['AUC-PR']} | {row['AUC-ROC']} | "
            f"{row['Prec@10%']} | {row['Prec@20%']} | {row['Rec@10%']} | {row['Brier']} |"
        )


if __name__ == "__main__":
    main()
