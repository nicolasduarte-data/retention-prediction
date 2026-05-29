"""Story 3.2.3 — Generate per-model per-cohort calibration figures.

Trains all 6 model×cohort cells on the real data and saves reliability
diagrams to reports/figures/calibration_<model>_<cohort>.png.

**What a reliability diagram shows:**
    x-axis: mean predicted probability in each probability bin
    y-axis: observed positive fraction in each bin
    45° line: perfect calibration reference
    Blue bars: model is under-confident (predicted < observed)
    Red bars: model is over-confident (predicted > observed)

**Why this script is separate from the comparison notebook:**
    The notebook (Story 3.6) narrates the full evaluation story.
    This script is the figure generator — it can be re-run independently
    to refresh figures when data changes, without re-running the whole
    analysis. Call it from ``make calibration-figures``.

Usage:
    uv run python scripts/generate_calibration_figures.py
    # or via Makefile:
    make calibration-figures

Output:
    reports/figures/calibration_LR_hris_only.png
    reports/figures/calibration_LR_hybrid.png
    reports/figures/calibration_GBM_hris_only.png
    reports/figures/calibration_GBM_hybrid.png
    reports/figures/calibration_EBM_hris_only.png
    reports/figures/calibration_EBM_hybrid.png
    reports/figures/calibration_all.png  (6-panel grid for README/notebook)
"""

from __future__ import annotations

# ruff: noqa: E402 — sys.path.insert() before imports is intentional for scripts/
# that live outside src/ and need to import the retention package without a
# full editable install. All module-level imports intentionally follow sys.path setup.

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt

from retention import config
from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.evaluation.calibration import expected_calibration_error, reliability_diagram
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.models.ebm import train_ebm
from retention.models.lr import train_lr
from retention.models.xgb import RetentionModel

# ── Constants ─────────────────────────────────────────────────────────────────

FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"
DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-28.csv"
if not DATA_CSV.exists():
    DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-29.csv"

# Model display names for figure titles
MODEL_NAMES = {"LR": "Logistic Regression", "GBM": "Gradient-Boosted Machine", "EBM": "EBM"}
COHORT_LABELS = {"hris_only": "HRIS-only (7 features)", "hybrid": "Hybrid (10 features)"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _train_all_models(
    train_df: object,
    val_df: object,
) -> dict[str, dict[str, object]]:
    """Train all 6 model×cohort cells. Return nested dict of fitted pipelines.

    Return structure: {'LR': {'hris_only': pipeline, 'hybrid': pipeline}, ...}

    Teaching note — why we seed before each fit:
        config.set_global_seed() resets both numpy.random and Python's random
        module to SEED=42. This ensures each model's stochastic components
        (GBM subsampling, EBM bagging) start from the same state regardless
        of training order. Without it, training GBM after LR would consume
        a different numpy state than training GBM alone.
    """
    from retention.features.cohorts import split_cohorts as _split  # noqa: PLC0415

    train_cohorts = _split(train_df)  # type: ignore[arg-type]
    # val_cohorts from _split(val_df) is not needed here — _train_all_models
    # only trains on training data; val split is handled by the caller.

    models: dict[str, dict[str, object]] = {"LR": {}, "GBM": {}, "EBM": {}}

    for cohort in ("hris_only", "hybrid"):
        X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)  # type: ignore[arg-type]

        # LR
        config.set_global_seed()
        models["LR"][cohort] = train_lr(X_tr, y_tr, cohort=cohort)  # type: ignore[arg-type]
        print(f"  LR ({cohort}) done")

        # GBM
        config.set_global_seed()
        gbm = RetentionModel(cohort=cohort)  # type: ignore[arg-type]
        gbm.fit(X_tr, y_tr)
        models["GBM"][cohort] = gbm
        print(f"  GBM ({cohort}) done")

        # EBM — slowest; done last so any timeout catches it first
        config.set_global_seed()
        models["EBM"][cohort] = train_ebm(X_tr, y_tr, cohort=cohort)  # type: ignore[arg-type]
        print(f"  EBM ({cohort}) done")

    return models


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Train all models, compute calibration, save figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load + split ─────────────────────────────────────────────
    print(f"Loading {DATA_CSV.name}...")
    df = load_attrition_features_local(DATA_CSV)
    print(f"  {len(df):,} rows")

    config.set_global_seed()
    train_df, val_df, _ = temporal_split(df, save_indices=False)
    print(f"  train={len(train_df):,}  val={len(val_df):,}")

    val_cohorts = split_cohorts(val_df)

    # ── Step 2: Train all models ─────────────────────────────────────────
    print("\nTraining 6 model×cohort cells...")
    all_models = _train_all_models(train_df, val_df)

    # ── Step 3: Per-cell calibration figures ─────────────────────────────
    print("\nGenerating calibration figures...")
    summary_rows: list[dict[str, object]] = []

    for model_key in ("LR", "GBM", "EBM"):
        for cohort in ("hris_only", "hybrid"):
            pipeline = all_models[model_key][cohort]
            X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)  # type: ignore[arg-type]

            proba = pipeline.predict_proba(X_val)  # type: ignore[union-attr]

            # Individual figure
            fig = reliability_diagram(
                y_val,
                proba,
                n_bins=10,
                model_name=model_key,
                cohort=cohort,
            )
            out_path = FIGURES_DIR / f"calibration_{model_key}_{cohort}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            ece = expected_calibration_error(y_val, proba)
            print(f"  {model_key} × {cohort}: ECE={ece:.3f}  -> {out_path.name}")
            summary_rows.append({"model": model_key, "cohort": cohort, "ece": round(ece, 3)})

    # ── Step 4: 6-panel grid figure ─────────────────────────────────────
    # One combined figure for the notebook / README — 3 cols (models) × 2 rows (cohorts).
    print("\nGenerating 6-panel grid figure...")
    fig_grid, axes = plt.subplots(2, 3, figsize=(12, 8))

    cohorts = ["hris_only", "hybrid"]
    model_keys = ["LR", "GBM", "EBM"]

    for row_idx, cohort in enumerate(cohorts):
        for col_idx, model_key in enumerate(model_keys):
            ax = axes[row_idx, col_idx]
            pipeline = all_models[model_key][cohort]
            X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)  # type: ignore[arg-type]
            proba = pipeline.predict_proba(X_val)  # type: ignore[union-attr]

            # Draw into existing Axes (ax parameter)
            reliability_diagram(
                y_val,
                proba,
                n_bins=10,
                model_name=model_key,
                cohort=cohort,
                ax=ax,
            )
            # Override title to shorter form for the grid
            ece = expected_calibration_error(y_val, proba)
            ax.set_title(f"{model_key} × {cohort}\nECE={ece:.3f}", fontsize=9)

    fig_grid.suptitle(
        "Reliability Diagrams — All 6 Model×Cohort Cells (val set)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    grid_path = FIGURES_DIR / "calibration_all.png"
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"  Grid saved -> {grid_path.name}")

    # ── Step 5: Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ECE Summary (validation set, n_bins=10)")
    print("=" * 55)
    print(f"{'Model':<6}  {'Cohort':<12}  ECE")
    print("-" * 35)
    for row in summary_rows:
        print(f"{row['model']:<6}  {row['cohort']:<12}  {row['ece']:.3f}")
    print("=" * 55)
    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
