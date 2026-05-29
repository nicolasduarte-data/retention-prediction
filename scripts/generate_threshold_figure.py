"""Story 3.3 — Generate the threshold-sweep figure for the leading model.

Trains GBM × hybrid (the point-estimate leader from the Loop 2 comparison) on
the real data, sweeps decision thresholds on the **validation** set, and saves
the treatment-arm comparison chart to reports/figures/threshold_sweep.png.

**Why GBM × hybrid only:** A deployed system has exactly one operating point,
so the threshold decision is made for the model we would actually ship. GBM ×
hybrid leads AUC-PR (0.309) and is the only cell that also tops the calibration
metric (Brier 0.158) — trustworthy probabilities are a precondition for a
trustworthy threshold. The same procedure applies unchanged to any other cell.

**Why the validation set, not test:** Choosing an operating point is a
model-selection decision. Optimising the threshold on test and then reporting
test numbers at that threshold is leakage. The winning arm is confirmed once on
the held-out test set at champion selection (Story 3.6); until then test is
pristine. See src/retention/models/threshold.py for the full rationale.

Usage:
    uv run python scripts/generate_threshold_figure.py

Output:
    reports/figures/threshold_sweep.png
    + a printed table of the F1 / F2 / Youden operating points (validation set)
"""

from __future__ import annotations

# ruff: noqa: E402 — sys.path.insert() before imports is intentional for scripts/
# that live outside src/ and need to import the retention package without a full
# editable install. All module-level imports intentionally follow sys.path setup.

import sys
from pathlib import Path
from typing import Literal

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from retention import config
from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.models.threshold import optimize_threshold, threshold_sweep, threshold_sweep_plot
from retention.models.xgb import RetentionModel

# ── Constants ─────────────────────────────────────────────────────────────────

FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"
DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-28.csv"
if not DATA_CSV.exists():
    DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-29.csv"

COHORT: Literal["hris_only", "hybrid"] = "hybrid"  # the leading cell uses the survey-signal cohort
CRITERIA = ("f1", "f2", "youden")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Train GBM × hybrid, sweep thresholds on val, save the figure + table."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load + temporal split ────────────────────────────────────
    print(f"Loading {DATA_CSV.name}...")
    df = load_attrition_features_local(DATA_CSV)
    print(f"  {len(df):,} rows")

    config.set_global_seed()
    train_df, val_df, _ = temporal_split(df, save_indices=False)
    print(f"  train={len(train_df):,}  val={len(val_df):,}")

    train_cohorts = split_cohorts(train_df)
    val_cohorts = split_cohorts(val_df)

    # ── Step 2: Train the leading model (GBM × hybrid) ───────────────────
    print(f"\nTraining GBM × {COHORT}...")
    X_tr, y_tr = extract_X_y(train_cohorts[COHORT], COHORT)
    X_val, y_val = extract_X_y(val_cohorts[COHORT], COHORT)

    config.set_global_seed()
    gbm = RetentionModel(cohort=COHORT)
    gbm.fit(X_tr, y_tr)
    proba = gbm.predict_proba(X_val)
    base_rate = float(np.asarray(y_val).mean())
    print(f"  done — val base rate = {base_rate:.3f}")

    # ── Step 3: Sweep + optimise (each criterion = one arm-ranking lens) ──
    print("\nSweeping thresholds 0.05..0.95 (validation set)...")
    sweep = threshold_sweep(y_val, proba)

    operating_points: dict[str, float] = {}
    for criterion in CRITERIA:
        t = optimize_threshold(y_val, proba, criterion)
        operating_points[criterion] = t
        row = sweep.loc[np.isclose(sweep["threshold"], t)].iloc[0]
        print(
            f"  {criterion.upper():<6} optimal t={t:.2f}  "
            f"precision={row['precision']:.3f}  recall={row['recall']:.3f}  "
            f"f1={row['f1']:.3f}  f2={row['f2']:.3f}  "
            f"(tp={int(row['tp'])} fp={int(row['fp'])} fn={int(row['fn'])})"
        )

    # ── Step 4: Figure — F1 arm as the headline, F2 arm for contrast ─────
    print("\nGenerating threshold_sweep.png...")
    t_f1 = operating_points["f1"]
    t_f2 = operating_points["f2"]

    fig = threshold_sweep_plot(
        sweep,
        optimal_threshold=t_f1,
        criterion="f1",
        model_name="GBM",
        cohort=COHORT,
    )
    # Overlay the F2 arm so the [FLIP-RISK] recall preference is visible: the
    # recall-weighted criterion sits at-or-below the F1 arm, flagging more people.
    axes = fig.axes[0]
    axes.axvline(
        t_f2,
        color="#dd8452",
        linestyle=":",
        linewidth=1.4,
        label=f"F2-optimal arm (t={t_f2:.2f}) — [FLIP-RISK]",
    )
    axes.legend(fontsize=8, loc="best")  # refresh legend to include the F2 line

    fig.suptitle(
        "Threshold selection as a controlled experiment — each threshold is a treatment arm",
        fontsize=11,
        y=1.00,
    )
    fig.text(
        0.5,
        -0.02,
        "Arms compared on the validation set; the empirically-winning arm is "
        "confirmed once on the held-out test set at champion selection (Story 3.6).",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#555555",
    )

    out_path = FIGURES_DIR / "threshold_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # ── Step 5: Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Threshold operating points — GBM × {COHORT} (validation set)")
    print("=" * 60)
    print(f"  base rate              : {base_rate:.3f}")
    print(f"  F1-optimal threshold   : {operating_points['f1']:.2f}")
    print(f"  F2-optimal threshold   : {operating_points['f2']:.2f}  (recall-weighted)")
    print(f"  Youden-optimal threshold: {operating_points['youden']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
