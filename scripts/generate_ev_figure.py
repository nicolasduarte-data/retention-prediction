"""Story 3.4 — Generate the expected-value sensitivity figure for the leading model.

Trains GBM × hybrid (the Loop 2 point-estimate leader), fixes the operating
point at the Story 3.3 F1-optimal threshold, and sweeps intervention
effectiveness p_eff across [0.1, 0.9] on the **validation** set — turning the
threshold decision into dollars. Saves reports/figures/ev_sensitivity.png and
prints the breakeven table.

**Why GBM × hybrid + the F1 threshold:** the same shipped operating point as the
threshold figure (Story 3.3). EV reuses the exact validation probabilities and
threshold, so the money story and the statistical story describe one model — not
two differently-tuned ones.

**Why a representative salary:** the dataset carries compa_ratio (a salary
*ratio*), not absolute salary, so replacement_cost uses the module default —
1.5 × $60k representative = $90k (SHRM 2024 multiplier). In a real deployment you
would substitute the cohort's actual mean salary via
``replacement_cost_from_salary(mean_salary)``; the framing is unchanged, only the
scale moves.

**Why the validation set, not test:** EV informs the operating-point decision,
which is a model-selection step. Optimising on test then reporting test numbers
is leakage. The winning arm is confirmed once on the held-out test set at
champion selection (Story 3.6).

Usage:
    uv run python scripts/generate_ev_figure.py

Output:
    reports/figures/ev_sensitivity.png
    + a printed breakeven table (F1 vs F2 arm: precision, breakeven p_eff, EV@0.30)
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
from retention.evaluation.expected_value import (
    DEFAULT_INTERVENTION_COST,
    DEFAULT_P_EFF,
    DEFAULT_REPLACEMENT_COST,
    breakeven_p_eff,
    ev_at_threshold,
    expected_value_plot,
    p_eff_sensitivity_sweep,
)
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.models.threshold import optimize_threshold, threshold_sweep
from retention.models.xgb import RetentionModel

# ── Constants ─────────────────────────────────────────────────────────────────

FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"
DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-28.csv"
if not DATA_CSV.exists():
    DATA_CSV = _PROJECT_ROOT / "data" / "raw" / "v_attrition_features_2026-05-29.csv"

COHORT: Literal["hris_only", "hybrid"] = "hybrid"  # the leading cell uses the survey-signal cohort


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Train GBM × hybrid, sweep p_eff on val at the F1 operating point, save figure."""
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

    # ── Step 3: Operating points from Story 3.3 (F1 ships, F2 for contrast) ─
    sweep = threshold_sweep(y_val, proba)
    t_f1 = optimize_threshold(y_val, proba, "f1")
    t_f2 = optimize_threshold(y_val, proba, "f2")
    print(f"\nOperating points (validation set):  F1 t={t_f1:.2f}   F2 t={t_f2:.2f}")

    # ── Step 4: EV sensitivity sweep at the shipped (F1) operating point ──
    print("\nSweeping p_eff 0.1..0.9 at the F1 operating point...")
    ev_sweep = p_eff_sensitivity_sweep(y_val, proba, t_f1)  # module-default economics
    be_f1 = breakeven_p_eff(y_val, proba, t_f1)
    ev_central = ev_at_threshold(y_val, proba, t_f1)  # at DEFAULT_P_EFF = 0.30
    print(
        f"  replacement_cost=${DEFAULT_REPLACEMENT_COST:,.0f}  "
        f"intervention_cost=${DEFAULT_INTERVENTION_COST:,.0f}"
    )
    print(f"  breakeven p_eff = {be_f1:.3f}    EV @ p_eff={DEFAULT_P_EFF:.2f} = ${ev_central:,.0f}")

    # ── Step 5: Figure ────────────────────────────────────────────────────
    print("\nGenerating ev_sensitivity.png...")
    fig = expected_value_plot(
        ev_sweep,
        breakeven=be_f1,
        threshold=t_f1,
        model_name="GBM",
        cohort=COHORT,
    )
    fig.suptitle(
        "Expected value of the retention program vs. doing nothing",
        fontsize=11,
        y=1.00,
    )
    fig.text(
        0.5,
        -0.02,
        # Escape the $ so matplotlib does not read the pair as mathtext delimiters
        # (a matched $...$ renders the span in math italic and eats the spaces).
        f"Validation set; replacement cost \\${DEFAULT_REPLACEMENT_COST:,.0f} "
        f"(SHRM 1.5x representative salary), intervention \\${DEFAULT_INTERVENTION_COST:,.0f}. "
        "Confirmed on test once, at champion selection (Story 3.6).",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#555555",
    )

    out_path = FIGURES_DIR / "ev_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    # ── Step 6: Breakeven table — precision drives breakeven ─────────────
    # The economic counterweight to 3.3's F2 recall pressure: the more precise
    # arm (F1) needs a *lower* intervention effectiveness to pay for itself.
    print("\n" + "=" * 64)
    print(f"Expected-value summary — GBM × {COHORT} (validation set)")
    print("=" * 64)
    print(f"  {'arm':<6}{'t':>6}{'precision':>11}{'breakeven p_eff':>18}{'EV@0.30':>12}")
    for name, t in (("F1", t_f1), ("F2", t_f2)):
        row = sweep.loc[np.isclose(sweep["threshold"], t)].iloc[0]
        be = breakeven_p_eff(y_val, proba, t)
        ev_030 = ev_at_threshold(y_val, proba, t)
        be_str = "inf" if not np.isfinite(be) else f"{be:.3f}"
        print(f"  {name:<6}{t:>6.2f}{row['precision']:>11.3f}{be_str:>18}{ev_030:>12,.0f}")
    print("=" * 64)
    print(
        "  Reading: breakeven = (intervention/replacement) / precision, so the\n"
        "  more precise arm needs less-effective interventions to break even."
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
