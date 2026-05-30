"""Generate notebooks/03_evaluation_rigor.ipynb (Story 3.6).

The Epic 3 capstone notebook. Epic 3 built five evaluation lenses —
discrimination (3.1), calibration (3.2), threshold (3.3), expected value (3.4),
and unbiased generalisation (3.5, nested CV). This notebook converges them into
a single, defensible decision: of the six model x cohort cells, which one ships?

Narrative arc:
    setup -> data + temporal split -> cohort splits -> train the six cells ->
    cache validation probabilities -> Lens 1: nested CV (optimism check) ->
    six-cell summary table -> champion selection (calibration gate, then AUC-PR
    rank) -> the champion's operating point (threshold sweep) -> the business
    case (EV sensitivity + breakeven) -> test-set confirmation (once) ->
    persist champion.pkl -> register rp-champion/Production (Story 2.7.10).

All logic lives in src/retention/* (mypy + pytest verified); this notebook only
imports and calls. The generator mirrors scripts/generate_comparison_notebook.py.

Run with:
    uv run python scripts/generate_evaluation_notebook.py
Then execute the notebook (nested CV needs a longer per-cell timeout than the
30s nbconvert default):
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 notebooks/03_evaluation_rigor.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NB_PATH = Path("notebooks/03_evaluation_rigor.ipynb")


def md(source: str):  # noqa: ANN201
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):  # noqa: ANN201
    return nbformat.v4.new_code_cell(source)


cells = [
    md(
        "# Story 3.6 — Evaluation Rigor → Champion Selection\n\n"
        "**Epic 3 capstone.** Epic 3 built five lenses for looking at a model:\n\n"
        "1. **Discrimination** (3.1) — AUC-PR, precision@k\n"
        "2. **Calibration** (3.2) — Brier, ECE: are the probabilities trustworthy?\n"
        "3. **Threshold** (3.3) — where do we draw the flag/no-flag line?\n"
        "4. **Expected value** (3.4) — what is the operating point worth in dollars?\n"
        "5. **Unbiased generalisation** (3.5) — nested CV: an honest AUC-PR estimate\n\n"
        "This notebook converges them into one decision: **of the six "
        "model × cohort cells (LR / GBM / EBM × `hris_only` / `hybrid`), which "
        "one do we ship?**\n\n"
        "The rule is a **gated rank, not a sort** (`src/retention/evaluation/"
        "champion.py`): a cell is *eligible* only if its Brier clears the "
        "base-rate bar `β·(1 − β)` — the Brier of the best constant predictor — "
        "and the champion is the highest-AUC-PR cell among the eligible ones. A "
        "cell can rank well while being mis-calibrated, and every downstream HR "
        "action (the threshold, the EV case) is computed from probabilities, so "
        "ranking alone is not enough to operate on.\n\n"
        "> **Test-set discipline.** Every selection number below is a "
        "**validation** number. The held-out test set is touched exactly once, "
        "in §11, to confirm the champion — never to choose it."
    ),
    md("## 1 — Setup"),
    code(
        r"""from __future__ import annotations

import datetime as dt
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from retention import config
from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.evaluation.calibration import expected_calibration_error
from retention.evaluation.champion import (
    ChampionArtifact,
    persist_champion,
    select_champion,
)
from retention.evaluation.expected_value import (
    DEFAULT_INTERVENTION_COST,
    DEFAULT_P_EFF,
    DEFAULT_REPLACEMENT_COST,
    breakeven_p_eff,
    ev_at_threshold,
    expected_value_plot,
    p_eff_sensitivity_sweep,
)
from retention.evaluation.metrics import auc_pr, brier_score, precision_at_k
from retention.evaluation.nested_cv import nested_cv_auc_pr
from retention.features.catalog import get_label_name
from retention.features.cohorts import extract_X_y, get_cohort_feature_names, split_cohorts
from retention.models.ebm import train_ebm
from retention.models.lr import train_lr
from retention.models.threshold import optimize_threshold, threshold_sweep, threshold_sweep_plot
from retention.models.tracking import register_champion
from retention.models.xgb import RetentionModel

# Silence only the library deprecation chatter (sklearn/mlflow/xgboost) so the
# narrative stays readable — NOT a blanket ignore. A genuine RuntimeWarning
# (e.g. a numerical issue) must still surface (feast T2-ORCH-2).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
config.configure_plot_style()
config.set_global_seed()

LABEL = get_label_name()
CRITERION = "f2"  # [FLIP-RISK]: recall weighted 2x precision — a missed exit
# costs more than a wasted retention conversation, so the operating point sits
# lower (flags more people). Same criterion the threshold story (3.3) argues for.
print(f"SEED={config.SEED}  label={LABEL}  operating-point criterion={CRITERION}")"""
    ),
    md(
        "## 2 — Data & split\n\n"
        "We load from the committed **CSV snapshot** (the reviewer path: no "
        "BigQuery credentials needed).\n\n"
        "**An honesty note about the split.** The current mart is a *single "
        "cross-section* — every row shares one `snapshot_date` "
        "(`docs/data_card.md` §3). `temporal_split()` therefore produces a "
        "**deterministic, leak-free holdout partitioned by `employee_id`** (the "
        "stable sort's tiebreak), **not** a true temporal split. With one row per "
        "employee, no employee appears in two splits, so the 70/15/15 holdout is "
        "a valid *generalisation* estimate — but it does **not** test *temporal* "
        "generalisation (train on month T, predict T+1). That needs `pa-warehouse` "
        "to ship time-series snapshots and is deferred. The `temporal_split` "
        "machinery (sort-by-date + `assert_no_temporal_leak`) is in place and "
        "arms automatically the day real time-series data lands; on a single "
        "snapshot it is correct but vacuous — no future to leak.\n\n"
        "The snapshot is pinned so this notebook reproduces the methodology's "
        "documented nested-CV and EV numbers, with a fallback to a later snapshot "
        "for a fresh clone."
    ),
    code(
        r"""# config.DATA_DIR is absolute (resolved from the installed package), so the
# load is independent of the working directory — nbconvert runs with cwd=notebooks/.
data_csv = config.DATA_DIR / "raw" / "v_attrition_features_2026-05-28.csv"
if not data_csv.exists():
    data_csv = config.DATA_DIR / "raw" / "v_attrition_features_2026-05-29.csv"

df = load_attrition_features_local(data_csv)
print(f"Loaded {data_csv.name}: {len(df):,} rows x {df.shape[1]} cols")

config.set_global_seed()
train_df, val_df, test_df = temporal_split(df, save_indices=False)
print(
    f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}  "
    f"(test held out until the single confirmation in section 11)"
)

val_base_rate = float(val_df[LABEL].mean())
print(
    f"base rates  ->  train={train_df[LABEL].mean():.3f}  "
    f"val={val_base_rate:.3f}  test={test_df[LABEL].mean():.3f}"
)"""
    ),
    md(
        "## 3 — Cohort splits\n\n"
        "The controlled experiment at the heart of Loop 2: **identical rows, "
        "different columns.** `hris_only` sees only HRIS features; `hybrid` adds "
        "the three survey-signal columns. Any AUC-PR difference is attributable "
        "to the survey signal, not to a different employee population."
    ),
    code(
        r"""train_cohorts = split_cohorts(train_df)
val_cohorts = split_cohorts(val_df)
test_cohorts = split_cohorts(test_df)

for cohort_name in ("hris_only", "hybrid"):
    n_feat = len(get_cohort_feature_names(cohort_name))
    print(f"  {cohort_name}: {n_feat} features")"""
    ),
    md(
        "## 4 — Train the six model × cohort cells\n\n"
        "Each model is fitted on the **training** split only. Preprocessors are "
        "fit on train (no val/test leakage) and assembled into fitted Pipelines "
        "so `predict_proba(X_raw)` transforms automatically. LR/GBM share the "
        "one-hot preprocessor; EBM uses native categorical handling — an "
        "acknowledged tradeoff documented in `docs/methodology.md`."
    ),
    code(
        r"""config.set_global_seed()
models = {}
for cohort in ("hris_only", "hybrid"):
    X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)

    models[("LR", cohort)] = train_lr(X_tr, y_tr, cohort=cohort)

    gbm = RetentionModel(cohort=cohort)
    gbm.fit(X_tr, y_tr)
    models[("GBM", cohort)] = gbm

    models[("EBM", cohort)] = train_ebm(X_tr, y_tr, cohort=cohort)

    print(f"[{cohort}] LR, GBM, EBM trained")"""
    ),
    md(
        "## 5 — Cache the validation probabilities\n\n"
        "Every lens below reads from the *same* frozen validation probabilities, "
        "so the summary table, the threshold sweep, and the EV case all describe "
        "one model — never differently-scored variants of it."
    ),
    code(
        r"""val_proba = {}
for (model_name, cohort), model in models.items():
    X_val, _ = extract_X_y(val_cohorts[cohort], cohort)
    val_proba[(model_name, cohort)] = model.predict_proba(X_val)

print(f"Cached validation probabilities for {len(val_proba)} cells")"""
    ),
    md(
        "## 6 — Lens 1: Nested cross-validation (the optimism check)\n\n"
        "A single validation score is optimistic: the model that wins on "
        "validation was, in part, *chosen* on validation. **Nested CV** "
        "separates the two jobs — an outer 5-fold loop measures performance on "
        "folds never seen during the inner 5-fold hyperparameter search — giving "
        "an unbiased AUC-PR estimate.\n\n"
        "Run on the **development set (train + val)** with the test set withheld, "
        "and on **GBM only**: it has the largest hyperparameter surface and is "
        "the one most prone to small-data overfitting, so it is the cell whose "
        "public AUC-PR claim most needs the safeguard (LR/EBM rationale in "
        "`nested_cv.py`). The gap between the flat validation score and the "
        "nested mean *is* the optimism we are correcting for."
    ),
    code(
        r"""# Nested CV is ~200 GBM fits per cohort — the slowest cell in the notebook
# (well under the raised nbconvert timeout). The test set is never passed in.
dev_df = pd.concat([train_df, val_df], ignore_index=True)
dev_cohorts = split_cohorts(dev_df)

nested_results = {}
for cohort in ("hris_only", "hybrid"):
    X_dev, y_dev = extract_X_y(dev_cohorts[cohort], cohort)
    config.set_global_seed()
    nested_results[cohort] = nested_cv_auc_pr(X_dev, y_dev, cohort=cohort)
    print(nested_results[cohort].summary())"""
    ),
    md(
        "## 7 — The six-cell summary table\n\n"
        "One row per model × cohort, all on the **validation** set:\n\n"
        "- **auc_pr** — discrimination (primary metric)\n"
        "- **prec_at_10** — of the top 10 % flagged, how many actually exit\n"
        "- **ece** — calibration error (lower = predicted probs match reality)\n"
        "- **brier** — squared-error calibration score (drives the gate)\n"
        "- **threshold** — the cell's own F2-optimal operating point\n"
        "- **ev_at_default** — EV in dollars at that threshold, p_eff = 0.30\n"
        "- **breakeven_p_eff** — intervention effectiveness at which EV = 0\n\n"
        "This table is the input to `select_champion`."
    ),
    code(
        r"""rows = []
for (model_name, cohort), proba in val_proba.items():
    _, y_val = extract_X_y(val_cohorts[cohort], cohort)
    threshold = optimize_threshold(y_val, proba, CRITERION)
    # Full precision on purpose: select_champion (§8) RANKS on auc_pr and GATES
    # on brier, so the selection must see unrounded values. Rounding is a display
    # concern only (below). Deciding on 3-dp-rounded metrics could flip the
    # champion between two cells within ~0.0005 of each other, or mis-classify a
    # cell whose true Brier sits just across the base-rate gate.
    rows.append(
        {
            "model": model_name,
            "cohort": cohort,
            "auc_pr": auc_pr(y_val, proba),
            "prec_at_10": precision_at_k(y_val, proba, k=0.10),
            "ece": expected_calibration_error(y_val, proba),
            "brier": brier_score(y_val, proba),
            "threshold": threshold,
            "ev_at_default": ev_at_threshold(y_val, proba, threshold),
            "breakeven_p_eff": breakeven_p_eff(y_val, proba, threshold),
        }
    )

summary = pd.DataFrame(rows)  # full precision — consumed by select_champion in §8

# Display only: round for legibility. The selection in §8 uses `summary` above.
summary.round(
    {"auc_pr": 3, "prec_at_10": 3, "ece": 3, "brier": 3,
     "threshold": 2, "ev_at_default": 0, "breakeven_p_eff": 3}
)"""
    ),
    md(
        "**Flat vs nested — the optimism gap.** Side by side, the single "
        "validation AUC-PR for GBM against its unbiased nested-CV mean. A flat "
        "score above the nested mean is the over-optimism nested CV exists to "
        "expose; the nested figure is the one to quote publicly."
    ),
    code(
        r"""for cohort in ("hris_only", "hybrid"):
    flat = summary.loc[
        (summary["model"] == "GBM") & (summary["cohort"] == cohort), "auc_pr"
    ].iloc[0]
    nested = nested_results[cohort]
    gap = flat - nested.mean
    print(
        f"GBM x {cohort:9s}:  flat val AUC-PR={flat:.3f}   "
        f"nested {nested.outer_splits}x{nested.inner_splits} CV={nested.mean:.3f} "
        f"+/- {nested.std:.3f}   (optimism gap {gap:+.3f})"
    )"""
    ),
    md(
        "## 8 — Champion selection: calibration gate, then AUC-PR rank\n\n"
        "`select_champion` applies the rule:\n\n"
        "1. **Gate** — keep only cells with `brier ≤ β·(1 − β)`, the base-rate "
        "Brier at the validation prevalence β. A cell above that bar is worse "
        "than predicting the base rate for everyone — its probabilities are not "
        "safe to operate, no matter how it ranks.\n"
        "2. **Rank** — among the eligible cells, take the highest AUC-PR.\n"
        "3. **Honest fallback** — if *no* cell clears the gate, pick the "
        "highest-AUC-PR cell overall, flag `passed_calibration_gate = False`, "
        "and say so: the ranking is usable, the probabilities need calibration "
        "first.\n\n"
        "The printed summary reports which path was taken — it is not assumed in "
        "advance."
    ),
    code(
        r"""selection = select_champion(summary, base_rate=val_base_rate, criterion=CRITERION)
print(selection.summary())"""
    ),
    md(
        "## 9 — The champion's operating point\n\n"
        "Threshold selection is a *within-system A/B test*: the model's ranking "
        'is frozen, and each candidate threshold is a treatment arm ("flag '
        'everyone with p ≥ t") scored on the same validation set. The dashed '
        "line marks the F2-optimal arm — the [FLIP-RISK] operating point the "
        "champion will ship with."
    ),
    code(
        r"""champ_key = (selection.model_name, selection.cohort)
champion_model = models[champ_key]
proba_val_champ = val_proba[champ_key]
_, y_val_champ = extract_X_y(val_cohorts[selection.cohort], selection.cohort)

sweep = threshold_sweep(y_val_champ, proba_val_champ)
threshold_sweep_plot(
    sweep,
    optimal_threshold=selection.threshold,
    criterion=CRITERION,
    model_name=selection.model_name,
    cohort=selection.cohort,
)
plt.show()"""
    ),
    md(
        "## 10 — The business case: expected value & breakeven\n\n"
        "An F-score is unit-free — it never prices a false positive. This puts "
        "the operating point in **dollars**: \n\n"
        "$$EV = p_{eff} \\cdot \\text{replacement\\_cost} \\cdot TP - "
        "\\text{intervention\\_cost} \\cdot (TP + FP)$$\n\n"
        "`p_eff` (how often a retention conversation actually works) is a forward "
        "assumption we cannot read off the data, so we never report a single "
        "number — we **sweep** it across [0.1, 0.9] and show the **breakeven**: "
        "the effectiveness at which the program starts paying for itself. The "
        "dataset carries a salary *ratio*, not absolute salary, so "
        "`replacement_cost` uses the documented default (1.5 × $60k = $90k, SHRM "
        "2024)."
    ),
    code(
        r"""ev_sweep = p_eff_sensitivity_sweep(y_val_champ, proba_val_champ, selection.threshold)
be = breakeven_p_eff(y_val_champ, proba_val_champ, selection.threshold)
ev_central = ev_at_threshold(y_val_champ, proba_val_champ, selection.threshold)

print(
    f"replacement_cost=${DEFAULT_REPLACEMENT_COST:,.0f}   "
    f"intervention_cost=${DEFAULT_INTERVENTION_COST:,.0f}"
)
print(f"breakeven p_eff = {be:.3f}    EV @ p_eff={DEFAULT_P_EFF:.2f} = ${ev_central:,.0f}")

expected_value_plot(
    ev_sweep,
    breakeven=be,
    threshold=selection.threshold,
    model_name=selection.model_name,
    cohort=selection.cohort,
)
plt.show()"""
    ),
    md(
        "## 11 — Test-set confirmation (once)\n\n"
        "The **only** time the held-out test set is touched. The champion was "
        "chosen entirely on validation; here we confirm it generalises. Both "
        "validation and test metrics are stored on the artifact so the optimism "
        "gap is visible in the persisted record, not hidden."
    ),
    code(
        r"""X_test_champ, y_test_champ = extract_X_y(test_cohorts[selection.cohort], selection.cohort)
proba_test_champ = champion_model.predict_proba(X_test_champ)

val_metrics = {
    "auc_pr": float(auc_pr(y_val_champ, proba_val_champ)),
    "prec_at_10": float(precision_at_k(y_val_champ, proba_val_champ, k=0.10)),
    "ece": float(expected_calibration_error(y_val_champ, proba_val_champ)),
    "brier": float(brier_score(y_val_champ, proba_val_champ)),
}
test_metrics = {
    "auc_pr": float(auc_pr(y_test_champ, proba_test_champ)),
    "prec_at_10": float(precision_at_k(y_test_champ, proba_test_champ, k=0.10)),
    "ece": float(expected_calibration_error(y_test_champ, proba_test_champ)),
    "brier": float(brier_score(y_test_champ, proba_test_champ)),
}

gap = pd.DataFrame({"validation": val_metrics, "test": test_metrics})
gap["optimism_gap"] = gap["validation"] - gap["test"]
gap.round(3)"""
    ),
    md(
        "## 12 — Persist the champion artifact\n\n"
        "`ChampionArtifact` bundles the fitted model + its operating threshold + "
        "full provenance (the selection decision, val/test metrics, seed, "
        "timestamp) and pickles it to `reports/models/champion.pkl` — the "
        "self-describing record Loop 4's write-back will load."
    ),
    code(
        r"""artifact = ChampionArtifact(
    model=champion_model,
    model_name=selection.model_name,
    cohort=selection.cohort,
    threshold=selection.threshold,
    feature_names=get_cohort_feature_names(selection.cohort),
    seed=config.SEED,
    selection=selection,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
)
champion_path = persist_champion(artifact)
print(f"persisted -> {champion_path}\n")
print(artifact.summary())"""
    ),
    md(
        "## 13 — Register to the MLflow Model Registry → Production\n\n"
        "**Story 2.7.10.** Where the experiment view records *every* run, the "
        "registry records the *one* winner: a named, versioned, stage-tagged "
        "entry. `register_champion` logs the fitted estimator, registers it "
        "under `rp-champion`, and promotes the new version to **Production** — "
        "the URI Loop 4 loads from (`models:/rp-champion/Production`). We pass "
        "the sklearn object (`RetentionModel.pipeline` for GBM; LR/EBM are "
        "Pipelines already) because `mlflow.sklearn` needs the estimator, not "
        "the wrapper."
    ),
    code(
        r"""# RetentionModel exposes the fitted sklearn Pipeline as .pipeline; LR/EBM
# champions already are Pipelines. getattr handles both.
sk_model = getattr(champion_model, "pipeline", champion_model)

# Explicit pip_requirements per family skips MLflow's slow (~20s) environment
# inference on every call — fine here because we know each family's deps.
pip_by_family = {
    "GBM": ["scikit-learn", "xgboost"],
    "LR": ["scikit-learn"],
    "EBM": ["scikit-learn", "interpret"],
}

run_id, version = register_champion(
    sk_model,
    params={
        "model": selection.model_name,
        "cohort": selection.cohort,
        "seed": config.SEED,
        "threshold": selection.threshold,
        "criterion": selection.criterion,
        "passed_calibration_gate": selection.passed_calibration_gate,
    },
    metrics={f"test_{k}": v for k, v in test_metrics.items()},
    pip_requirements=pip_by_family.get(selection.model_name, ["scikit-learn"]),
)
print(f"registered rp-champion v{version} (run {run_id[:8]}) -> Production")"""
    ),
    md(
        "## 14 — Summary\n\n"
        "The five Epic 3 lenses converged into one gated-rank decision. The "
        "champion is selected on validation, confirmed once on test, persisted "
        "to `reports/models/champion.pkl`, and registered as "
        "`rp-champion/Production`.\n\n"
        "**Reproduce:** `make evaluate` regenerates and re-executes this "
        "notebook end to end. **Inspect the registry:** `make mlflow-ui`, then "
        "the *Models* tab → `rp-champion` → the *Production* version.\n\n"
        "Loop 4 loads the champion with "
        "`retention.evaluation.champion.load_champion()` (file store) or "
        "`mlflow.sklearn.load_model('models:/rp-champion/Production')` "
        "(registry) — same model, two access paths."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Generated {NB_PATH}")
