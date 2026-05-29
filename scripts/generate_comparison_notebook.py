"""Generate notebooks/02_model_comparison.ipynb (Story 2.5).

6-cell comparison table: LR × hris_only, LR × hybrid, GBM × hris_only,
GBM × hybrid, EBM × hris_only, EBM × hybrid.

Metrics: AUC-PR (primary), AUC-ROC, precision@10%, precision@20%, recall@10%, Brier.

Run with:
    uv run python scripts/generate_comparison_notebook.py
Then execute the notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NB_PATH = Path("notebooks/02_model_comparison.ipynb")


def md(source: str):  # noqa: ANN201
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):  # noqa: ANN201
    return nbformat.v4.new_code_cell(source)


cells = [
    md(
        "# Story 2.5 — Three-Model × Two-Cohort Comparison\n\n"
        "**Loop 2 deliverable.** Trains LR, GBM (XGBoost), and EBM on both\n"
        "cohorts (`hris_only` and `hybrid`) using the temporal train/val split,\n"
        "then evaluates each on the held-out **validation** set.\n\n"
        "Metrics reported per model × cohort:\n"
        "- **AUC-PR** (primary) — invariant to class imbalance\n"
        "- **AUC-ROC** (secondary) — context, but optimistic under imbalance\n"
        "- **Precision@10%** — operational: of the top 10% flagged, how many exit?\n"
        "- **Precision@20%** — operational: of the top 20% flagged, how many exit?\n"
        "- **Recall@10%** — FLIP-RISK: what fraction of real exits are in the top 10%?\n"
        "- **Brier** — calibration: mean squared error of predicted probabilities\n\n"
        "> **Test set is held out.** This notebook uses validation scores only.\n"
        "> Final test-set evaluation runs after threshold selection in Epic 3."
    ),
    md("## 1 — Setup"),
    code(
        "from __future__ import annotations\n"
        "\n"
        "import warnings\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from retention import config\n"
        "from retention.data.load import load_attrition_features\n"
        "from retention.data.split import temporal_split\n"
        "from retention.evaluation.metrics import (\n"
        "    auc_pr,\n"
        "    auc_roc,\n"
        "    brier_score,\n"
        "    format_rung1_caption,\n"
        "    precision_at_k,\n"
        "    recall_at_k,\n"
        ")\n"
        "from retention.features.catalog import get_label_name\n"
        "from retention.features.cohorts import extract_X_y, get_cohort_feature_names, split_cohorts\n"
        "from retention.models.ebm import train_ebm\n"
        "from retention.models.lr import train_lr\n"
        "from retention.models.tracking import log_run\n"
        "from retention.models.xgb import RetentionModel\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "LABEL = get_label_name()   # 'voluntary_exit_label'\n"
        "print(f'SEED={config.SEED}  label={LABEL}')"
    ),
    md("## 2 — Data Loading & Temporal Split"),
    code(
        "# config.DATA_DIR resolves from the package location (absolute), so the\n"
        "# load works regardless of cwd — nbconvert runs with cwd=notebooks/.\n"
        "df = load_attrition_features(snapshot_dir=config.DATA_DIR / 'raw')\n"
        "print(f'Loaded {len(df):,} rows × {df.shape[1]} cols')\n"
        "\n"
        "train_df, val_df, test_df = temporal_split(df)\n"
        "print(\n"
        "    f'Train: {len(train_df):,} rows  '\n"
        "    f'Val: {len(val_df):,} rows  '\n"
        "    f'Test: {len(test_df):,} rows'\n"
        ")\n"
        "print(f'Val base rate: {val_df[LABEL].mean():.1%}')"
    ),
    md(
        "## 3 — Cohort Splitting\n\n"
        "Both cohorts use **identical row indices** — only the column set differs.\n"
        "`hris_only` has 7 HRIS feature columns; `hybrid` adds 3 survey columns."
    ),
    code(
        "train_cohorts = split_cohorts(train_df)\n"
        "val_cohorts   = split_cohorts(val_df)\n"
        "\n"
        "for cohort_name, cdf in train_cohorts.items():\n"
        "    n_features = len(get_cohort_feature_names(cohort_name))\n"
        "    print(f'  {cohort_name}: train={len(cdf):,} rows, {n_features} features')"
    ),
    md("## 4 — Train All 6 Models"),
    code(
        "# extract_X_y is defined in src/retention/features/cohorts.py\n"
        "# (not inline here — notebook-defined functions bypass mypy + pytest)\n"
        "models = {}\n"
        "for cohort in ('hris_only', 'hybrid'):\n"
        "    X_tr, y_tr = extract_X_y(train_cohorts[cohort], cohort)\n"
        "\n"
        "    # LR\n"
        "    models[('LR', cohort)] = train_lr(X_tr, y_tr, cohort=cohort)\n"
        "\n"
        "    # GBM (XGBoost)\n"
        "    gbm = RetentionModel(cohort=cohort)\n"
        "    gbm.fit(X_tr, y_tr)\n"
        "    models[('GBM', cohort)] = gbm\n"
        "\n"
        "    # EBM\n"
        "    models[('EBM', cohort)] = train_ebm(X_tr, y_tr, cohort=cohort)\n"
        "\n"
        "    print(f'[{cohort}] LR, GBM, EBM trained.')"
    ),
    md("## 5 — Evaluate on Validation Set"),
    code(
        "# Default hyperparameters for each model — mirrors src/retention/models/*.py defaults.\n"
        "# Logged to MLflow so the UI shows the exact settings behind each score.\n"
        "# MLflow metric keys must match [a-zA-Z0-9._\\-/ ]+ so 'Prec@10%' → 'prec_at_10'.\n"
        "MODEL_PARAMS = {\n"
        "    'LR':  {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000, 'solver': 'lbfgs'},\n"
        "    'GBM': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,\n"
        "            'eval_metric': 'aucpr'},\n"
        "    'EBM': {'interactions': 10, 'max_bins': 256, 'outer_bags': 8},\n"
        "}\n"
        "\n"
        "rows = []\n"
        "for (model_name, cohort), pipeline in models.items():\n"
        "    X_val, y_val = extract_X_y(val_cohorts[cohort], cohort)\n"
        "    proba = pipeline.predict_proba(X_val)[:, 1]\n"
        "\n"
        "    # Compute metrics — display keys stay pretty, mlflow keys are safe identifiers\n"
        "    ap   = round(auc_pr(y_val, proba), 3)\n"
        "    ar   = round(auc_roc(y_val, proba), 3)\n"
        "    p10  = round(precision_at_k(y_val, proba, k=0.10), 3)\n"
        "    p20  = round(precision_at_k(y_val, proba, k=0.20), 3)\n"
        "    r10  = round(recall_at_k(y_val, proba, k=0.10), 3)\n"
        "    bs   = round(brier_score(y_val, proba), 3)\n"
        "\n"
        "    rows.append({'Model': model_name, 'Cohort': cohort,\n"
        "                 'AUC-PR': ap, 'AUC-ROC': ar, 'Prec@10%': p10,\n"
        "                 'Prec@20%': p20, 'Rec@10%': r10, 'Brier': bs})\n"
        "\n"
        "    run_id = log_run(\n"
        "        run_name=f'{model_name}_{cohort}',\n"
        "        params={'model': model_name, 'cohort': cohort, 'seed': config.SEED,\n"
        "                **MODEL_PARAMS[model_name]},\n"
        "        metrics={'auc_pr': ap, 'auc_roc': ar, 'prec_at_10': p10,\n"
        "                 'prec_at_20': p20, 'rec_at_10': r10, 'brier': bs},\n"
        "    )\n"
        "    print(f'  [{model_name}/{cohort}] AUC-PR={ap:.3f}  run_id={run_id[:8]}…')\n"
        "\n"
        "results = pd.DataFrame(rows).set_index(['Model', 'Cohort'])\n"
        "results"
    ),
    md(
        "## 6 — Comparison Table\n\n"
        "Primary metric: **AUC-PR**. No-skill baseline ≈ val base rate.\n"
        "All values on the **validation** set (test set held out until Epic 3)."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Styled table for the notebook\n"
        "styled = (\n"
        "    results.style\n"
        "    .highlight_max(subset=['AUC-PR', 'AUC-ROC', 'Prec@10%', 'Prec@20%', 'Rec@10%'],\n"
        "                   color='#d4edda', axis=0)\n"
        "    .highlight_min(subset=['Brier'], color='#d4edda', axis=0)\n"
        "    .format('{:.3f}')\n"
        "    .set_caption('Loop 2 — Val-Set Comparison (green = best per metric)')\n"
        ")\n"
        "styled"
    ),
    code(
        "# AUC-PR bar chart — primary metric\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "results['AUC-PR'].plot(\n"
        "    kind='bar', ax=ax, color=['#2166ac', '#4dac26', '#d01c8b'] * 2,\n"
        "    width=0.6, edgecolor='white', linewidth=0.8\n"
        ")\n"
        "base_rate = val_df[LABEL].mean()\n"
        "ax.axhline(base_rate, color='grey', linestyle='--', linewidth=1,\n"
        "           label=f'No-skill baseline ({base_rate:.1%} base rate)')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('AUC-PR')\n"
        "ax.set_title('Loop 2 — AUC-PR by Model × Cohort (validation set)')\n"
        "ax.legend(fontsize=9)\n"
        "ax.set_ylim(0, 1)\n"
        "for bar in ax.patches:\n"
        "    h = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,\n"
        "            f'{h:.3f}', ha='center', va='bottom', fontsize=8)\n"
        "plt.xticks(rotation=30, ha='right')\n"
        "plt.tight_layout()\n"
        "fig_path = config.REPORTS_DIR / 'figures' / 'loop2_comparison_auc_pr.png'\n"
        "plt.savefig(fig_path, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Saved: reports/figures/loop2_comparison_auc_pr.png')"
    ),
    code(
        "# Print Rung 1 captioned winner\n"
        "best_idx = results['AUC-PR'].idxmax()\n"
        "best_val = results.loc[best_idx, 'AUC-PR']\n"
        "print(f'Best validation model: {best_idx[0]} / {best_idx[1]}')\n"
        "print(format_rung1_caption(best_val))\n"
        "print()\n"
        "print('Full results table:')\n"
        "print(results.to_string())"
    ),
    md(
        "## 7 — Methodology Note\n\n"
        "**OHE tradeoff:** LR and GBM share `build_preprocessor()` which\n"
        "one-hot-encodes categorical columns (`performance_tier`, `gender`).\n"
        "EBM uses `build_ebm_preprocessor()` (imputation only) and receives\n"
        "raw string columns for native categorical bin detection.\n\n"
        "This means the three models are not on strictly equal preprocessing\n"
        "footing — an acknowledged tradeoff, not a hidden one. EBM's native\n"
        "handling is its natural operating mode; applying OHE to EBM would\n"
        "degrade its interpretability advantage by fragmenting category levels\n"
        "into disconnected binary features.\n\n"
        "Full treatment: `docs/methodology.md → Cross-Model Comparison Methodology`\n"
        "(written, Story 2.5.6).\n\n"
        "**Imbalance handling summary:**\n"
        "| Model | Strategy |\n"
        "|-------|----------|\n"
        "| LR    | `class_weight='balanced'` |\n"
        "| GBM   | `eval_metric='aucpr'` (default); `scale_pos_weight` available |\n"
        "| EBM   | `compute_sample_weight('balanced')` passed to `fit()` |"
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Generated {NB_PATH}")
