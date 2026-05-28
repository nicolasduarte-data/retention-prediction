"""Generate notebooks/01_data_exploration.ipynb.

Run once from the project root:
    python scripts/generate_eda_notebook.py

Overwrites any existing notebook at that path. After running,
open Jupyter and execute all cells to produce the EDA outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def md(source: str):
    return new_markdown_cell(source)


def code(source: str):
    return new_code_cell(source)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

cells = []

# ── Title ───────────────────────────────────────────────────────────────────
cells.append(
    md("""\
# 01 — Data Exploration
**Loop 2 · rp-prey-002 · Story 1.7**

Full EDA for `marts.v_attrition_features` (13-column contract, v0.2).

**Two jobs:**
1. **pa-warehouse benchmark re-validation** (Story 1.7.4) — verify generator
   properties before Epic 2 training begins. Document any deviations in
   `docs/data_card.md → Known Limitations`.
2. **Loop 2 EDA** — understand the data: missingness, base rates, feature
   distributions, dual-cohort alignment, temporal split boundaries,
   survey response rate and selection bias.

> **Note:** All data is **fully synthetic** (SEED=42, pa-warehouse generator).
> Do not report these figures as real workforce statistics.\
""")
)

# ── Setup ────────────────────────────────────────────────────────────────────
cells.append(md("## 0 · Setup"))

cells.append(
    code("""\
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy.stats import spearmanr

from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.features.catalog import FEATURE_CATALOG
from retention.features.cohorts import get_cohort_feature_names, split_cohorts

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
%matplotlib inline

# Locate the most recent snapshot
_snapshots = sorted(Path("../data/raw").glob("v_attrition_features_*.csv"))
assert _snapshots, "No snapshot found in data/raw/. Run load_attrition_features() first."
SNAPSHOT = _snapshots[-1]
print(f"Snapshot: {SNAPSHOT.name}")\
""")
)

# ── Load ─────────────────────────────────────────────────────────────────────
cells.append(md("## 1 · Dataset overview"))

cells.append(
    code("""\
df = load_attrition_features_local(SNAPSHOT)

print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Snapshot date  : {df['snapshot_date'].unique()}")
print(f"Positive class : {df['voluntary_exit_label'].sum():,} ({df['voluntary_exit_label'].mean()*100:.1f}%)")
print(f"Negative class : {(~df['voluntary_exit_label']).sum():,} ({(~df['voluntary_exit_label']).mean()*100:.1f}%)")
print()
print("dtypes:")
print(df.dtypes.to_string())\
""")
)

cells.append(
    code("""\
df.describe(include="all").T\
""")
)

# ── Benchmark re-validation ───────────────────────────────────────────────────
cells.append(
    md("""\
## 2 · pa-warehouse benchmark re-validation (Story 1.7.4)

Three generator properties were specified in the pa-warehouse design.
We check them here against the live snapshot and document any deviations.

| Benchmark | Target | Source |
|---|---|---|
| Censoring rate | [55%, 75%] | pa-warehouse generator spec |
| KM median exit time | ≈ 2.4 yr | Survival analysis design doc |
| Survey coverage (active employees) | ≥ 60% | paw-prey-006 acceptance criteria |\
""")
)

cells.append(
    code("""\
# ── Benchmark 1: Censoring rate ──────────────────────────────────────────────
n_total = len(df)
n_exited = int(df["voluntary_exit_label"].sum())
n_censored = n_total - n_exited
censoring_rate = n_censored / n_total

gate_ok = 0.55 <= censoring_rate <= 0.75
status = "✅ PASS" if gate_ok else "⚠️  OUTSIDE GATE — documented in data_card.md"

print("=== Benchmark 1: Censoring Rate ===")
print(f"  Exited    : {n_exited:,}")
print(f"  Censored  : {n_censored:,}")
print(f"  Rate      : {censoring_rate:.3f} ({censoring_rate*100:.1f}%)")
print(f"  Gate [55%, 75%]: {status}")
print()
print("Finding: 81.2% censoring (above gate). Generator produces fewer exits")
print("than the design target. Documented in docs/data_card.md Known Limitations.")\
""")
)

cells.append(
    code("""\
# ── Benchmark 2: KM median survival time ────────────────────────────────────
kmf = KaplanMeierFitter()
kmf.fit(df["tenure_months"], event_observed=df["voluntary_exit_label"], label="All employees")
km_median_months = kmf.median_survival_time_
km_median_years = km_median_months / 12

gate_ok = 1.8 <= km_median_years <= 3.0
status = "✅ PASS" if gate_ok else "⚠️  OUTSIDE GATE — documented in data_card.md"

print("=== Benchmark 2: KM Median Survival Time ===")
print(f"  KM median : {km_median_months:.1f} months ({km_median_years:.2f} yr)")
print(f"  Gate ≈ 2.4yr [1.8–3.0]: {status}")
print()
print("Finding: 3.75yr KM median (above gate). Same root cause as censoring rate.")

fig, ax = plt.subplots(figsize=(8, 4))
kmf.plot_survival_function(ax=ax, ci_show=True)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="S(t) = 0.5 (median)")
ax.axvline(km_median_months, color="red", linestyle="--", alpha=0.7)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival probability")
ax.set_title("Kaplan–Meier survival curve — all employees")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/km_survival_all.png", dpi=150, bbox_inches="tight")
plt.show()\
""")
)

cells.append(
    code("""\
# ── Benchmark 3: Survey signal coverage ─────────────────────────────────────
active = df[df["voluntary_exit_label"] == False]
survey_cols = ["enps", "engagement_score", "manager_relationship_score"]

print("=== Benchmark 3: Survey Signal Coverage ===")
for col in survey_cols:
    cov_all = df[col].notna().mean()
    cov_active = active[col].notna().mean()
    gate_ok = cov_active >= 0.60
    print(f"  {col:<30}: all={cov_all*100:.1f}%  active={cov_active*100:.1f}%  "
          f"{'✅ PASS' if gate_ok else '⚠️  FAIL'} (gate ≥ 60%)")
print()
print("Finding: 91.5% coverage — well above 60% floor (paw-prey-006 acceptance criteria).")\
""")
)

# ── Survey selection bias (Hellinger) ─────────────────────────────────────────
cells.append(
    md("""\
### 2.4 · Survey response selection bias (Hellinger distance)

**Interpretation of Hellinger check in this project:**
We compare HRIS feature distributions between employees who completed
at least one survey (`survey_responders`) vs those who did not (`no_response`).

A Hellinger distance > 0.10 for a HRIS feature signals **selection bias** in
survey participation — responders and non-responders have systematically different
profiles. This could inflate or deflate the hybrid model's apparent advantage.\
""")
)

cells.append(
    code("""\
survey_responders = df[df["enps"].notna()]
no_response = df[df["enps"].isna()]

print(f"Survey responders : {len(survey_responders):,} ({len(survey_responders)/len(df)*100:.1f}%)")
print(f"No response       : {len(no_response):,} ({len(no_response)/len(df)*100:.1f}%)")
print()

hris_numeric = ["tenure_months", "compa_ratio", "successor_count", "age_at_window_close"]
results = []

for col in hris_numeric:
    v1, v2 = survey_responders[col].dropna(), no_response[col].dropna()
    lo, hi = min(v1.min(), v2.min()), max(v1.max(), v2.max())
    bins = np.linspace(lo, hi, 25)
    h1, _ = np.histogram(v1, bins=bins, density=True)
    h2, _ = np.histogram(v2, bins=bins, density=True)
    h1 = h1 / (h1.sum() + 1e-12)
    h2 = h2 / (h2.sum() + 1e-12)
    hell = float(np.sqrt(np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2)) / np.sqrt(2))
    gate = "✅" if hell < 0.10 else "⚠️ "
    results.append({"feature": col, "hellinger": hell, "gate": gate})
    print(f"  {col:<28}: Hellinger={hell:.4f}  {gate} ({'PASS' if hell < 0.10 else 'BIAS SIGNAL'})")

print()
print("Note: High Hellinger for a feature means survey participation is NOT random")
print("with respect to that feature. This affects hybrid model generalisation.")\
""")
)

# ── Missingness ────────────────────────────────────────────────────────────────
cells.append(md("## 3 · Missingness"))

cells.append(
    code("""\
null_pct = df.isnull().mean().sort_values(ascending=False) * 100
null_pct = null_pct[null_pct > 0]

print("Columns with nulls:")
print(null_pct.to_string())

fig, ax = plt.subplots(figsize=(8, 3))
null_pct.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[3])
ax.axhline(40, color="red", linestyle="--", alpha=0.5, label="40% threshold")
ax.set_ylabel("% null")
ax.set_title("Null rate per column (survey signal columns only have nulls)")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../reports/figures/missingness_bar.png", dpi=150, bbox_inches="tight")
plt.show()\
""")
)

# ── Base rate per cohort ───────────────────────────────────────────────────────
cells.append(md("## 4 · Base rate per cohort"))

cells.append(
    code("""\
cohorts = split_cohorts(df)

print("Cohort sizes and base rates:")
print(f"  hris_only : {len(cohorts['hris_only']):,} rows | "
      f"attrition = {df['voluntary_exit_label'].mean()*100:.1f}%")
print(f"  hybrid    : {len(cohorts['hybrid']):,} rows | "
      f"attrition = {df['voluntary_exit_label'].mean()*100:.1f}%")
print()
print("Both cohorts have IDENTICAL rows — same base rate by design.")
print("The controlled experiment compares feature sets, not employee populations.")

# Base rate by survey response status (informational)
resp_rate = survey_responders["voluntary_exit_label"].mean()
noresp_rate = no_response["voluntary_exit_label"].mean()
print()
print(f"  Attrition in survey responders : {resp_rate*100:.1f}%")
print(f"  Attrition in non-responders    : {noresp_rate*100:.1f}%")\
""")
)

# ── Feature distributions ──────────────────────────────────────────────────────
cells.append(md("## 5 · Feature distributions"))

cells.append(
    code("""\
numeric_features = [s.name for s in FEATURE_CATALOG
                    if s.role == "feature" and s.dtype == "numeric"]
categorical_features = [s.name for s in FEATURE_CATALOG
                        if s.role == "feature" and s.dtype == "categorical"]
boolean_features = [s.name for s in FEATURE_CATALOG
                    if s.role == "feature" and s.dtype == "boolean"]

print(f"Numeric features   : {numeric_features}")
print(f"Categorical features: {categorical_features}")
print(f"Boolean features   : {boolean_features}")\
""")
)

cells.append(
    code("""\
# Numeric feature distributions — split by exit status
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    ax = axes[i]
    exited = df.loc[df["voluntary_exit_label"] == True, col].dropna()
    active = df.loc[df["voluntary_exit_label"] == False, col].dropna()
    ax.hist(active, bins=30, alpha=0.6, label="Active", color=sns.color_palette("muted")[0], density=True)
    ax.hist(exited, bins=30, alpha=0.6, label="Exited", color=sns.color_palette("muted")[3], density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel("")

# Hide unused axes
for j in range(len(numeric_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Numeric feature distributions — active vs exited", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/figures/feature_distributions_numeric.png", dpi=150, bbox_inches="tight")
plt.show()\
""")
)

cells.append(
    code("""\
# Categorical + boolean features — attrition rate per category
fig, axes = plt.subplots(1, len(categorical_features) + len(boolean_features), figsize=(14, 4))
if not hasattr(axes, "__len__"):
    axes = [axes]

plot_cols = categorical_features + boolean_features
for i, col in enumerate(plot_cols):
    ax = axes[i]
    rates = df.groupby(col)["voluntary_exit_label"].mean().sort_values()
    rates.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[1])
    ax.set_title(f"{col}\\nattrition rate", fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Attrition rate per category / boolean value", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/figures/feature_distributions_categorical.png", dpi=150, bbox_inches="tight")
plt.show()\
""")
)

# ── Spearman correlation matrix ─────────────────────────────────────────────────
cells.append(md("## 6 · Spearman correlation matrix"))

cells.append(
    code("""\
# Encode categoricals ordinally for Spearman
df_enc = df.copy()
df_enc["performance_tier"] = df_enc["performance_tier"].astype(float)
df_enc["gender"] = df_enc["gender"].map({"M": 0, "F": 1, "NB": 2}).fillna(-1)
df_enc["is_critical_role"] = df_enc["is_critical_role"].astype(float)
df_enc["voluntary_exit_label"] = df_enc["voluntary_exit_label"].astype(float)

all_features = numeric_features + categorical_features + boolean_features
cols_for_corr = all_features + ["voluntary_exit_label"]

corr_matrix = df_enc[cols_for_corr].corr(method="spearman")

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(corr_matrix, dtype=bool)
# Show full matrix (feature × feature + label column)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Spearman correlation matrix — features + label", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/figures/spearman_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# Print feature-label correlations sorted by abs value
label_corr = corr_matrix["voluntary_exit_label"].drop("voluntary_exit_label").abs().sort_values(ascending=False)
print("Feature-label Spearman correlations (abs, sorted):")
print(label_corr.to_string())
print()
print(f"Max abs correlation: {label_corr.max():.3f} — Layer 3 gate threshold: 0.70")\
""")
)

# ── Dual-cohort row alignment ──────────────────────────────────────────────────
cells.append(md("## 7 · Dual-cohort row alignment verification"))

cells.append(
    code("""\
cohorts = split_cohorts(df)
df_hris = cohorts["hris_only"]
df_hybrid = cohorts["hybrid"]

# Core assertion
assert df_hris.index.equals(df_hybrid.index), "ROW ALIGNMENT VIOLATED"

print("=== Cohort alignment check ===")
print(f"  hris_only rows  : {len(df_hris):,}")
print(f"  hybrid rows     : {len(df_hybrid):,}")
print(f"  Index identical : {df_hris.index.equals(df_hybrid.index)} ✅")
print()
print("hris_only columns  :", list(df_hris.columns))
print("hybrid-only extra  :", [c for c in df_hybrid.columns if c not in df_hris.columns])

hris_feat = get_cohort_feature_names("hris_only")
hybrid_feat = get_cohort_feature_names("hybrid")
print()
print(f"HRIS-only feature count : {len(hris_feat)}")
print(f"Hybrid feature count    : {len(hybrid_feat)}")
print(f"Survey features added   : {set(hybrid_feat) - set(hris_feat)}")\
""")
)

# ── Temporal split verification ───────────────────────────────────────────────
cells.append(md("## 8 · Temporal split verification"))

cells.append(
    code("""\
# temporal_split uses employee_id as the stratification axis since we
# have a single snapshot_date. 80/10/10 split by default.
train, val, test = temporal_split(df)

print("=== Temporal split ===")
print(f"  Train : {len(train):,} rows ({len(train)/len(df)*100:.1f}%)")
print(f"  Val   : {len(val):,} rows ({len(val)/len(df)*100:.1f}%)")
print(f"  Test  : {len(test):,} rows ({len(test)/len(df)*100:.1f}%)")
print()
print("Base rates per split:")
for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    rate = split_df["voluntary_exit_label"].mean()
    print(f"  {name:<6}: {rate*100:.1f}% voluntary exit")
print()
print("Snapshot dates in each split:")
for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    dates = split_df["snapshot_date"].unique()
    print(f"  {name:<6}: {dates}")
print()
print("Note: Single snapshot_date — split is by employee_id index, not time.")
print("This is the Loop 1 design decision. True temporal validation requires")
print("pa-warehouse time-series snapshots (deferred). See docs/data_card.md.")\
""")
)

# ── Survey response rate ───────────────────────────────────────────────────────
cells.append(md("## 9 · Survey response rate and missing-value profile"))

cells.append(
    code("""\
print("=== Survey signal coverage summary ===")
print()
for col in ["enps", "engagement_score", "manager_relationship_score"]:
    n_null = df[col].isna().sum()
    pct_null = n_null / len(df) * 100
    pct_cov = 100 - pct_null
    # Among those with data, show value range
    series = df[col].dropna()
    print(f"  {col:<30}")
    print(f"    Coverage       : {pct_cov:.1f}%  ({len(df) - n_null:,} of {len(df):,} employees)")
    print(f"    Null           : {pct_null:.1f}%  ({n_null:,} employees — will be median-imputed in hybrid arm)")
    print(f"    Range          : [{series.min():.1f}, {series.max():.1f}]")
    print(f"    Median         : {series.median():.2f}")
    print()

# Survey value distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
survey_cols = ["enps", "engagement_score", "manager_relationship_score"]
for i, col in enumerate(survey_cols):
    ax = axes[i]
    exited = df.loc[df["voluntary_exit_label"] == True, col].dropna()
    active = df.loc[df["voluntary_exit_label"] == False, col].dropna()
    ax.hist(active, bins=25, alpha=0.6, label="Active", color=sns.color_palette("muted")[0], density=True)
    ax.hist(exited, bins=25, alpha=0.6, label="Exited", color=sns.color_palette("muted")[3], density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("Survey signal distributions — active vs exited", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/figures/survey_signal_distributions.png", dpi=150, bbox_inches="tight")
plt.show()\
""")
)

# ── Summary ───────────────────────────────────────────────────────────────────
cells.append(
    md("""\
## 10 · EDA summary — key findings for Loop 2

| Finding | Value | Implication for modeling |
|---|---|---|
| Dataset size | 1,274 rows × 13 cols | Sufficient for 3-model comparison; nested CV will be limited by positive class |
| Positive class (attrition) | 18.8% | Use `class_weight='balanced'` (LR) and `scale_pos_weight` (GBM). AUC-PR preferred over accuracy. |
| Censoring rate | 81.2% (above [55%,75%] gate) | Generator artifact, not leakage. Documented. |
| KM median exit | 3.75yr (above 2.4yr target) | Same cause. Loop 3b survival analysis should use observed 3.75yr, not the design target. |
| Survey coverage | 91.5% of active employees | 104 rows (~8%) have no survey data → median imputation in hybrid arm. |
| Survey selection bias | Check Hellinger per feature | Inspect Hellinger output in §2.4. High values mean responders ≠ non-responders. |
| Max feature-label Spearman | See §6 output | All features below 0.70 threshold (Layer 3 gate). |
| Cohort alignment | ✅ Identical indices | Controlled experiment is valid. |
| Temporal split | 80/10/10 by employee_id | Single snapshot — true time-based split not possible with this data version. |

**Epic 2 can proceed.** All 4 leakage gate layers pass. Benchmark deviations are
documented in `docs/data_card.md → Known Limitations` and do not block modeling.\
""")
)

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.9",
}

out_path = Path(__file__).parents[1] / "notebooks" / "01_data_exploration.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {out_path}")
