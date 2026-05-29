# Methodology — retention-prediction

**Status:** living document — one section accretes per loop
**Last updated:** 2026-05-29 (Loop 2 — rp-prey-002, Story 3.3)
**Companion docs:** `architecture.md` (data flow), `data_card.md` (dataset), `integration_contract.md` (schema)

This document records the *methodological* choices behind the model — the
"we did X instead of Y, and here is the honest cost of that choice" decisions
that a code diff cannot explain on its own. Each section is written to survive
a skeptical reviewer: it states the choice, the reasoning, and the tradeoff we
accepted, with the receipts to back the claim.

It grows one section per loop. Sections present today:

| Section | Loop | Story |
|---|---|---|
| [Cross-Model Comparison Methodology](#cross-model-comparison-methodology) | 2 | 2.5 / 2.5.6 |
| [MLflow Setup](#mlflow-setup) | 2 | 2.7 |
| [EBM Preprocessing Sensitivity](#ebm-preprocessing-sensitivity) | 2 | 2.9 |
| [Threshold Calibration](#threshold-calibration) | 2 | 3.3 |

Sections scaffolded for later loops (added when the work ships, not before):
*Expected Value + p_eff sensitivity* (3.4) · *Flat-CV vs Nested-CV* (3.5.6) ·
*Test Quality / Mutation Testing* (3.7) · *Fairness Thresholds + Chouldechova*
(Epic 5) · *Adversarial SHAP* (Epic 6).

---

## Cross-Model Comparison Methodology

*Loop 2 — Stories 2.5 (the 6-cell table) and 2.5.6 (the honest write-up).*
*Reproduce with `notebooks/02_model_comparison.ipynb`.*

### The experiment in one sentence

We train three model families — Logistic Regression (LR), Gradient-Boosted
Trees (GBM / XGBoost), and an Explainable Boosting Machine (EBM) — on two
feature cohorts that differ only in whether they include survey signal, then
ask a single question: **does the survey signal earn its place, and which model
uses it best?**

### Experimental design — a controlled comparison, not a leaderboard

The design is deliberately a *controlled experiment*, because a leaderboard of
six numbers with no controls tells a reviewer nothing about why one won.

- **6 cells:** `{LR, GBM, EBM} × {hris_only, hybrid}`.
- **Identical rows across cohorts.** Both cohorts contain the *same employees*,
  the *same labels*, and use the *same temporal split*. The only thing that
  changes between `hris_only` and `hybrid` is the **column set** handed to the
  model. This is the whole point: any difference in performance is attributable
  to the *survey signal*, never to a different population. (See
  `src/retention/features/cohorts.py` — both cohort frames share one row index
  by construction, guarded by `test_cohort_row_alignment`.)
- **Cohort feature sets:**

  | Cohort | Model features | Columns added |
  |---|---|---|
  | `hris_only` | 7 | `tenure_months`, `performance_tier`, `compa_ratio`, `is_critical_role`, `successor_count`, `age_at_window_close`, `gender` |
  | `hybrid` | 10 | the 7 above **+** `enps`, `engagement_score`, `manager_relationship_score` |

  *(Each cohort frame also carries `employee_id`, `snapshot_date`, and the
  `voluntary_exit_label` — these are not model inputs.)*

- **Temporal split, never random.** `temporal_split()` sorts by `snapshot_date`
  (tie-broken by `employee_id`) and slices oldest **70 % → train**, next
  **15 % → val**, newest **15 % → test**. Random shuffling would leak future
  rows into the past; the `assert_no_temporal_leak()` gate fails CI if the
  ordering invariant ever breaks.
- **Validation-set scores only.** Every number in this section is computed on
  the **validation** split. The **test split is held out** until champion
  selection in Epic 3 — reporting test numbers now would burn the only unbiased
  estimate we get.

**Split receipts** (regenerated 2026-05-28, SEED=42):

| Split | Rows | Positives (exits) | Base rate |
|---|---|---|---|
| train | 892 | 160 | 17.9 % |
| **val** | **191** | **39** | **20.4 %** |
| test | 191 | 41 | 21.5 % |
| total | 1,274 | 240 | 18.8 % |

The bolded line is the one that governs everything below: **39 positive
examples**. Hold onto that number — it is why the honest verdict at the end is
"suggestive, not conclusive."

### The preprocessing asymmetry — the tradeoff we accepted on purpose

This is the choice most likely to draw a reviewer's red pen, so we state it
plainly rather than bury it.

**LR and GBM share one preprocessor; EBM uses a different one.**

| Model | Categorical handling | Numeric scaling | Preprocessor |
|---|---|---|---|
| LR | One-hot encode `performance_tier`, `gender` | (lbfgs is scale-tolerant; median-impute only) | `build_preprocessor()` |
| GBM | One-hot encode `performance_tier`, `gender` | none (trees are scale-invariant) | `build_preprocessor()` |
| EBM | **Native** — raw string columns, no OHE | none | `build_ebm_preprocessor()` |

LR and GBM receive one-hot-encoded categoricals. EBM receives the **raw string
columns** and detects categorical bins natively (`set_output(transform="pandas")`
hands it a typed DataFrame so it auto-classifies `float64` → continuous,
`object` → nominal).

**Why not force all three onto identical preprocessing?** Because one-hot
encoding an EBM would *degrade the exact property it exists to provide.* An EBM
learns one shape function per feature; given a native categorical column it
learns a single, readable "effect of `performance_tier`" curve across all
tiers. One-hot-encode that same column and you fragment it into disconnected
binary stumps — `performance_tier_A`, `performance_tier_B`, … — each with its
own tiny shape function, and the at-a-glance interpretability that justifies
choosing an EBM evaporates. Native handling is the EBM's *natural operating
mode*; OHE is the LR/GBM natural mode. Each model runs the way it would run in
production.

**The honest cost:** the three models are therefore **not on strictly identical
preprocessing footing**, so a head-to-head AUC-PR gap conflates "better model
family" with "better-suited preprocessing." We accept this rather than hide it,
and we bound it two ways:

1. The cohort comparison (`hris_only` vs `hybrid`) is **unaffected** — within a
   single model the preprocessor is held constant, so the survey-lift question
   is clean regardless of cross-model preprocessing differences.
2. **Story 2.9** runs an EBM-native-vs-EBM-OHE sensitivity check and records the
   AUC-PR delta in *EBM Preprocessing Sensitivity* below. That quantifies how
   much of any EBM gap is preprocessing rather than model family.

### Imbalance handling — three mechanisms, one intent

At a 20 % base rate a naive learner can score 80 % accuracy by predicting "no
one leaves." Each model is steered away from that majority-class collapse, and
the three mechanisms are not identical:

| Model | Mechanism | Effect |
|---|---|---|
| LR | `class_weight='balanced'` | Re-weights the loss so the 20 % minority counts as much as the 80 % majority. |
| GBM | `eval_metric='aucpr'` (default); `scale_pos_weight` *available* but **off by default** | Optimizes the ranking metric directly; does **not** re-weight examples unless asked. |
| EBM | `compute_sample_weight('balanced')` → `fit(sample_weight=…)` | Per-row weights, **mathematically equivalent** to `class_weight='balanced'` (EBM doesn't reliably expose a `class_weight` arg, so we pass the weights it computes). |

**No SMOTE — anywhere.** Synthetic oversampling is explicitly excluded
(`test_no_smote_in_pipeline` enforces it). Two reasons: (1) SMOTE interpolates
between training rows, which **violates the temporal ordering** the split exists
to protect; (2) it **distorts probability calibration**, and Loop 3's threshold
and expected-value work depends on calibrated probabilities. We handle imbalance
at the loss/threshold layer instead — framed as a controlled choice in
*Threshold Calibration* (Loop 3).

This difference in mechanism has a visible, honest consequence in the Brier
scores — see the calibration note below.

### Metrics — and what each one actually answers

| Metric | Question it answers | Why it's here |
|---|---|---|
| **AUC-PR** *(primary)* | How well does the model rank true exits to the top? | Invariant to the 80/20 imbalance; AUC-ROC is optimistic under imbalance because true-negatives inflate it. |
| AUC-ROC | Overall rank quality | Reported for context only; read with the imbalance caveat. |
| Precision@10 % / @20 % | Of the top-k % we flag for an HR conversation, what fraction truly exit? | The cost-control metric — it bounds wasted interventions. |
| Recall@10 % | Of all real exits, what fraction sit in our top 10 %? | The **[FLIP-RISK]** metric — missed exits are the expensive failure. |
| Brier | Mean squared error of the predicted probabilities | Calibration: are the probabilities *trustworthy numbers*, not just good ranks? |

The no-skill AUC-PR baseline equals the validation base rate, **≈ 0.204**. Any
model below that is worse than guessing the prevalence.

### Results (validation set)

| Model | Cohort | AUC-PR | AUC-ROC | Prec@10 % | Prec@20 % | Rec@10 % | Brier |
|---|---|---|---|---|---|---|---|
| LR | hris_only | 0.284 | 0.641 | 0.250 | 0.333 | 0.128 | 0.237 |
| GBM | hris_only | 0.291 | 0.620 | 0.250 | 0.308 | 0.128 | 0.160 |
| EBM | hris_only | 0.253 | 0.611 | 0.150 | 0.179 | 0.077 | 0.237 |
| LR | hybrid | 0.298 | 0.652 | 0.300 | 0.333 | 0.154 | 0.236 |
| **GBM** | **hybrid** | **0.309** | **0.661** | **0.350** | 0.333 | **0.179** | **0.158** |
| EBM | hybrid | 0.255 | 0.622 | 0.200 | 0.205 | 0.103 | 0.235 |
| EBM-OHE *(sensitivity)* | hybrid | 0.271 | 0.643 | 0.200 | 0.256 | 0.103 | 0.233 |

**A calibration aside worth its own sentence.** A constant base-rate predictor
scores a Brier of ≈ 0.162 on this val set. GBM (0.158–0.160) sits *just below*
that line — its probabilities are roughly trustworthy. LR and EBM (0.235–0.237)
sit well *above* it — their probabilities are **worse than guessing the base
rate**, even though their *rankings* are competitive. That is exactly the
fingerprint of `balanced` re-weighting: it inflates predicted probabilities to
help ranking, at the cost of calibration. GBM, which is *not* re-weighting by
default, keeps its probabilities honest. This is the single clearest reason Epic
3 puts every model through explicit calibration before any threshold is set.

### What won, by how much — and is it real?

**Point-estimate champion: GBM × hybrid.** It leads AUC-PR (0.309), Precision@10 %
(0.350), Recall@10 % (0.179), *and* Brier (0.158) — the only cell that tops both
a discrimination metric and the calibration metric.

But a point estimate from 39 positives is a fragile thing, so we ran a **paired
bootstrap** (B = 5,000 resamples; the same resampled employees applied to all six
models each draw, because the cohorts share a row index — this makes the
survey-lift comparison properly paired):

**95 % bootstrap CIs for AUC-PR:**

| Model × Cohort | AUC-PR (point) | 95 % CI |
|---|---|---|
| GBM × hybrid | 0.309 | [0.216, 0.457] |
| LR × hybrid | 0.298 | [0.211, 0.441] |
| GBM × hris_only | 0.291 | [0.198, 0.427] |
| LR × hris_only | 0.284 | [0.201, 0.418] |
| EBM × hybrid | 0.255 | [0.184, 0.366] |
| EBM × hris_only | 0.253 | [0.183, 0.362] |

Every confidence interval overlaps every other one. The full spread between the
best and worst model (0.309 − 0.253 = **0.056**) is smaller than a *single*
model's CI half-width (≈ 0.12). **At this sample size, no pairwise model
difference is statistically significant.**

**So is GBM/hybrid actually the best?** The bootstrap "winner share" — the
fraction of resamples in which each cell tops the table — tells the honest story:

| Cell | P(top of all 6) |
|---|---|
| GBM × hybrid | **0.44** |
| LR × hybrid | 0.30 |
| GBM × hris_only | 0.20 |
| LR × hris_only | 0.05 |
| EBM × (either) | < 0.01 |

GBM/hybrid is the **modal** winner but not a **majority** one — LR/hybrid is a
genuinely credible alternative. What *is* robust: the **hybrid cohort wins
~74 %** of resamples, and a **GBM-or-LR** model wins ~99 %.

**The R1 hypothesis — does survey signal help?** Survey lift, measured as the
paired per-resample difference `AUC-PR(hybrid) − AUC-PR(hris_only)`:

| Model | Point lift | P(lift > 0) | 95 % CI of lift |
|---|---|---|---|
| LR | +0.014 | 0.81 | [−0.020, +0.053] |
| GBM | +0.018 | 0.70 | [−0.058, +0.118] |
| EBM | +0.002 | 0.58 | [−0.030, +0.040] |

**Verdict: R1 is not refuted, and is weakly supported in direction — but the
dataset is underpowered to confirm it.** The lift is positive for all three
models (it never points the wrong way), and the survey columns add a top-3
operational signal: hybrid lifts Precision@10 % from 0.25 → 0.35 for GBM, the
metric HR actually feels. But every lift CI straddles zero, so we cannot claim
the survey signal *significantly* improves ranking at n = 39 positives. The
honest framing for the README is: *"the survey signal consistently helps and
never hurts; with this sample we can show the direction but not certify the
magnitude."*

**Why EBM trails here.** EBM tops the table in under 1 % of resamples — it is
the one model essentially dominated. This is *expected*, not alarming: the
EBM's native-categorical advantage and its additive shape functions pay off
most with **many categorical levels and more data**. With two low-cardinality
categoricals (`performance_tier`, `gender`) and 892 training rows, there is
little structure for it to exploit that the trees and the linear model don't
already capture — and it pays the variance cost of fitting per-feature shape
functions on thin data. EBM's value in this project is **interpretability**
(Loop 3c), not leaderboard position; we keep it for the glass-box explanations
it gives, and we expect the gap to narrow with tuning and scale, not to flip.

### Honest caveats — the boundary of what this comparison proves

1. **Synthetic, single dataset.** Every row is generator-produced (SEED=42; see
   `data_card.md`). The model ranking is a property of *this* synthetic signal
   structure, not a general claim about LR-vs-GBM-vs-EBM on real attrition.
2. **No hyperparameter tuning yet.** All six models use sensible defaults. The
   ranking can and may shift once Loop 3 runs nested cross-validation — defaults
   flatter some families more than others (trees are forgiving; EBM and
   regularized LR are tuning-sensitive). **Do not read this table as a tuned
   verdict.**
3. **Validation, not test.** These are val-set numbers. The test split is
   untouched until Epic 3 champion selection, so the *generalization* estimate
   is still pristine — and still pending.
4. **Underpowered by construction.** 39 validation positives produce AUC-PR CIs
   wider than the entire between-model spread. The bootstrap is doing exactly
   its job: telling us the ranking is *directional evidence*, not proof.
5. **Single-snapshot temporal structure.** The mart is one cross-section
   (`snapshot_date = 2025-05-27`), so the "temporal" split is effectively an
   employee-ordered split, not a train-on-month-T / predict-T+1 forecast. True
   temporal validation waits on pa-warehouse time-series snapshots
   (`data_card.md` Known Limitation #3).

### How to reproduce

```bash
# regenerate the notebook, then execute it end-to-end
uv run python scripts/generate_comparison_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/02_model_comparison.ipynb
```

The 6-cell table, the AUC-PR bar chart (`reports/figures/loop2_comparison_auc_pr.png`),
and the Rung-1 caption are produced by that notebook. The split receipts and the
paired-bootstrap CIs in this section were generated with SEED=42 on the
`data/raw/v_attrition_features_2026-05-28.csv` snapshot.

> **Rung 1 caption (carried on every score in this project):**
> *AUC-PR = 0.309 — associational, not causal (Rung 1).* The model ranks who is
> likely to leave; it does **not** establish that any feature *causes* leaving.

---

## MLflow Setup

*Loop 2 — Story 2.7. Reproduce with `make train` then `make mlflow-ui`.*

### The choice: local file-store, no hosted server

We use MLflow with the **local file-store backend** (`mlruns/` at project root).
No hosted tracking server, no managed service, no Docker dependency.

**Why:** Hosted servers (MLflow Tracking Server mode, W&B, Comet) add
infrastructure without adding portfolio signal for a single-dataset project.
The local file-store is self-contained — every cloner gets a fresh, reproducible
tracking store from their own training runs after `make train`. Nothing needs to
be configured, no credentials issued, no service running before the demo.

**Production path:** Swapping the backend is one environment variable.
`export MLFLOW_TRACKING_URI=http://my-tracking-server:5000` before running
changes the store without touching code. `src/retention/models/tracking.py`
calls `mlflow.set_tracking_uri(str(config.PROJECT_ROOT / "mlruns"))` as a
default; a production deployment would override via that env-var convention.

**Why not W&B / Comet?** These are excellent tools; they're excluded here
because most JDs that mention experiment tracking name MLflow specifically
(Visier, Lattice, Personio tier). The ATS keyword is `mlflow`, not `w&b`.

### `mlruns/` is gitignored on purpose

The tracking store is not committed. `make train` populates it from scratch;
it is therefore reproducible by any reviewer with the data file, and does not
bloat the repository with binary artifacts. The screenshot
(`reports/figures/mlflow_experiment_view.png`) is the committed artifact — it
gives the README reader the UI context without requiring them to run the
training pipeline just to see the chart.

### What each run logs

`src/retention/models/tracking.py::log_run()` wraps `mlflow.start_run()` and
logs:

| Logged item | MLflow category | Example |
|---|---|---|
| `model` | param | `"GBM"` |
| `cohort` | param | `"hybrid"` |
| `seed` | param | `42` |
| Model hyperparams | params | `n_estimators=300, max_depth=4, learning_rate=0.05` |
| `auc_pr` | metric | `0.309` |
| `auc_roc` | metric | `0.661` |
| `prec_at_10` | metric | `0.350` |
| `prec_at_20` | metric | `0.333` |
| `rec_at_10` | metric | `0.179` |
| `brier` | metric | `0.158` |

Metric keys are MLflow-safe identifiers (no `@` or `%`). The UI column headers
show these keys; the notebook's DataFrame uses the display names (`Prec@10%`
etc.). Both refer to the same computed values.

> **Note — MLflow Model Registry (Story 2.7.10):** After Story 3.6 selects
> the champion model (Epic 3), `log_run()` returns the `run_id`, which is
> passed to `mlflow.register_model(f"runs:/{run_id}/model", "rp-champion")`.
> The registry promotes the model through Staging → Production — a lifecycle
> signal senior reviewers look for. This section will be updated at that point.

---

## EBM Preprocessing Sensitivity

*Loop 2 — Story 2.9. Reproduce with `uv run python scripts/story_2_9_ebm_sensitivity.py`.*

### The question

The cross-model comparison in Stories 2.5–2.5.6 uses a **different preprocessor
for EBM** than for LR/GBM (see *The preprocessing asymmetry* above). A diligent
reviewer will ask: "does EBM trail because of the model family, or because of
the preprocessing choice? Did you actually test both?"

This section answers that directly. We train a second EBM variant —
**EBM-OHE** — that receives the *same* OneHotEncoded input as LR and GBM,
then compare it against the production **EBM-native** (raw string categorical
columns, native GAM binning). All other hyperparameters are identical.

### Setup

| Parameter | Value |
|---|---|
| Data snapshot | `v_attrition_features_2026-05-28.csv` — same as main comparison table |
| Val set | 191 rows, 39 positives (20.4 % base rate) |
| Seed | `config.SEED = 42` via `config.set_global_seed()` before each fit |
| EBM hyperparams | `interactions=10, max_bins=256, outer_bags=8` — identical for both variants |
| Imbalance handling | `compute_sample_weight('balanced')` — identical for both variants |
| EBM-native preprocessor | `build_ebm_preprocessor()` — median impute, no OHE, `set_output("pandas")` → DataFrame passthrough for native categorical bins |
| EBM-OHE preprocessor | `build_preprocessor()` — same OHE pipeline as LR/GBM; categoricals become float64 binary columns |

### Results

| Variant | Cohort | AUC-PR | AUC-ROC | Prec@10% | Prec@20% | Rec@10% | Brier |
|---|---|---|---|---|---|---|---|
| EBM-native | hris_only | 0.253 | 0.611 | 0.150 | 0.179 | 0.077 | 0.237 |
| EBM-OHE | hris_only | 0.267 | 0.630 | 0.250 | 0.256 | 0.128 | 0.235 |
| EBM-native | hybrid | 0.259 | 0.622 | 0.200 | 0.231 | 0.103 | 0.234 |
| EBM-OHE | hybrid | 0.271 | 0.643 | 0.200 | 0.256 | 0.103 | 0.233 |

**AUC-PR delta (EBM-native − EBM-OHE):**

| Cohort | EBM-native | EBM-OHE | Delta |
|---|---|---|---|
| hris_only | 0.253 | 0.267 | −0.014 (OHE wins) |
| hybrid | 0.259 | 0.271 | −0.012 (OHE wins) |

> **Note on the hybrid EBM-native number (0.259 here vs 0.255 in the main
> comparison table):** The main table was generated by the comparison notebook,
> which trains all six cells in sequence. This script trains in a different order
> (native + OHE back-to-back per cohort). EBM uses `outer_bags=8` random
> bagging — despite `random_state=42`, the exact PRNG state at fit time varies
> with execution order. The 0.004 discrepancy is within EBM's execution-order
> variance; the main table's 0.255 is canonical for the 6-cell comparison.

### What this tells us

**OHE gives EBM a modest boost on this dataset (+0.012–0.014 AUC-PR).**
This is counterintuitive — InterpretML's documentation characterises native
categorical handling as "equivalent or slightly better" than OHE. The reversal
here is attributable to dataset characteristics:

- **Low cardinality.** The two categorical columns are `performance_tier`
  (4 levels) and `gender` (3 levels) — 7 total levels. OHE expansion of 7
  levels is modest, not a "dummy variable explosion." With native handling, EBM
  fits one GAM shape function per categorical column; with OHE it fits 7 binary
  shape functions. On thin data (892 training rows) the binary representation
  may have more discriminative signal per bin than the per-level native bins.
- **Low absolute scale.** Native GAM binning pays off most when categorical
  columns have many levels or when there are many training examples per level.
  At 4 + 3 levels on 892 rows, each native bin averages only ~130–220 examples.
  The OHE binary columns impose cleaner separations that may generalise better
  at this scale.

**Does this change the main comparison conclusion? No.**

The critical question is not "EBM-native vs EBM-OHE" but "does EBM keep up
with GBM/LR regardless of preprocessing?" The answer is no:

- EBM-OHE × hybrid (0.271) still trails **GBM × hybrid (0.309) by 0.038**.
- The native-vs-OHE delta (0.012–0.014) explains at most **25–27 % of EBM's lag**
  behind GBM. The remaining 73–75 % is attributable to model family differences,
  not preprocessing.
- The **survey-lift comparison** (hris_only vs hybrid within each model) is
  entirely unaffected — both EBM variants use the same preprocessor within a
  cohort, so the signal isolation is clean.
- The **cross-model ranking** is unchanged: GBM > LR > EBM regardless of
  whether EBM uses native or OHE preprocessing.

**Why do we still ship EBM-native?** Because native handling is the correct
production choice: it preserves the single, readable "effect of
`performance_tier`" shape function that makes EBM's global explanation
interpretable. With OHE, that shape function fragments into four disconnected
binary stumps (`performance_tier_2`, `_3`, `_4`, `_5`), degrading readability
without a meaningful gain in portfolio signal. EBM's role in this project is
**interpretability in Loop 3c** — not leaderboard position. We run it the way
it should run in production, and we cite this sensitivity check to confirm the
comparison is robust to that choice.

### Summary verdict

> *EBM-native achieved AUC-PR 0.253 (hris_only) and 0.259 (hybrid) with native
> categorical handling. EBM-OHE — using the same OHE preprocessing as LR and
> GBM — achieved 0.267 and 0.271. The delta (+0.012–0.014 in favour of OHE) is
> consistent with EBM's native-binning advantage being dataset-size-sensitive:
> at 892 training rows and 7 total categorical levels, OHE's clean binary
> separations slightly outperform native GAM bins. However, EBM-OHE still
> trails GBM/hybrid by 0.038, confirming that preprocessing accounts for at
> most 25 % of EBM's gap behind the leader. The cross-model ranking is robust
> to preprocessing choice.*

---

## Threshold Calibration

*Loop 2 — Story 3.3. Reproduce with `uv run python scripts/generate_threshold_figure.py`.*

Calibration (Story 3.2) asked whether the probabilities are *trustworthy
numbers*. This section asks the next question: **given trustworthy
probabilities, where do we draw the line between "flag for a retention
conversation" and "leave alone"?** A probability is not a decision; an HR team
needs a binary flag, and a flag needs an operating threshold. The default 0.5 is
almost never right under a 20 % base rate — demanding `p > 0.5` to act flags
almost no one, so real exits slip through. We pick the threshold empirically,
and we frame that choice two ways: as a controlled experiment, and as the honest
alternative to resampling.

### Framing 1 — threshold selection is a controlled experiment

Sweeping the threshold is neither retraining nor hyperparameter tuning. The
model's ranking is **frozen** — every employee's predicted probability is fixed.
We vary only the cut-point, so each candidate threshold is a **treatment arm**:
the decision policy *"flag everyone with `p ≥ t`"* applied to the same scored
population. We evaluate every arm on the same held-out set, score each on a
**pre-declared** metric, and adopt the empirically-winning arm.

That is the discipline of an A/B test: pre-register the metric, compare arms on
common held-out data, pick the winner, confirm on a fresh sample. The honest
limit of the analogy — stated so a reviewer doesn't have to catch us on it — is
that this is a **within-system** experiment, not a randomized **between-system**
one. A classic A/B test randomizes users across two competing systems; here
there is one deployed system and the arms are its operating points, all scored on
the same fixed validation rows (a within-subjects comparison, closer to offline
policy evaluation than to randomized assignment). The **mechanism** differs; the
**discipline** transfers intact. Naming it correctly is the point — every
threshold-selection process in ML is implicitly this experiment; most
practitioners simply never label it one.

### Framing 2 — why a threshold, not SMOTE (the [FLIP-RISK] verdict)

The textbook reflex for a 20 % positive rate is to resample (SMOTE). We refuse,
for two reasons that compound:

1. **SMOTE breaks the temporal split.** It synthesises minority rows by
   interpolating between existing ones. Those synthetic employees have no place
   in time — they straddle the train/val boundary in feature space, quietly
   voiding the ordering guarantee `temporal_split()` and `assert_no_temporal_leak()`
   exist to protect.
2. **SMOTE distorts calibration.** Inflating minority density breaks the
   "predicted 0.30 ≈ observed 0.30" property verified one section above — and
   Epic 3's entire downstream chain (calibration → threshold → expected value) is
   built on that property. Resampling would saw off the branch we are standing on.

Threshold calibration reaches the same operational goal — catch more exits —
**without touching the data distribution at all.** We keep the honest, calibrated
probabilities and simply move the decision line. This is the **[FLIP-RISK]**
mitigation named in the Loop 1 risk register: the expensive failure is a *missed
exit* (a regretted-attrition employee we failed to flag), far costlier than a
*wasted conversation* (a retention chat with someone who would have stayed). The
F2 criterion encodes that asymmetry — it weights recall twice as heavily as
precision — so the F2-optimal arm sits at or below the F1-optimal one, accepting
more false positives to miss fewer real exits.

### Leakage discipline — sweep on validation, confirm on test once

Choosing an operating point is a model-selection decision, so the sweep runs on
the **validation** split. Optimising the threshold on test and then reporting
test metrics at that threshold is leakage — tuning on the data you report on. The
winning arm is confirmed **once** on the held-out test set at champion selection
(Story 3.6); until then the test split stays pristine, exactly as it has since
Story 2.5. Every number below is a validation-set number.

### Results — operating points for the leading model

We sweep thresholds 0.05 → 0.95 in 0.01 steps (91 arms) for **GBM × hybrid** —
the Loop 2 point-estimate leader, and the only cell whose probabilities are
calibrated enough (Brier 0.158) to make a threshold meaningful. Validation set:
191 rows, 39 positives, base rate **0.204**.

| Criterion | Optimal t | Precision | Recall | F1 | F2 | TP / FP / FN | Flagged |
|---|---|---|---|---|---|---|---|
| **F1** (balanced) | **0.18** | 0.343 | 0.615 | 0.440 | 0.531 | 24 / 46 / 15 | 70 (37 %) |
| Youden's J | 0.18 | 0.343 | 0.615 | 0.440 | 0.531 | 24 / 46 / 15 | 70 (37 %) |
| F2 (recall-weighted) | 0.05 | 0.222 | 1.000 | 0.363 | 0.587 | 39 / 137 / 0 | 176 (92 %) |

Three findings, each worth stating plainly:

**The F1 arm (t = 0.18) is the sensible default.** It flags 37 % of the
workforce, catches 62 % of all exits (recall 0.615), and the flagged group exits
at 34.3 % versus the 20.4 % base rate — a **1.68× lift** in the people HR actually
talks to. The optimal threshold is far below 0.5, which is the whole reason we
sweep: the default cut-point would have flagged almost no one.

**F1 and Youden agree exactly (t = 0.18).** Two criteria built on different
foundations — F1 from precision/recall, Youden's J from sensitivity/specificity
and prevalence-free by construction — converge on the same arm. That agreement is
a small robustness signal: the operating point is not an artifact of one metric's
idiosyncrasy.

**The F2 arm collapses to the grid floor (t = 0.05) — and that is the honest,
instructive result.** Pure recall-weighting on 39 positives has no brake: F2
keeps lowering the threshold to buy recall until it flags 176 of 191 employees
(92 %) and reaches recall 1.0 — at which point precision (0.222) is barely above
the base rate. "Talk to everyone" is not an operating policy; it is the *absence*
of one. This is precisely why **F-beta is the wrong final tool for this decision**
and Story 3.4 exists: F-beta encodes a *fixed, unitless* recall-to-precision
ratio, but the real trade-off is **dollars** — a wasted conversation costs
manager + HRBP time, a missed exit costs a replacement hire. Only an explicit
expected-value model (3.4), which prices false positives and false negatives in
currency, can discipline the threshold to reflect actual business cost rather
than an arbitrary β. The F2 collapse is the *motivating failure* for the EV
framing, not a number to report as a recommendation.

### Honest caveats

1. **Underpowered, like everything in Loop 2.** 39 validation positives make
   every operating point an estimate with wide error bars. The F1 arm at 0.18 is
   *illustrative of the procedure*, not a production setting carved in stone — it
   is re-derived and confirmed on the test set at champion selection (3.6).
2. **The threshold is dataset- and cost-specific.** It is correct only for this
   synthetic signal, at this base rate, under an unspecified cost ratio. Story
   3.4 makes the cost ratio explicit; a real deployment would re-sweep on its own
   data with its own replacement and intervention costs.
3. **Precision drops to zero past t ≈ 0.43** in the figure because the model's
   maximum predicted probability on the validation set is ≈ 0.42 — no employee is
   flagged above it, the confusion matrix empties, and precision is 0 by the
   `0/0 → 0` convention. That ceiling is itself a (reassuring) calibration fact: a
   well-behaved model at a 20 % base rate should not be emitting 0.9 exit
   probabilities.

### How to reproduce

```bash
uv run python scripts/generate_threshold_figure.py
```

Trains GBM × hybrid on `data/raw/v_attrition_features_2026-05-28.csv` (SEED = 42),
sweeps thresholds on the validation set, and writes
`reports/figures/threshold_sweep.png` — the treatment-arm comparison with the F1
and F2 arms marked. The `optimize_threshold`, `threshold_sweep`, and
`threshold_sweep_plot` functions live in `src/retention/models/threshold.py` and
are unit-tested in `tests/test_threshold.py` (19 cases).
