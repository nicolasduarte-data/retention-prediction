# Methodology — retention-prediction

**Status:** living document — one section accretes per loop
**Last updated:** 2026-05-29 (Loop 2 — rp-prey-002, Story 3.6)
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
| [Expected Value and p_eff Sensitivity](#expected-value-and-p_eff-sensitivity) | 2 | 3.4 |
| [Flat-CV vs Nested-CV](#flat-cv-vs-nested-cv) | 2 | 3.5 |
| [Champion Selection](#champion-selection) | 2 | 3.6 / 2.7.10 |

Sections scaffolded for later loops (added when the work ships, not before):
*Test Quality / Mutation Testing* (3.7) ·
*Fairness Thresholds + Chouldechova* (Epic 5) · *Adversarial SHAP* (Epic 6).

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

> **MLflow Model Registry (Story 2.7.10) — shipped.** Where `log_run()` records
> *every* training run for the experiment view, `register_champion()`
> (`tracking.py`) records the *one* winner: it logs the champion's fitted
> pipeline, registers it as **`rp-champion`**, and promotes the new version to
> the **Production** stage — the named, versioned, stage-tagged lifecycle signal
> senior reviewers look for. The current champion (GBM × hybrid) is registered
> as `rp-champion` v1 → Production; Loop 4 loads it with
> `mlflow.sklearn.load_model("models:/rp-champion/Production")`. The selection
> logic and the val/test confirmation behind that registration are in
> [Champion Selection](#champion-selection) below.

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

---

## Expected Value and p_eff Sensitivity

*Loop 2 — Story 3.4. Reproduce with `uv run python scripts/generate_ev_figure.py`.*

Threshold Calibration (Story 3.3) ended on a deliberate cliffhanger. Asked to
honour the [FLIP-RISK] recall preference, the F2 criterion slid to the grid floor
and flagged 92 % of the workforce — because **an F-score has no brake.** It
encodes a fixed, unitless recall-to-precision ratio and never asks what a wasted
conversation or a missed exit actually *costs*. This section supplies the brake by
pricing the decision in dollars. It is the resolution the 3.3 write-up promised,
not a new experiment.

### The model — value of running the program vs. doing nothing

Each employee is either flagged (`p ≥ t` → a retention conversation) or left
alone. Three numbers price the consequences:

| Symbol | Meaning | Value used |
|---|---|---|
| `rc` | replacement cost — fully-loaded cost to backfill a departure | $90,000 |
| `ic` | intervention cost — manager + HRBP time for one conversation | $2,000 |
| `p_eff` | effectiveness — fraction of flagged genuine exits actually retained | swept 0.1–0.9 |

We score each confusion-matrix cell **relative to the do-nothing baseline** (flag
no one, absorb every exit's replacement cost):

| Cell | Outcome under the model | Value vs. baseline |
|---|---|---|
| TP | flag a true exit; retain with probability `p_eff` | `+p_eff·rc − ic` |
| FP | flag someone who'd have stayed; waste the conversation | `−ic` |
| FN | miss a true exit (identical to baseline) | `0` |
| TN | correctly leave a stayer alone | `0` |

Summing the only two non-zero cells:

> **EV = p_eff · rc · TP − ic · (TP + FP)**

The decisive line is that **the FN term cancels.** A missed exit costs the same
replacement dollars whether or not the model exists, so it cannot be part of the
model's *added* value. EV depends only on the **flagged** population (TP and FP) —
the formula is telling you something true: this is the expected value of the *act
of flagging*, and its quality is governed entirely by who lands on the list.

### The double-count we did not make

The textbook EV formula for this problem usually reads `EV = TP·(p_eff·rc) −
FP·ic − FN·rc`, and it is wrong here. Subtracting `FN·rc` while also crediting
`p_eff·rc` to TP **double-counts a flagged true exit**: it banks the `p_eff·rc`
retention benefit *and* the full avoided `rc` (by lifting that employee out of the
FN penalty). But a flagged exit is only saved `p_eff` of the time; the
`(1 − p_eff)` who leave anyway still cost `rc`, which the naive formula silently
books as $0. The overstatement is ≈ `rc + ic` per catch — and because it scales
with TP, it stampedes the EV-optimal threshold toward "flag everyone" for a reason
that is an **accounting error, not an economic truth.** The do-nothing baseline
removes the temptation by construction: `rc` only ever appears multiplied by
`p_eff`, never at full value. (The absolute unprevented loss, `FN·rc`, is real and
worth *reporting* beside the EV as context — never inside the objective being
optimised.)

### Parameters and citations

| Parameter | Value | Source |
|---|---|---|
| Replacement-cost multiplier | 1.5 × annual salary | SHRM (2024) fully-loaded replacement-cost rule of thumb — recruiting + onboarding + productivity ramp ≈ 1.5× salary |
| Representative salary | $60,000 → `rc = $90,000` | stand-in; see note below |
| Intervention cost | $2,000 | manager + HRBP preparation and meeting time for one retention conversation |
| Effectiveness `p_eff` | 0.30 central, **swept 0.1–0.9** | a forward assumption — no dataset can measure it without running the program; the sweep is what makes the framing defensible |

**Why a representative salary, not the cohort's actual mean.** The mart exposes
`compa_ratio` (salary ÷ band midpoint), **not absolute salary** — the dataset is
anonymised by design. So `rc` uses a representative $60k → $90k rather than a
figure derived from the data. A real deployment substitutes the cohort's true mean
salary via `replacement_cost_from_salary()`; every breakeven below is scale-free
in `rc`, so only the y-axis magnitude moves, never the decision.

### The breakeven closed form — precision buys robustness

Set EV = 0 and solve for the effectiveness at which the program starts paying:

> **p_eff\* = ic·(TP + FP) / (rc·TP) = (ic / rc) / precision**

The breakeven is the cost ratio divided by the precision at that threshold. Two
readings of the same identity:

- **As a number:** at the F1 operating point (precision 0.343),
  `p_eff* = (2000/90000)/0.343 ≈ 0.065`. The program pays for itself if retention
  conversations work even ~6–7 % of the time — a low, very crossable bar.
- **As a principle:** **breakeven falls as precision rises.** A cleaner flag list
  wastes less budget on false positives, so it tolerates *less* effective
  interventions before it loses money. This is the economic argument for
  precision@k (Story 3.1) and the natural counterweight to 3.3's F2 recall
  pressure: recall fills the flag list; precision is what makes funding it
  defensible.

### Results — and the honest twist

Applying the framing to the two operating points from Story 3.3 (GBM × hybrid,
validation set, 39 positives):

| Arm (from 3.3) | t | Precision | TP / FP | Breakeven p_eff | EV @ p_eff = 0.30 |
|---|---|---|---|---|---|
| **F1** (balanced) | 0.18 | 0.343 | 24 / 46 | **0.065** | **+$508,000** |
| F2 (recall-weighted) | 0.05 | 0.222 | 39 / 137 | 0.100 | +$701,000 |

The twist a careful reader must see: **at the central p_eff = 0.30, the
recall-heavy F2 arm posts the higher total EV** ($701k vs $508k). At these costs
an intervention is cheap — $2,000 against a $90,000 replacement — so the marginal
flag pays off as long as the people it adds exit above the **marginal precision
bar** `ic/(p_eff·rc) = 2000/27000 ≈ 7.4 %`. The base rate is 20.4 %, so at
p_eff = 0.30 even indiscriminate widening adds value, and "flag almost everyone"
maximises EV. **EV did not abolish the F2 collapse; it priced it.**

So what does EV buy over the bare F-score? It converts the single unknown —
effectiveness — into an explicit decision map:

| If you believe… | Then… | Because |
|---|---|---|
| `p_eff > 0.10` | flag wide (F2 arm) | both arms profit; the wider net banks more total EV |
| `0.065 < p_eff < 0.10` | flag selectively (F1 arm) | F2 is underwater here; only the precise arm pays |
| `p_eff < 0.065` | don't run the program | no operating point breaks even |

The F1 arm tolerates interventions **35 % less effective** than the F2 arm before
it loses money (0.065 vs 0.100). That is the brake the F-score lacked: not a
smaller flag list handed down by fiat, but an in-dollars statement of exactly
*what each operating point is betting on*. Because we have never run the program
and cannot read `p_eff` off the data, the lower-breakeven F1 arm is the more
defensible default — and champion selection (Story 3.6) confirms it on the test
set with this map in hand.

### Leakage discipline — same rule as the sweep

EV is computed on the **validation** split, from the same frozen probabilities and
the same operating points the threshold sweep used. Choosing an operating point is
model selection; the test set is confirmed once, at champion selection (3.6).
Every dollar figure above is a validation-set number.

### Honest caveats

1. **`p_eff` is an assumption, not a measurement.** The whole program economics
   hinge on a number no dataset can supply. We never report a single EV; the sweep
   and the decision map are the honest deliverable. Only a pilot that measures
   actual retention lift collapses the range.
2. **Representative salary, not actual.** `rc = $90k` rests on a $60k stand-in
   because the data is ratio-only (above). Breakevens are scale-free in `rc`, but
   the absolute dollar magnitudes are illustrative, not this workforce's true P&L.
3. **Underpowered, like all of Loop 2.** TP and FP come from 39 validation
   positives, and the EV line inherits that fragility. The decision *map* is
   robust (it is algebra); the specific dollar values are wide-error estimates.
4. **EV is exactly linear in `p_eff`** at a fixed threshold — TP and FP are
   constant, so the chart is a straight line by construction, not an empirical fit.
   The structure worth reading is *where it crosses zero*, not its shape.
5. **Rung 1 throughout.** EV prices the *ranking* the model produces; it makes no
   causal claim that flagging an employee, or intervening, *causes* retention. The
   `p_eff` parameter is exactly where a real causal effect would have to be
   measured rather than assumed.

### How to reproduce

```bash
uv run python scripts/generate_ev_figure.py
```

Trains GBM × hybrid on `data/raw/v_attrition_features_2026-05-28.csv` (SEED = 42),
fixes the F1 operating point from Story 3.3, sweeps `p_eff` on the validation set,
and writes `reports/figures/ev_sensitivity.png`. The `ev_at_threshold`,
`p_eff_sensitivity_sweep`, `breakeven_p_eff`, and `expected_value_plot` functions
live in `src/retention/evaluation/expected_value.py` and are unit-tested in
`tests/test_expected_value.py` (32 cases).

> **Rung 1 caption:** *EV in dollars is built on an associational ranking (Rung 1)
> plus an assumed intervention effectiveness. It bounds the program's value under
> stated assumptions; it does not prove that intervention causes retention.*

---

## Flat-CV vs Nested-CV

*Loop 2 — Story 3.5. Reproduce with `uv run python -c "..."` (see below).*

### The optimism-bias problem with a single validation split

Everything above this section evaluates models on a **single, fixed validation
split**: train once on 892 rows → score once on 191 rows → report one number.
That is fast and readable, but it conceals a subtle bias when **hyperparameter
tuning is involved**.

In this project the GBM has several tunable knobs — tree depth, learning rate,
number of estimators. If we had tuned those knobs by picking whichever
combination scored best on the validation set and then *reported* the validation
score as our performance estimate, we would be optimistic: the hyperparameters
were chosen *because* they scored well on those 191 rows, so the reported score
over-estimates generalisation.

**Flat (single-loop) cross-validation** has the same problem. A k-fold CV that
uses the same folds for both tuning and reporting inflates the reported metric by
an amount proportional to the size of the parameter grid and the noise level of
the data.

Cawley & Talbot (2010, JMLR 11:2079–2107) showed that the model-selection bias
in flat CV can be as large as the variance it was meant to measure — making
reported AUC-PR numbers look more stable *and* more favourable than they are.

### The nested-CV remedy

**Nested (double-loop) cross-validation** separates the two concerns:

| Loop | Role | Data seen | What it produces |
|---|---|---|---|
| **Outer** (5 folds) | Evaluation | Never sees inner decisions | 5 unbiased AUC-PR scores → mean ± std |
| **Inner** (5 folds per outer fold) | Hyperparameter selection | Only outer training rows | Best param combo per outer fold |

The outer test fold is completely invisible during inner-loop selection and during
the refit on the outer training set. Its score is therefore free of
model-selection bias — the hyperparameters were chosen *without* looking at it.
Averaging 5 such scores yields an honest mean ± std that reflects both the
expected performance *and* the fold-to-fold variance of the procedure.

### Why GBM and not LR or EBM?

**LR** has one effective regularisation knob (`class_weight='balanced'`, fixed by
the no-SMOTE constraint) and a convex loss surface — its flat-CV variance is small
enough to be inconsequential.  
**EBM** is computationally expensive to nest at this sample size and its
intrinsic regularisation (learning rate × max bins) is well-behaved; the
EBM-native-vs-OHE sensitivity check (Story 2.9) already quantifies its variance.  
**GBM** has a larger hyperparameter surface — depth × learning rate × estimator
count × subsampling — and is known to overfit on small tabular HR datasets. It is
the cell where the honest claim "nested CV confirms the point estimate" most needs
to be earned.

### Setup

| Parameter | Value |
|---|---|
| Data | train + val rows = 1,083 (test withheld for champion selection at Story 3.6) |
| Base rate | 0.184 (184 voluntary exits in 1,083 rows) |
| Seed | `config.SEED = 42` for both StratifiedKFold instances |
| Outer splits | 5 |
| Inner splits | 5 |
| Parameter grid | 4 combinations — `max_depth ∈ {3, 4}`, `learning_rate ∈ {0.05, 0.10}`, `n_estimators ∈ {100, 200}` |
| `scale_pos_weight` | Computed from each outer training fold (neg/pos ratio) — never from the test fold |
| Total model fits | 5 outer folds × (5 inner folds × 4 combos + 1 refit) = 105 per cohort |

### Results

| Cohort | Nested CV AUC-PR | ± std | Per-fold scores |
|---|---|---|---|
| hris_only | **0.263** | ±0.022 | 0.274, 0.265, 0.232, 0.246, 0.296 |
| hybrid | **0.269** | ±0.024 | 0.259, 0.275, 0.235, 0.307, 0.268 |

For reference — the flat single-split validation scores (from the 6-cell
comparison table, evaluated on 191 val rows):

| Cohort | Flat CV AUC-PR (val) | Nested CV AUC-PR | Gap (optimism) |
|---|---|---|---|
| hris_only | 0.291 | 0.263 | **−0.028** |
| hybrid | 0.309 | 0.269 | **−0.040** |

**Three findings worth stating plainly:**

**1 — The optimism bias is real and quantifiable.** The flat validation estimate
for GBM × hybrid (0.309) is 0.040 AUC-PR units above the nested CV estimate
(0.269) — about **13 % inflation**. For hris_only the gap is 0.028 (about 10 %).
This is not a data quality failure; it is expected when the same validation set
was used for threshold sweeping and operating-point selection in Stories 3.3–3.4.
Nested CV removes that bias.

**2 — The nested CV headline confirms GBM's standing.** The corrected estimate
(0.269 ± 0.024) is still meaningfully above the no-skill baseline (≈ 0.184 =
base rate), and GBM × hybrid remains the point-estimate leader even after
deflation. The correction shrinks the headline number; it does not change the
ranking.

**3 — The survey lift narrows.** In the flat comparison, the hybrid lift over
hris_only was 0.018 AUC-PR for GBM. Under nested CV the lift shrinks to 0.006
(0.269 − 0.263). The direction is preserved — hybrid still leads — but the
magnitude is within the ±0.022–0.024 fold-to-fold noise. This is consistent with
the bootstrap finding from Story 2.5: the survey lift is positive in direction but
underpowered to confirm. Nested CV neither refutes nor strengthens the R1
hypothesis; it simply provides the honest per-fold variance that the single-split
estimate cannot.

### Honest caveats

1. **Comparing nested CV to flat CV is not apples-to-apples.** The flat estimate
   uses 892 training rows → 191 test rows (one split). The nested CV uses 1,083
   rows (train + val) across 5 folds → each outer test fold ≈ 217 rows. Different
   training set sizes and different test populations mean the gap partly reflects
   training-set size, not only optimism bias. The comparison is instructive, not
   exact.
2. **105 fits on 1,083 rows.** At this sample size, each inner fold trains on
   ~693 rows. The grid is deliberately small (4 combos) to avoid selecting on
   noise; a larger grid would not be meaningful here.
3. **StratifiedKFold with shuffle=True + seed=42 — deterministic.** Running the
   script twice with the same seed produces identical scores. The fold variance
   (±0.022–0.024) is therefore *structural* (genuine between-fold variation in the
   data) not *stochastic* (random sampling variation). This is the right
   interpretation: fold-to-fold variance measures how sensitive the model is to
   which quarter's exits land in the test fold.

### How to reproduce

```python
from retention import config
from retention.data.load import load_attrition_features_local
from retention.data.split import temporal_split
from retention.features.cohorts import extract_X_y, split_cohorts
from retention.evaluation.nested_cv import nested_cv_auc_pr
import pandas as pd

df = load_attrition_features_local("data/raw/v_attrition_features_2026-05-28.csv")
config.set_global_seed()
train_df, val_df, _ = temporal_split(df, save_indices=False)
trainval_df = pd.concat([train_df, val_df], ignore_index=True)

for cohort in ("hris_only", "hybrid"):
    X, y = extract_X_y(split_cohorts(trainval_df)[cohort], cohort)
    result = nested_cv_auc_pr(X, y, cohort, outer_splits=5, inner_splits=5)
    print(result.summary())
```

The `nested_cv_auc_pr` function and `NestedCVResult` live in
`src/retention/evaluation/nested_cv.py` and are unit-tested in
`tests/test_nested_cv.py` (17 cases). The full narrative including champion
selection appears in `notebooks/03_evaluation_rigor.ipynb` (Story 3.6).

---

## Champion Selection

*Loop 2 — Stories 3.6 (champion) and 2.7.10 (registry). Reproduce with `make evaluate`.*
*Full narrative: `notebooks/03_evaluation_rigor.ipynb`.*

Epic 3 built five lenses — discrimination (3.1), calibration (3.2), threshold
(3.3), expected value (3.4), unbiased generalisation (3.5). This section is where
they converge into the single decision the whole loop exists to make: **of the
six model × cohort cells, which one ships?**

### The rule — a gated rank, not a sort

The naive answer is "highest AUC-PR." We reject it, because a cell can rank well
while being mis-calibrated — and every downstream HR action (the threshold of
3.3, the EV case of 3.4) is computed from *probabilities*, not ranks. A model
whose probabilities lie is unsafe to operate even if it sorts employees
correctly. So selection is a **two-stage gate**
(`src/retention/evaluation/champion.py::select_champion`):

1. **Calibration gate.** A cell is *eligible* only if its Brier ≤ the base-rate
   Brier `β·(1 − β)`, where β is the validation prevalence. That bar is the Brier
   of the best *constant* predictor (output β for everyone) — derivation in
   `_base_rate_brier`. A cell below it provably beats "predict the base rate"; a
   cell above it is worse than the trivial baseline at the one thing Brier
   measures, and we refuse to operate its probabilities no matter how it ranks.
2. **Discrimination rank.** Among the eligible cells, the champion is the highest
   AUC-PR — the primary metric across every loop.
3. **Honest fallback.** If *no* cell clears the gate, select the highest-AUC-PR
   cell overall, set `passed_calibration_gate = False`, and say so. That path was
   **not** taken here.

This is not a new claim — it is the codification of the calibration fingerprint
the comparison table (Story 2.5) already surfaced: GBM keeps its probabilities
honest (`eval_metric='aucpr'`, no re-weighting), while `class_weight='balanced'`
/ `compute_sample_weight` inflate LR's and EBM's.

### The gate in numbers

Validation prevalence β = 0.204, so the gate is `0.204 · 0.796 = 0.162`.

| Cell | AUC-PR | Brier | Brier ≤ 0.162? | ECE |
|---|---|---|---|---|
| LR × hris_only | 0.284 | 0.237 | ✗ fail | 0.278 |
| GBM × hris_only | 0.291 | 0.160 | ✓ **pass** | 0.061 |
| EBM × hris_only | 0.253 | 0.237 | ✗ fail | 0.280 |
| LR × hybrid | 0.298 | 0.236 | ✗ fail | 0.279 |
| **GBM × hybrid** | **0.309** | **0.158** | ✓ **pass** | **0.048** |
| EBM × hybrid | 0.259 | 0.234 | ✗ fail | 0.278 |

**Exactly the two GBM cells clear the gate.** Among them, GBM × hybrid wins on
AUC-PR (0.309 vs 0.291). The four `balanced`-reweighted cells (LR, EBM) fail by a
wide margin — Brier ≈ 0.234–0.237, ~45 % above the bar — exactly as the
comparison-table calibration aside predicted. The gate did real work: it
eliminated four cells *before* discrimination was consulted, and two of those
(LR × hybrid 0.298, LR × hris_only 0.284) **out-rank** the eligible GBM ×
hris_only on AUC-PR. A naive sort would have shortlisted them; the gate correctly
refused, because their probabilities are worse than guessing the base rate.

### The champion: GBM × hybrid

| Metric | Validation (selected on) | Test (confirmed once) | Optimism gap |
|---|---|---|---|
| AUC-PR | 0.309 | 0.274 | +0.035 |
| Precision@10 % | 0.350 | 0.250 | +0.100 |
| ECE | 0.048 | 0.096 | −0.048 |
| Brier | 0.158 | 0.173 | −0.015 |

Operating threshold: **0.05** — the F2-optimal point on validation, i.e. the
[FLIP-RISK] recall-weighted operating point Story 3.3 argues for (recall matters
more than precision when a missed exit costs more than a wasted conversation). At
p_eff = 0.30 that operating point is worth **+$701,000**, with breakeven at
p_eff = **0.100** (Story 3.4's framing applied to the champion's own threshold).

### Test-set discipline — confirmed once, and the honest read

The champion was chosen **entirely on validation**. The held-out test set is
touched exactly once, here, to confirm it generalises — never to choose it. Both
metric sets are stored on the persisted artifact (`val_metrics`, `test_metrics`)
so the optimism gap is visible in the record, not hidden.

The read is honest both ways:

- **Discrimination holds.** Test AUC-PR 0.274 sits +0.035 below validation — a
  modest, expected optimism gap, and still well above the test no-skill baseline
  (β = 0.215). It also lands inside one standard deviation of the unbiased
  nested-CV mean from Story 3.5 (0.269 ± 0.024) — the cross-check that the
  validation pick was not a fold-luck artefact.
- **Calibration is marginal on test.** Test Brier 0.173 is *just above* the
  test base-rate Brier (`0.215 · 0.785 = 0.169`): the champion clears the gate on
  the selection set (as the rule requires) but narrowly misses it on test. This
  is the calibration analogue of the AUC-PR optimism gap, and exactly what
  39-/41-positive splits produce. The honest framing — the *ranking* generalises
  cleanly; the *probabilities* are good but not bullet-proof off-sample. That is
  why every EV figure ships with the p_eff sweep rather than a point claim, and
  why a production deployment would add an explicit calibration step (isotonic /
  Platt) before freezing any threshold.

### Persistence and registry — two paths to the same model

- **File store.** `persist_champion()` pickles a `ChampionArtifact` — the fitted
  pipeline + operating threshold + full provenance (the selection decision,
  val/test metrics, feature names, seed, UTC timestamp) — to
  `reports/models/champion.pkl`. Loop 4's write-back loads it with
  `load_champion()`. The file is self-describing: a reviewer can unpickle it and
  read exactly how it was chosen.
- **MLflow Model Registry (Story 2.7.10).** `register_champion()` logs the
  sklearn pipeline, registers it as **`rp-champion`**, and promotes the new
  version to **Production**. The current champion is `rp-champion` **v1 →
  Production**; load it with
  `mlflow.sklearn.load_model("models:/rp-champion/Production")`. Where the
  experiment view (Story 2.7) records *every* run, the registry records the *one*
  winner. Inspect it with `make mlflow-ui` → *Models* tab → `rp-champion`.

### Honest caveats

1. **Underpowered, like all of Loop 2.** The champion is selected on 39
   validation positives and confirmed on 41 test positives. The decision
   *procedure* is sound; the specific numbers carry the wide error bars the
   bootstrap (2.5) and nested CV (3.5) already quantified.
2. **The gate is a validation-set gate by design.** Eligibility is decided at
   β = 0.204 on validation, because selection must happen before the test set is
   opened. Test calibration is *reported*, not gated — and it came in marginal,
   as disclosed above.
3. **Single synthetic dataset, no tuning.** The champion uses GBM defaults
   (`scale_pos_weight` off). The ranking is a property of this synthetic signal;
   a real deployment re-runs the whole gate on its own data, and would likely add
   hyperparameter tuning and explicit probability calibration before freezing a
   threshold.

### How to reproduce

```bash
make evaluate
```

Regenerates and re-executes `notebooks/03_evaluation_rigor.ipynb` end to end (the
nested-CV cell needs a raised per-cell timeout, set in the target). It writes
`reports/models/champion.pkl` and registers `rp-champion/Production` in `mlruns/`.
The `select_champion`, `ChampionSelection`, `ChampionArtifact`,
`persist_champion`, and `load_champion` objects live in
`src/retention/evaluation/champion.py` and are unit-tested in
`tests/test_champion.py`; `register_champion` lives in
`src/retention/models/tracking.py` and is tested in `tests/test_tracking.py`.

> **Rung 1 caption:** *The champion ranks who is likely to leave (test AUC-PR
> 0.274 — associational, Rung 1). It does not establish that any feature *causes*
> leaving, nor that intervening *causes* retention; the EV case prices the
> ranking under an assumed p_eff, it does not measure a causal effect.*
