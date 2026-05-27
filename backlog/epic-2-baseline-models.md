# Epic 2 — Baseline Models (LR + GBM + EBM)

**Prey:** rp-prey-003 (create when Epic 1 ships)
**Status:** ⏸ Blocked on Epic 1
**Effort:** 3–4 days
**Public artifact at end:** Comparison notebook with apples-to-apples table + MLflow UI screenshot

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 2 detail only.

---

*Three models, apples-to-apples, two cohorts. The first analytical contribution.*

**Why this epic:** This is where the dual-cohort experiment delivers its result. The output is a comparison table that either confirms or refutes R1's claim that survey features are necessary. Either result is a strong portfolio narrative.

**Epic 2 Definition of Done:**
- `src/retention/models/lr.py` (logistic regression) + `gbm.py` (XGBoost) + `ebm.py` (InterpretML EBM) modules
- Each trained on both cohorts with identical preprocessing
- Comparison table: AUC-PR, precision@10%, calibration error (Brier), interpretability tier — for 6 model×cohort cells
- `notebooks/02_baselines_lr_gbm_ebm.ipynb` narrates the experiment with R1 citations and honest result reporting
- Unit tests for each model wrapper (`pytest tests/test_models.py`)
- **MLflow tracks all 6 runs with sortable UI; screenshot ready for README**
- **Reproducibility smoke test (Story 2.8) green: training pipeline runs twice on same seed, byte-identical outputs**
- **Integration + data quality tests (Story 2.6 extended) cover full pipeline correctness, not just module fits**

### STORY 2.1 — Preprocessing pipeline
- [ ] 2.1.1 — In `src/retention/features/transforms.py`, build a `sklearn.compose.ColumnTransformer`:
  - Numeric: `StandardScaler`
  - Categorical: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
  - Boolean: passthrough
- [ ] 2.1.2 — Wrap in a `Pipeline` so the same preprocessing applies to all three models
- [ ] 2.1.3 — Fit only on training data — transform val/test. Critical: scaler stats from train only.
- [ ] 2.1.4 — Test: `tests/test_features.py::test_preprocessing_no_test_leak`

### STORY 2.2 — Logistic regression baseline
- [ ] 2.2.1 — In `src/retention/models/lr.py`, write `train_lr(X_train, y_train, cohort: str) -> Pipeline`
- [ ] 2.2.2 — `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)` — `class_weight='balanced'` is the R1-approved imbalance treatment (NOT SMOTE)
- [ ] 2.2.3 — Wrap preprocessing + model in single Pipeline
- [ ] 2.2.4 — Fit on train, predict on val and test
- [ ] 2.2.5 — Persist model to `reports/models/lr_{cohort}.pkl` (joblib)
- [ ] 2.2.6 — **What you'll learn:** LR with `class_weight='balanced'` re-weights the loss function — equivalent intuition to SMOTE without distorting the feature distribution. Calibration stays clean.

### STORY 2.3 — XGBoost gradient boosting
- [ ] 2.3.1 — In `src/retention/models/gbm.py`, write `train_gbm(X_train, y_train, cohort: str, params: dict | None = None) -> XGBClassifier`
- [ ] 2.3.2 — Default params: `n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=(neg/pos), random_state=SEED, eval_metric='aucpr'`
- [ ] 2.3.3 — Use `scale_pos_weight` for imbalance (NOT SMOTE)
- [ ] 2.3.4 — Train with `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=30` to prevent overfitting
- [ ] 2.3.5 — Persist model + feature names + training metadata
- [ ] 2.3.6 — Skip hyperparameter tuning at this story — Epic 3 handles it via nested CV

### STORY 2.4 — EBM (Explainable Boosting Machine)
- [ ] 2.4.1 — In `src/retention/models/ebm.py`, write `train_ebm(X_train, y_train, cohort: str) -> ExplainableBoostingClassifier`
- [ ] 2.4.2 — `from interpret.glassbox import ExplainableBoostingClassifier; ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)`
- [ ] 2.4.3 — **EBM preprocessing — explicitly documented as separate from LR/GBM (Tier 1 fix 2026-05-21 — hunt review):** EBM handles categorical encoding internally via its native binning. Building a separate preprocessing pipeline (numeric scaling only, no OneHotEncoder) preserves EBM's interpretability advantage AND uses each model's native feature handling. **This means the comparison is NOT raw-input-identical across models** — see Story 2.5.6 for the honest reframing.
- [ ] 2.4.4 — Persist model
- [ ] 2.4.5 — **What you'll learn:** EBM (a GA²M — Generalized Additive Model with pairwise interactions) is interpretable-by-design. Same accuracy class as GBM, fully interpretable globally. G3 found 12 of 13 repos skip this — closing a universal gap. The preprocessing tradeoff (Story 2.4.3) is the price of EBM's intrinsic interpretability — a tradeoff worth documenting, not hiding.

### STORY 2.5 — Apples-to-apples comparison
- [ ] 2.5.1 — `notebooks/02_baselines_lr_gbm_ebm.ipynb`: train all 6 cells (LR × {hris_only, hybrid}, GBM × {hris_only, hybrid}, EBM × {hris_only, hybrid})
- [ ] 2.5.2 — On test set, compute: AUC-PR, AUC-ROC (secondary), precision@10%, precision@20%, recall@10%, recall@20%, Brier score
- [ ] 2.5.3 — Build comparison DataFrame (6 rows × 7+ metric columns)
- [ ] 2.5.4 — Visualize: grouped bar chart per metric, color-coded by cohort
- [ ] 2.5.5 — Save to `reports/figures/baseline_comparison.png`
- [ ] 2.5.6 — **Honest write-up (Tier 1 fix 2026-05-21 — hunt review reframing):** state which cohort × model won, by how much, and whether the hybrid-vs-HRIS-only gap is statistically meaningful at this sample size. R1 predicts hybrid wins — confirm or refute with receipts. **Critical methodology note:** the comparison is "apples-to-apples WITHIN each model's native feature representation" — LR + GBM share the OneHotEncoded pipeline; EBM uses its native categorical handling per Story 2.4.3 + InterpretML guidance. Cross-model AUC-PR comparison is quantitatively valid (each model is fit on its native input — the model architecture choice IS what's being compared, not a counterfactual "same input" experiment); cross-model interpretability comparison is qualitative. Document this explicitly in the notebook narrative AND in `docs/methodology.md → Cross-Model Comparison Methodology`. Pre-empts the senior reviewer critique: *"you're comparing apples-to-oranges and calling it apples-to-apples."* Acknowledged tradeoff > hidden tradeoff.

### STORY 2.6 — Tests for model wrappers + integration tests (extended)
- [ ] 2.6.1 — `tests/test_models.py::test_lr_fits_on_tiny_data` — fixture with 50 rows, assert model fits without error and `.predict_proba` shape is correct
- [ ] 2.6.2 — `tests/test_models.py::test_gbm_uses_scale_pos_weight` — assert `model.get_params()['scale_pos_weight']` is the expected ratio
- [ ] 2.6.3 — `tests/test_models.py::test_ebm_fits` — same shape check
- [ ] 2.6.4 — `tests/test_models.py::test_no_smote_in_pipeline` — assert no `imblearn` import in `src/retention/models/`

- [ ] 2.6.5 — **🟡 Pipeline integration test (Risk 7 mitigation):**
  - `tests/test_integration.py::test_pipeline_smoke`
  - Fixture: 100-row synthetic dataset (deterministic seed) — covers all expected feature types
  - Full pipeline: load → cohort split → temporal split → preprocess → train (LR + GBM + EBM) → predict
  - Assertions: predictions have correct shape, all probabilities ∈ [0, 1], no NaN, distribution is not degenerate (assert std > 0.01)
  - Runs in <30 seconds; lives in CI

- [ ] 2.6.6 — **🟡 Data quality test — prediction sanity (Risk 7 mitigation):**
  - `tests/test_data_quality.py::test_prediction_calibration_sanity`
  - Train GBM on fixture, predict on held-out 30-row subset
  - Assert: mean predicted probability is within ±5% of fixture base rate (catches calibration collapse)
  - Assert: max - min predicted probability > 0.2 (catches degenerate models)
  - **Why:** 70% src/ coverage doesn't catch a model that returns "always 0.5". Sanity tests do.

- [ ] 2.6.7 — **🟡 No-leakage-at-predict test (Risk 7 mitigation):**
  - `tests/test_data_quality.py::test_no_leakage_at_predict_time`
  - Build fixture where some "training" rows have timestamps AFTER some "test" rows (deliberately broken)
  - Call `temporal_split()` — assert it correctly orders by timestamp (no shuffle leak)
  - Complement to Story 1.5.5 unit-level tests, validates integration

### STORY 2.7 — MLflow experiment tracking integration
**Why this story:** Some Senior People Analytics Scientist / Data Scientist JDs (Visier, Lattice tier) explicitly ask for MLflow. ~1h of effort across Epic 2 produces a recognizable UI artifact for the README and a real ATS keyword for cloud/MLOps-aware JDs. Cost is trivial; signal is real.

- [ ] 2.7.1 — Verify `mlflow = "^2.12"` is in `pyproject.toml` (added in Story 0.1.5 — confirm; if Epic 0 already shipped without it, add now and re-run `uv sync`)
- [ ] 2.7.2 — In `src/retention/models/tracking.py`, write a context manager wrapper: `def log_run(run_name: str, params: dict, metrics: dict, artifact_paths: list[str]) -> None` using `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.log_artifact()`
- [ ] 2.7.3 — Retrofit each of the 6 training cells from Story 2.5 (LR/GBM/EBM × hris_only/hybrid) to wrap with `log_run()` — log: hyperparams, AUC-PR, AUC-ROC, Brier score, precision@10%, precision@20%, model `.pkl` artifact
- [ ] 2.7.4 — Add Makefile target so reviewers can launch the UI from a fresh clone:
  ```makefile
  mlflow-ui:
  	uv run mlflow ui --backend-store-uri mlruns
  ```
- [ ] 2.7.5 — Take a clean screenshot of the MLflow UI showing all 6 runs side-by-side (sortable by AUC-PR). Save to `reports/figures/mlflow_experiment_view.png` for README inclusion (Epic 8).
- [ ] 2.7.6 — Add `mlruns/` to `.gitignore` (local store is git-noisy — only the screenshot ships with the repo)
- [ ] 2.7.7 — Document the choice in `docs/methodology.md`: *"We use MLflow with a local-file backend store (no hosted tracking server). This keeps reproducibility clean — every cloner gets a fresh tracking store from their own training runs — and avoids hosted-service dependency in a portfolio context. Production deployments would swap the backend to a server-mode tracking URI."*
- [ ] 2.7.8 — **What you'll learn:** MLflow's value is not the UI — it's the discipline of logging every run's params/metrics/artifacts to the same store. Catches silent regressions (you tuned a hyperparam and AUC dropped) that you'd miss eyeballing notebook outputs. Senior practitioners track every meaningful run, not just the final one.

- [ ] 2.7.9 — **🟢 MLflow fresh-clone reproduction docs (Risk 11 mitigation):**
  - In `README.md`, add subsection "Reproducing the MLflow experiment view":
    > "Step 1: `make install` — sync dependencies
    > Step 2: `make train` — runs the 6 training cells (LR/GBM/EBM × hris_only/hybrid), populates `mlruns/`
    > Step 3: `make mlflow-ui` — opens browser to localhost:5000 with sortable comparison
    > Note: `mlruns/` is gitignored; the UI is empty until you run `make train`."
  - Embed both the screenshot AND the reproduction steps — fast viewers see the screenshot, serious reviewers can reproduce

- [ ] 2.7.10 — **🟢 MLflow Model Registry — register the champion (Tier 2 fix 2026-05-21 — hunt review):**
  - After Story 3.6 (champion model selection in Epic 3) completes, call `mlflow.register_model(model_uri, "rp-champion")` to register the champion in the local MLflow registry
  - Tag the registered model with stage `Production` (local-context only — there's no actual deployment, but the lifecycle signal matters)
  - In Story 8.5 (writeback), load the champion via `mlflow.pyfunc.load_model('models:/rp-champion/Production')` rather than directly from disk — demonstrates registry usage
  - Screenshot the MLflow Registry view (separate from the runs view) — save to `reports/figures/mlflow_registry_view.png` for README inclusion
  - Document the choice in `docs/methodology.md → MLflow Setup`: *"We use MLflow's Model Registry (not just run logging) to demonstrate champion-model lifecycle awareness. The registry tracks which run produced the deployed model and supports staged promotion (Staging → Production → Archived) that a production deployment would actually exercise."*
  - ~15 minutes of work, signals staff-IC depth beyond junior MLflow usage

### STORY 2.8 — Reproducibility smoke test (🟡 Risk 6 mitigation — full)
**Why this story:** SEED = 42 is set, but does every library + every code path respect it? This test catches non-determinism that would otherwise surface as "numbers different on your machine."

- [ ] 2.8.1 — In `src/retention/evaluation/repro.py`, write `run_training_twice_assert_identical(cohort: str = 'hybrid') -> bool`:
  - Load tiny fixture (100 rows)
  - Run full train pipeline twice with `set_global_seed()` between runs
  - Capture predictions on test set both times
  - Assert: `np.allclose(predictions_run_1, predictions_run_2, atol=1e-10)`
- [ ] 2.8.2 — Add Makefile target:
  ```makefile
  repro:
  	uv run pytest tests/test_reproducibility.py -v
  ```
- [ ] 2.8.3 — Write `tests/test_reproducibility.py`:
  - `test_lr_reproducible` — assert LR predictions byte-identical across runs
  - `test_gbm_reproducible` — assert GBM predictions byte-identical
  - `test_ebm_reproducible` — assert EBM predictions byte-identical
  - `test_pipeline_end_to_end_reproducible` — full pipeline two runs, identical outputs
- [ ] 2.8.4 — If any test fails, debug:
  - XGBoost: confirm `random_state=SEED` everywhere
  - EBM: confirm `random_state=SEED`
  - sklearn: confirm all stochastic estimators have `random_state=SEED`
  - BigQuery: confirm `ORDER BY` clause (Story 1.1.2)
  - numpy operations: confirm `set_global_seed()` called before any random op
- [ ] 2.8.5 — Add to CI workflow — runs on every push
- [ ] 2.8.6 — **What you'll learn:** Reproducibility is one of those things that "should just work" and frequently doesn't. The test is cheap; the failure when discovered late is expensive.

### STORY 2.9 — OHE sensitivity check for EBM (🟢 r02 hunt fix 2026-05-21 — strengthens Tier 1 r01 fix #1)
**Why this story:** The r01 fix reframed the EBM comparison as "apples-to-apples within each model's native feature representation." That's defensible but not bulletproof — a staff IC will still ask "did you also run EBM on OHE input to isolate model architecture from preprocessing choice?" Adding the sensitivity check answers that question proactively.

- [ ] 2.9.1 — Train a SECOND EBM variant on the SAME OneHotEncoded pipeline as LR + GBM (using the standard preprocessing from Story 2.1, not the EBM-native pipeline from Story 2.4.3)
- [ ] 2.9.2 — Compare the two EBM variants:
  - EBM with native categorical handling (Story 2.4, the primary)
  - EBM with OneHotEncoded input (this story, the sensitivity check)
- [ ] 2.9.3 — Report on the same metrics as Story 2.5: AUC-PR, AUC-ROC, precision@10%, precision@20%, Brier
- [ ] 2.9.4 — Document the comparison in `docs/methodology.md → EBM Preprocessing Sensitivity`:
  > *"EBM-native (binning own categoricals) achieved AUC-PR X.XX; EBM-on-OHE achieved Y.YY. The X% delta is consistent with InterpretML's documentation that native binning is generally equivalent-or-slightly-better than OHE on categorical features. We report the primary EBM result using native handling but cite this sensitivity check to confirm the comparison is robust to preprocessing choice."*
- [ ] 2.9.5 — Add to the comparison table in Story 2.5.5 — now 7 rows instead of 6 (LR/GBM/EBM-native/EBM-OHE × hris_only/hybrid where applicable)
- [ ] 2.9.6 — **What you'll learn:** the rigorous comparison isolates each variable. Showing "EBM beats LR" is one comparison; showing "EBM beats LR REGARDLESS of preprocessing choice" is a stronger claim. The sensitivity check pre-empts the "you only ran one preprocessing variant" critique.

### STORY 2.10 — JD market re-validation checkpoint (🟢 r02 hunt fix 2026-05-21 — revisits deferred T2.2 from r01)
**Why this story:** The BACKLOG was designed against 2026-05-12 JobSpy data. By Epic 2 ship (~6 days in), data is 18+ days old; by Epic 3 ship, ~24-30 days. Markets shift. Best practice: re-validate the stack before committing to Epic 3+ time investment.

- [ ] 2.10.1 — **Timing:** run this at Epic 2 close, BEFORE opening Epic 3 prey. 30 minutes total.
- [ ] 2.10.2 — Pull a fresh JobSpy run (or use existing tools) for current People Analytics + Data Scientist JDs in target markets (US remote, LATAM)
- [ ] 2.10.3 — Compare to 2026-05-12 synthesis:
  - Are forecasting / predictive modeling JDs still at 50–66%? (the gap the project closes)
  - Have new keywords risen above 20% that the stack DOESN'T address? (e.g., a new ML framework, a new BI tool, a new fairness library)
  - Has Snowflake/Looker requirement shifted? (paw-prey-005 should still close it)
- [ ] 2.10.4 — If material shift (any new >20% keyword in stack gaps): pause Epic 3, evaluate whether to add a story for the new gap OR continue as planned
- [ ] 2.10.5 — If no shift: proceed to Epic 3 with the validated stack
- [ ] 2.10.6 — Document the check + outcome in `career/jobspy-analysis-2026-06-XX.md` (date of the re-validation run)
- [ ] 2.10.7 — **What you'll learn:** Plans built on snapshot market data have a half-life. Re-validating at major milestones is the rigorous version of "build during the latency window" — make sure the latency window is still pointing at the right market.

**Epic 2 Checkpoint:** Three models (+EBM sensitivity check) trained on two cohorts. First comparison table exists. Notebook narrates the result honestly. **MLflow tracks all 6+1 runs with sortable UI; screenshot ready for README.** **Reproducibility smoke test green. Integration + data quality tests cover pipeline correctness. JD market re-validation done before opening Epic 3.**

**Epic 2 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 2

| Risk | Mitigation | Where |
|---|---|---|
| 🟡 6 — Reproducibility (full smoke test) | Pipeline runs twice, byte-identical assertion | Story 2.8 |
| 🟡 7 — Test coverage gaps (integration + data quality) | Pipeline smoke + prediction sanity + no-leakage-at-predict | Stories 2.6.5–2.6.7 |
| 🟢 11 — MLflow fresh-clone UI confusion | Reproduction docs in README | Story 2.7.9 |
