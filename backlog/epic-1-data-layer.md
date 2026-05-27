# Epic 1 — Data Layer + Feature Engineering

**Prey:** rp-prey-002 (create when pa-warehouse age column ships)
**Status:** ⏸ Blocked on `paw-prey-005` — age column + mutability metadata
**Effort:** 2–3 days
**Public artifact at end:** Dual-cohort feature catalog + Data Card draft + leakage audit signed off

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 1 detail only.

---

*Read from pa-warehouse, build the dual-cohort feature catalog, document the Data Card.*

**Why this epic:** The dual-cohort design (HRIS-only vs hybrid) is the controlled experiment — the analytical contribution of the project. Getting this layer wrong invalidates everything downstream.

**Epic 1 Definition of Done:**
- `src/retention/data/load.py` pulls `marts.v_attrition_features` from BigQuery into a pandas DataFrame
- CSV snapshot saved to `data/processed/` for offline reproducibility
- `src/retention/features/cohorts.py` cleanly splits HRIS-only vs hybrid feature sets
- `src/retention/features/catalog.py` documents every feature with mutability metadata (consumed by DiCE in Epic 6)
- Train/val/test split with temporal awareness (no leakage)
- Data Card draft at `docs/data_card.md`
- Unit tests for data loading and cohort filtering pass (`pytest tests/test_data_load.py tests/test_features.py`)
- `notebooks/01_data_exploration.ipynb` ships with: schema check, missingness analysis, base rate calc, dual-cohort feature counts, target leakage check
- **Critical gates pass:** schema contract validation green (Story 1.1.8), leakage MUST-PASS gate green (Story 1.5.5), BigQuery cost discipline (Story 1.1.9)

### STORY 1.1 — BigQuery data loader
- [ ] 1.1.1 — In `src/retention/data/load.py`, write `load_attrition_features(project_id: str, snapshot_to_csv: bool = True) -> pd.DataFrame`
- [ ] 1.1.2 — Use `pandas-gbq.read_gbq` with query: `SELECT * FROM \`pa-warehouse-prod.marts.v_attrition_features\` ORDER BY employee_id` — **ORDER BY for deterministic ordering (Risk 6 mitigation)**
- [ ] 1.1.3 — Service account auth via `GOOGLE_APPLICATION_CREDENTIALS` env var (same pattern as pa-warehouse)
- [ ] 1.1.4 — On first load, save snapshot to `data/processed/v_attrition_features_YYYY-MM-DD.csv` (parameterize date)
- [ ] 1.1.5 — Add `load_attrition_features_local(csv_path: Path)` fallback for offline reproducibility — reviewers without BigQuery access can still run the pipeline
- [ ] 1.1.6 — Write `tests/test_data_load.py` testing schema (column names + types) against the integration contract from `docs/integration_contract.md`
- [ ] 1.1.7 — **What you'll learn:** Dual-path data loading (cloud + local CSV) is the production pattern for portfolio repos — it shows you think about reviewer experience, not just your own laptop.

- [ ] 1.1.8 — **🔴 Contract validation test (Risk 2 mitigation):**
  - Write `tests/test_integration_contract.py::test_v_attrition_features_schema`
  - Query: `SELECT * FROM \`pa-warehouse-prod.marts.v_attrition_features\` LIMIT 0` (free — returns schema only, zero data scanned)
  - Parse columns + types into a dict
  - Assert against the schema declared in `docs/integration_contract.md` (load via a parser or hardcoded constant)
  - **Fail loudly in CI if drift detected.** Test must be skippable if no BigQuery creds (use `pytest.mark.skipif`).
  - **When pa-warehouse touches `v_attrition_features`:** they MUST update `integration_contract.md` AND ping rp project. Add this to the paw-prey Win Conditions whenever a future paw-prey touches the mart.

- [ ] 1.1.9 — **🟡 BigQuery cost discipline (Risk 8 mitigation):**
  - In `src/retention/data/load.py`, add `--dry-run` flag to `load_attrition_features()` that wraps the query in BigQuery's `dry_run=True` API call
  - Logs the estimated bytes scanned + estimated cost BEFORE running the actual query
  - Default behavior: log estimate, proceed automatically
  - Strict mode (`--cost-confirm` flag): exit if estimate > threshold ($0.01 default), require manual confirmation
  - Document billing alert at $5 threshold in `docs/architecture.md`
  - Add `mlruns/` and `data/processed/*.csv` to `.gitignore` if CSVs are large

### STORY 1.2 — Feature catalog with mutability metadata
- [ ] 1.2.1 — In `src/retention/features/catalog.py`, define `FEATURE_CATALOG` as a list of dataclass instances (extended 2026-05-21 per hunt review Tier 1 fix — leakage_audited + snapshot_date for semantic leakage tests):
  ```python
  from datetime import date

  @dataclass(frozen=True)
  class FeatureSpec:
      name: str
      dtype: str  # 'numeric' | 'categorical' | 'boolean' | 'date'
      source: str  # 'HRIS' | 'survey' | 'derived'
      mutable: bool  # for DiCE actionability constraint
      protected: bool  # for fairness audit (gender, age, etc.)
      cohort: Literal['both', 'hybrid_only']
      snapshot_date_offset_months: int  # months BEFORE observation_window_end at which this feature value is measured
                                        # 0 = measured AT window end (the "current state" features)
                                        # +N = measured N months BEFORE window end (the "_lNm" lag features)
                                        # value used in tests/test_no_leakage.py for semantic temporal check
      leakage_audited: bool  # MUST be True before the feature appears in any training set. False → fail leakage gate
      leakage_rationale: str  # required non-empty string when leakage_audited=True — why this feature is safe (e.g., "computed at snapshot_date_offset_months=6 from observation_window_end, well before label_event_date")
      description: str
  ```
- [ ] 1.2.2 — Populate `FEATURE_CATALOG` for every column in `v_attrition_features`. Include:
  - **HRIS features (both cohorts):** tenure_months, compa_ratio, performance_tier, job_level, dept_id, manager_id, location_id, time_since_last_promotion, succession_readiness_count
  - **Survey features (hybrid cohort only):** enps_rolling_avg, enps_trend_slope, last_response_category, response_count_l12m, survey_decay_score
  - **Protected attributes:** gender (immutable), age_at_window_close (immutable), tenure_band (derived from tenure, mutable=False since age advances naturally)
  - **Derived features:** comp_change_count_l12m, manager_changes_count_l12m, dept_changes_count_l12m
- [ ] 1.2.3 — Add helper functions: `get_mutable_features() -> list[str]`, `get_protected_features() -> list[str]`, `get_cohort_features(cohort: str) -> list[str]`
- [ ] 1.2.4 — Write `tests/test_features.py::test_catalog_coverage` — every column in the loaded DataFrame must be in `FEATURE_CATALOG` (catches schema drift — complements Story 1.1.8 contract test)

### STORY 1.3 — Dual-cohort filtering
- [ ] 1.3.1 — In `src/retention/features/cohorts.py`, write `split_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]` returning `{'hris_only': df_hris, 'hybrid': df_hybrid}`
- [ ] 1.3.2 — HRIS-only cohort: drop survey-derived columns per `FEATURE_CATALOG`
- [ ] 1.3.3 — Hybrid cohort: keep all features
- [ ] 1.3.4 — Both cohorts must have **identical row indices** — only column sets differ. This guarantees apples-to-apples comparison in Epic 2.
- [ ] 1.3.5 — Write `tests/test_features.py::test_cohort_row_alignment` — assert `df_hris.index.equals(df_hybrid.index)`
- [ ] 1.3.6 — **What you'll learn:** Apples-to-apples comparison requires identical *samples*, not just identical evaluation metrics. Most public repos compare models on *different* train/test splits and conclude one is better — that's a methodology error.

### STORY 1.4 — Train/val/test split with temporal awareness
- [ ] 1.4.1 — In `src/retention/data/split.py`, write `temporal_split(df: pd.DataFrame, target_col: str, val_size: float = 0.15, test_size: float = 0.15) -> tuple[pd.DataFrame, ...]`
- [ ] 1.4.2 — Sort by `observation_window_end` (or similar temporal anchor). Train = oldest rows, test = newest. **No random shuffling.** This prevents future-leak.
- [ ] 1.4.3 — Stratify by target only within each temporal block to preserve class ratios
- [ ] 1.4.4 — Save split indices to `data/processed/splits.parquet` for reproducibility
- [ ] 1.4.5 — Write `tests/test_features.py::test_split_no_temporal_leak` — assert all train timestamps < all test timestamps
- [ ] 1.4.6 — **What you'll learn:** Random shuffling in HR data leaks the future. Senior reviewers grep for `train_test_split(shuffle=True)` and reject on sight.

### STORY 1.5 — Target leakage audit (🔴 MUST-PASS GATE for Epic 2)
**Why elevated:** Risk 3 — temporal leakage is the single most common failure mode in attrition modeling and silently inflates AUC. Model looks great on test, deploys to noise. Portfolio narrative dies if a reviewer catches it.

- [ ] 1.5.1 — Write `notebooks/01_data_exploration.ipynb` cell that scans features for any column derived from post-target events
- [ ] 1.5.2 — Specific checks: does `time_since_last_promotion` use the exit_date? does `comp_change_count_l12m` include post-exit comp events?
- [ ] 1.5.3 — Document any flagged features and decide: drop, transform, or accept
- [ ] 1.5.4 — Add the audit findings to `docs/data_card.md` "Known Limitations" section

- [ ] 1.5.5 — **🔴 Leakage MUST-PASS test gate (Risk 3 + r01 + r02 fix 2026-05-21 — three-layer defense):**
  - **Layer 1: Explicit allowlist (human audit).** Write `tests/test_no_leakage.py::test_explicit_leakage_allowlist`
    - Every feature in the training set MUST appear in `FEATURE_CATALOG` with `leakage_audited=True`
    - The `leakage_rationale` field MUST cite the EXACT pa-warehouse SQL/dbt model definition that produces the column (e.g., `"computed in dbt/models/core/dim_employee.sql line 47, snapshot_date set to MIN(effective_from) where effective_from <= observation_window_end"`)
    - **Generic rationales fail review.** Strings like `"looks safe"`, `"computed before label"`, or `"audited"` without specific dbt model references fail this test. Force the citation.
    - Run validation in test: `assert "dbt/models/" in rationale.lower() OR "_core.yml" in rationale.lower()`
  - **Layer 2: Semantic temporal check.** Write `tests/test_no_leakage.py::test_feature_snapshot_before_label`
    - For every row, for every feature column, the feature's `snapshot_date` (from `FEATURE_CATALOG` `snapshot_date_offset_months`) must be such that `row.observation_window_end - offset_months <= row.observation_window_end < row.label_event_date`
    - Requires the per-feature `snapshot_date_offset_months` metadata from Story 1.2 (FeatureSpec dataclass)
  - **Layer 3: Empirical correlation check (r02 fix — catches what human review misses).** Write `tests/test_no_leakage.py::test_no_suspicious_single_feature_correlation`
    - For each feature column, compute Spearman correlation with the label (`voluntary_exit_label`) on the training set
    - Assert `abs(corr) < 0.70` for every feature. Single features with correlation >0.7 to the label are highly suspicious — likely leakage or near-direct label encoding.
    - **Exception:** mark known-high-correlation features explicitly in `FEATURE_CATALOG` with `expected_correlation_above_threshold=True` AND rationale (e.g., "tenure_months is mechanically correlated with exit timing — this is desired signal, not leakage"). The test passes these features only when explicitly marked.
    - This catches leakage even when allowlist marking is wrong (human error) and patterns are missed.
  - **Layer 4: Pattern-based safety net.** Write `tests/test_no_leakage.py::test_no_termination_features` and `tests/test_no_leakage.py::test_no_post_label_aggregations` as in r01.
  - **All 4 layers MUST PASS before Epic 2 starts.** Add to Epic 2 prereqs.
  - **Why defense in depth:** allowlist relies on human discipline (Layer 1); semantic check is metadata-driven (Layer 2); empirical correlation catches human errors and unknown patterns (Layer 3); pattern-matching is the trivial safety net (Layer 4). Each layer catches a different failure class. r02 added Layer 3 (the missing empirical detector).

### STORY 1.6 — Data Card v1
- [ ] 1.6.1 — Create `docs/data_card.md` using the Google Dataset Datasheet template (simplified) or HuggingFace's data card format. Sections:
  - **Dataset name:** v_attrition_features
  - **Source:** pa-warehouse synthetic generation (cross-link to pa-warehouse's generator)
  - **Composition:** N employees, N attrition events, base rate, observation window
  - **Generation methodology:** lognorm tenure, Gaussian copula, causal pay correlation, benchmark targets
  - **Intended use:** retention prediction model training + fairness research + interpretability demonstration
  - **Forbidden use:** real-world HR decisions, hiring decisions, performance evaluations, individual surveillance
  - **Known limitations:** synthetic data limits external validity, no real-world deployment validation, demographics simplified
  - **Maintenance:** regenerated when pa-warehouse generator updates; versioned by CSV snapshot date
- [ ] 1.6.2 — This is the first version — final polish lands in Epic 8

### STORY 1.7 — Data exploration notebook
- [ ] 1.7.1 — `notebooks/01_data_exploration.ipynb` covers:
  - Schema check (assert against `FEATURE_CATALOG`)
  - Missingness heatmap
  - Base rate computation (overall + per cohort + per protected attribute)
  - Feature distributions (histograms grouped by target)
  - Correlation matrix (Spearman) — flag high-correlation pairs that ALE will need to handle
  - Dual-cohort row alignment verification
  - Temporal split verification (no leakage)
  - Survey response rate per cohort (sanity check)
- [ ] 1.7.2 — Save key figures to `reports/figures/` for README inclusion
- [ ] 1.7.3 — Notebook should `import` from `src/retention/`, not duplicate logic

- [ ] 1.7.4 — **🟡 pa-warehouse benchmark re-validation post-paw-prey-005 (Tier 2 fix 2026-05-21 — hunt review):**
  - paw-prey-005 added `birth_date` / `age_at_hire` columns to the pa-warehouse generator. The existing pa-warehouse benchmarks (KM median exit ≈ 2.4yr; Hellinger distance per column < 0.1; censoring rate ∈ [55%, 75%]) were calibrated against the OLD generator output.
  - Compute the same benchmarks against the NEW data inside this notebook:
    - Kaplan-Meier median time-to-exit (via `lifelines` or scikit-survival `kaplan_meier_estimator`)
    - Hellinger distance per column vs. published pa-warehouse benchmarks
    - Censoring rate (`event_observed == False` fraction)
  - **If benchmarks regressed:** flag in `docs/data_card.md` Known Limitations + escalate to paw-prey-005 author for investigation. **Do NOT proceed to Epic 2** with regressed data quality — model results would be untrustworthy.
  - **If benchmarks still hold:** document confirmation in `docs/data_card.md` Versioning section ("validated against paw-prey-005 schema 2026-XX-XX")
  - **Why this matters:** Adding `birth_date` is a generator change. Generator changes can subtly shift distributions (e.g., if age correlates with tenure via hire-date filtering). paw-prey-005 Story 5.1.5 re-runs the validation harness on the pa-warehouse side; this surfaces the cross-project verification on the rp side as a hard check before Epic 2.

**Epic 1 Checkpoint:** Data loaded, cohorts split, catalog documented, splits committed, Data Card drafted, exploration notebook narrates the dataset. **Critical gates: schema contract green (1.1.8), leakage tests green (1.5.5 — all 4 layers passing), BigQuery cost discipline in place (1.1.9), pa-warehouse benchmark re-validation done (1.7.4).** Ready for modeling.

**Epic 1 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 1

| Risk | Mitigation | Where |
|---|---|---|
| 🔴 2 — pa-warehouse contract drift | Schema validation test in CI | Story 1.1.8 |
| 🔴 3 — Temporal leakage | MUST-PASS test gate (3 tests) blocking Epic 2 | Story 1.5.5 |
| 🟡 6 — Reproducibility (BigQuery determinism) | `ORDER BY employee_id` in query | Story 1.1.2 |
| 🟡 8 — BigQuery cost spiral | Dry-run cost estimation + CSV snapshot strategy | Story 1.1.9 |
