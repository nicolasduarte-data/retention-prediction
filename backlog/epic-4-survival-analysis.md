# Epic 4 — Survival Analysis Arm + SurvSHAP(t)

**Prey:** rp-prey-005 (create when Epic 3 ships)
**Status:** ⏸ Blocked on Epic 3 (parallelizable with later Epic 3 stories)
**Effort:** 3–4 days
**Public artifact at end:** Survival notebook + SurvSHAP(t) chart + IBS report

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 4 detail only.

---

*The highest-leverage single gap per G3. Zero HR repos use SurvSHAP(t).*

**Why this epic:** Retention is fundamentally a time-to-event problem, not a binary one. The classification arm answers "who will leave?"; the survival arm answers "when?" Senior People Analytics teams need both. SurvSHAP(t) makes feature importance time-varying — a tenured employee's risk drivers differ from a new hire's.

**Epic 4 Definition of Done:**
- `src/retention/models/survival.py` with Cox PH baseline + Random Survival Forest
- C-index + Integrated Brier Score + calibration at 3/6/12-month horizons
- SurvSHAP(t) chart showing time-varying feature importance for top features (OR documented fallback per Story 4.5.1 if Epic 0 Story 0.1.10 triggered)
- `notebooks/04_survival_arm.ipynb` ships the survival narrative

### STORY 4.1 — Survival data preparation
- [ ] 4.1.1 — In `src/retention/data/load.py`, add `load_survival_format(df: pd.DataFrame) -> pd.DataFrame` — converts the classification target to `(event_observed: bool, time_to_event: float)` format expected by scikit-survival
- [ ] 4.1.2 — `time_to_event` for terminated employees = `exit_date - hire_date` in months
- [ ] 4.1.3 — `time_to_event` for active employees = `window_end - hire_date` in months (right-censored)
- [ ] 4.1.4 — `event_observed = True` for voluntary terminations only (we predict regrettable churn, not involuntary)
- [ ] 4.1.5 — Test: assert censoring rate ∈ [55%, 75%] (matches pa-warehouse benchmark)

### STORY 4.2 — Cox proportional hazards baseline
- [ ] 4.2.1 — `from sksurv.linear_model import CoxPHSurvivalAnalysis`
- [ ] 4.2.2 — Train on hybrid cohort (the champion cohort from Epic 3)
- [ ] 4.2.3 — Report hazard ratios for top features — these are interpretable as "multiplier on baseline hazard"
- [ ] 4.2.4 — Save coefficients table to `reports/figures/cox_hazard_ratios.png` (forest plot)
- [ ] 4.2.5 — **What you'll learn:** CoxPH is the LR-of-survival — interpretable lower bound. RSF is the GBM-of-survival — better fit, harder to interpret without SurvSHAP(t).

### STORY 4.3 — Random Survival Forest
- [ ] 4.3.1 — `from sksurv.ensemble import RandomSurvivalForest`
- [ ] 4.3.2 — Default params: `n_estimators=300, max_depth=10, min_samples_leaf=15, random_state=SEED, n_jobs=-1`
- [ ] 4.3.3 — Fit on hybrid cohort
- [ ] 4.3.4 — Save model

### STORY 4.4 — Survival evaluation metrics
- [ ] 4.4.1 — In `src/retention/evaluation/metrics.py`, add survival metric wrappers:
  - `concordance_index(model, X_test, y_test) -> float` (sksurv's `concordance_index_censored`)
  - `integrated_brier_score(model, X_test, y_test, times) -> float` (sksurv's `integrated_brier_score`)
  - `survival_calibration_at_horizon(model, X_test, y_test, horizon_months: float) -> tuple[float, plt.Figure]` — calibration of predicted survival at a specific horizon vs observed Kaplan-Meier
- [ ] 4.4.2 — Report C-index for Cox + RSF
- [ ] 4.4.3 — Report IBS over `[3, 6, 12]` month horizons for both models
- [ ] 4.4.4 — Report calibration at each horizon
- [ ] 4.4.5 — **What you'll learn:** C-index is to survival what AUC is to classification. IBS is the time-varying Brier score — measures probabilistic accuracy across the whole survival curve. G3: only `Naresh1401` reports IBS today. Easy bar.

### STORY 4.5 — SurvSHAP(t) time-varying chart
- [ ] 4.5.1 — **Check Epic 0 Story 0.1.10 outcome:** if survshap installed cleanly, use it. **If fallback was triggered:** use the manual path — `shap.TreeExplainer(rsf.estimators_[0])` wrapped over horizon-specific predictions, then aggregate SHAP values per feature per horizon. Less elegant; still publishable.
- [ ] 4.5.2 — Compute SurvSHAP(t) values for top 5 features in RSF
- [ ] 4.5.3 — Plot: x-axis time (months), y-axis SHAP value, one line per feature. The lines should diverge over time — that's the time-varying story.
- [ ] 4.5.4 — Save to `reports/figures/survshap_time_varying.png`
- [ ] 4.5.5 — Caption template: *"The model weighted [feature] heavily at [time horizon] — associational, not causal."* (Rung 1 discipline per R3)
- [ ] 4.5.6 — Write a paragraph in the notebook: "What this chart says about retention timing." Example narrative: *"Compa-ratio's negative SHAP contribution grows monotonically over 24 months — underpaid employees are increasingly likely to leave the longer they stay. Performance tier's contribution peaks at 6 months then declines — high performers leave fast or stay; the slow-burn departure is rare."*

### STORY 4.6 — Survival notebook
- [ ] 4.6.1 — `notebooks/04_survival_arm.ipynb` covers: data conversion → CoxPH → RSF → evaluation table → SurvSHAP(t)
- [ ] 4.6.2 — Final comparison: classification (Epic 3 champion) vs survival (CoxPH or RSF) — which framing is more honest for HR teams making 1:1 prioritization decisions?
- [ ] 4.6.3 — **Rung 1 discipline:** the conclusion is associational. Document this explicitly.

**Epic 4 Checkpoint:** Survival arm complete. SurvSHAP(t) chart ships — the differentiating artifact per G3 (or fallback per Story 4.5.1).

**Epic 4 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 4

| Risk | Mitigation | Where |
|---|---|---|
| 🔴 1 — survshap fallback execution | Story 4.5.1 references Story 0.1.10 outcome; uses fallback path if needed | Story 4.5.1 |
