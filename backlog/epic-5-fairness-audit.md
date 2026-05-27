# Epic 5 — Fairness Audit + Mitigation

**Prey:** rp-prey-006 (create when Epic 2 ships — Epic 5 parallelizable with Epic 3/4)
**Status:** ⏸ Blocked on Epic 2 **AND on paw-prey-005 age column delivery** (Tier 1 fix 2026-05-21 — hunt review surfaced this dual blocker; Epic 1 and Epic 5 both depend on paw-prey-005)
**Effort:** 4–5 days
**Public artifact at end:** Fairness notebook + audit matrix + Fairlearn mitigation comparison

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 5 detail only.

---

*The R2 minimum-viable senior stack. Compound attribute, calibration parity, intervention-category audit.*

**Why this epic:** Hiring managers at Visier/Lattice/Workday test for this section before they read modeling depth. R2's recipe is the bar. G3 confirmed 13/13 public HR repos miss most of it.

**Epic 5 Definition of Done:**
- `src/retention/fairness/` modules: audit, compound, mitigation, intervention, proxy
- 3-metric audit (demographic parity + equalized odds + **calibration parity** as primary) on protected attributes + compound attribute
- Chouldechova impossibility theorem paragraph in `docs/methodology.md`
- Top-k 4/5 rule audit
- Intervention-category demographic audit (R2 open gap)
- Fairlearn `ExponentiatedGradient` in-processing mitigation compared to baseline
- Reweighing pre-processing comparison
- Department-as-proxy flag if SHAP surfaces it
- `notebooks/05_fairness_audit.ipynb` ships the fairness narrative
- **Minimum group size handling formalized (Story 5.1.5) — groups with n<30 collapsed or reported as "insufficient data"**
- **Fairlearn ExponentiatedGradient pre-tested for convergence on small data (Story 5.7.0)**

### STORY 5.1 — Compound attribute construction
- [ ] 5.1.1 — In `src/retention/fairness/compound.py`, write `build_compound_attribute(df: pd.DataFrame, attrs: list[str], bins: dict[str, list]) -> pd.Series`
- [ ] 5.1.2 — Build `gender × age_band × tenure_band` compound attribute. age_band = [22-30, 31-40, 41-50, 51-65]. tenure_band = [0-1yr, 1-3yr, 3-7yr, 7+yr].
- [ ] 5.1.3 — Result: ~24 (3×4×4) compound groups. Some groups will be tiny — report counts.
- [ ] 5.1.4 — Drop groups with < 20 samples from the audit (statistical noise — flag but skip)

- [ ] 5.1.5 — **🟢 Minimum group size handling formalized (Risk 9 mitigation):**
  - Write `compute_group_sizes(df: pd.DataFrame, attr: str) -> pd.DataFrame` returning per-group counts
  - Define threshold constants: `MIN_GROUP_SIZE_AUDIT = 30` (per-cell fairness metrics unstable below), `MIN_GROUP_SIZE_REPORT = 5` (don't even mention groups below this — privacy + noise)
  - For each protected attribute audit, run `compute_group_sizes` first. If any group < 30: collapse to "Other" OR explicitly mark as "Insufficient data — N=X"
  - Document the rule in `docs/methodology.md` → Fairness section:
    > "Groups with fewer than 30 samples are excluded from per-cell fairness metrics — calibration error and equalized-odds gaps are unstable below this threshold. Groups with fewer than 5 samples are not reported by name (privacy + signal-to-noise). Excluded groups are flagged in the audit table as 'Insufficient data (N=X)' for transparency."
  - Add `tests/test_fairness.py::test_min_group_size_enforced` — fixture with one tiny group, assert it's flagged not metric-ed

### STORY 5.2 — 3-metric audit using Fairlearn MetricFrame
- [ ] 5.2.1 — In `src/retention/fairness/audit.py`, build `MetricFrame` with metrics:
  - `selection_rate` (demographic parity)
  - `true_positive_rate` and `false_positive_rate` (equalized odds)
  - `expected_calibration_error` (calibration parity — **primary metric per R1+R2**)
- [ ] 5.2.2 — Audit on single protected attributes: gender, age_band, tenure_band
- [ ] 5.2.3 — Audit on the compound attribute
- [ ] 5.2.4 — Report `difference()` and `ratio()` from MetricFrame for each metric
- [ ] 5.2.5 — Visualize: heatmap of metric × group (rows=metric, cols=group). Save to `reports/figures/fairness_audit_matrix.png`.

### STORY 5.3 — Calibration parity as primary metric (with reliability diagrams per group)
- [ ] 5.3.1 — For each protected attribute group, generate a reliability diagram on the same axes (one line per group)
- [ ] 5.3.2 — Compute per-group ECE — report max/min/range
- [ ] 5.3.3 — Save to `reports/figures/calibration_per_group.png`
- [ ] 5.3.4 — **Document why calibration parity is primary:** "Demographic parity would force the model to flag protected groups at the same rate regardless of true risk — anti-fair when true rates differ. Equalized odds enforces equal error rates — useful but secondary. Calibration parity is the minimum guarantee that the *probability* means the same thing across groups — without it, decision thresholds become group-specific without anyone noticing."

### STORY 5.4 — Chouldechova impossibility theorem paragraph
- [ ] 5.4.1 — In `docs/methodology.md`, write a paragraph stating: "Chouldechova (2017) and Kleinberg-Mullainathan-Raghavan (2017) prove that you cannot simultaneously satisfy calibration parity + equalized FPR + equalized FNR when base rates differ between groups. Since base rates of voluntary attrition differ between gender/age/tenure groups in our data (cite values), we choose calibration parity as primary and accept some FPR/FNR disparity, documenting it transparently. This is the Visier convention per G3."
- [ ] 5.4.2 — Cite the papers
- [ ] 5.4.3 — Cross-reference the paragraph from the README

### STORY 5.5 — Top-k 4/5 rule audit
- [ ] 5.5.1 — For each protected attribute, compute selection rate within the top 10% (Epic 3.1's precision@k cutoff)
- [ ] 5.5.2 — Check 4/5 rule: min_group_selection_rate / max_group_selection_rate ≥ 0.80
- [ ] 5.5.3 — Report per attribute; flag failures
- [ ] 5.5.4 — Visualize: bar chart of top-k selection rates per group with 4/5 threshold line. Save to `reports/figures/four_fifths_rule.png`.
- [ ] 5.5.5 — **What you'll learn:** The 4/5 rule (EEOC) is a *legal* threshold, not a statistical one. Failing it doesn't prove discrimination, but it triggers regulatory scrutiny. Reporting compliance is a senior signal.

### STORY 5.6 — Reweighing pre-processing
- [ ] 5.6.1 — In `src/retention/fairness/mitigation.py`, write `apply_reweighing(X_train, y_train, protected: pd.Series) -> np.ndarray` — returns sample weights
- [ ] 5.6.2 — Use Fairlearn or implement manually: weight = `P(y) × P(group) / P(y, group)` per Kamiran-Calders 2012
- [ ] 5.6.3 — Train GBM with `sample_weight=reweigh_weights`
- [ ] 5.6.4 — Re-audit — compare fairness metrics before/after reweighing
- [ ] 5.6.5 — Document the AUC-PR cost (if any) — fairness mitigation usually trades accuracy

### STORY 5.7 — ExponentiatedGradient in-processing (the `ctriz` pattern)

- [ ] 5.7.0 — **🟢 ExponentiatedGradient convergence pre-test (Risk 9 mitigation — companion):**
  - In a scratch notebook BEFORE 5.7.1: fit `ExponentiatedGradient(LogisticRegression(), constraints=EqualizedOdds())` on a 500-row subset
  - Time the fit. Confirm convergence within reasonable iterations (default 50)
  - **If non-convergent OR >2 min on small data:** document limitation, consider alternative (`GridSearch` with constraint, or skip in-processing and rely on Story 5.6 reweighing alone)
  - Update `docs/methodology.md` with the pre-test result and the chosen mitigation strategy

- [ ] 5.7.1 — `from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds`
- [ ] 5.7.2 — Train GBM wrapped in `ExponentiatedGradient(estimator, constraints=EqualizedOdds())` on the compound attribute
- [ ] 5.7.3 — Compare three models on the same eval set: baseline GBM → reweighed GBM → in-processed GBM
- [ ] 5.7.4 — Table: AUC-PR + ECE + max-min disparity per metric × group
- [ ] 5.7.5 — Pick the model that best trades fairness vs performance — document the choice
- [ ] 5.7.6 — **What you'll learn:** G3 found `ctriz/HR_Attrition_Analysis` is the only public repo using `ExponentiatedGradient`. It's not exotic — Fairlearn ships it. Most repos just don't reach for it.

### STORY 5.8 — Intervention-category demographic audit (R2 open gap)
- [ ] 5.8.1 — In `src/retention/fairness/intervention.py`, write `categorize_predictions(y_proba: np.ndarray, thresholds: dict) -> pd.Series` — maps probabilities to intervention tiers
- [ ] 5.8.2 — **🔴 Define tiers calibrated to Epic 3 outputs (Tier 1 fix 2026-05-21 — hunt review; NOT arbitrary thresholds):**
  - **Immediate 1:1:** `predicted_proba ≥ EV_optimal_threshold` — the threshold from Story 3.4 that maximizes Expected Value at default `p_eff = 0.30`. Empirically calibrated.
  - **Quarterly check-in:** `EV_breakeven_threshold ≤ predicted_proba < EV_optimal_threshold` — Story 3.4's `breakeven_p_eff` boundary. The "intervention is EV-positive at higher p_eff but not at our default" zone.
  - **Passive monitoring:** `base_rate ≤ predicted_proba < EV_breakeven_threshold` — above baseline risk but below intervention-economic threshold.
  - **No action:** `predicted_proba < base_rate` — at or below baseline risk; the model adds no signal here.
  - Each tier boundary is **empirically grounded via Epic 3 outputs** (cited in `docs/methodology.md → Tier Calibration`). Pre-empts the senior reviewer critique: *"why 0.7? you picked it."* Now the answer is: *"because EV is maximized at this threshold per Story 3.4's sensitivity analysis."*
  - **When auditing intervention-category demographic distribution (Stories 5.8.3+), the boundaries are defensible — and the audit is meaningful, not auditing arbitrary numbers.**
- [ ] 5.8.3 — Audit demographic distribution within each intervention tier — is the top tier disproportionately a protected group?
- [ ] 5.8.4 — Report 4/5 rule per intervention tier
- [ ] 5.8.5 — Save to `reports/figures/intervention_demographics.png`
- [ ] 5.8.6 — **What you'll learn:** This is R2's explicit open gap. The argument: even a calibrated, equalized-odds-correct model can produce *interventions* that disproportionately target protected groups simply because the thresholds aren't group-aware. Reporting this is what separates "we did fairness" from "we did fairness with operational receipts."

### STORY 5.9 — Department-as-proxy detector
- [ ] 5.9.1 — In `src/retention/fairness/proxy.py`, write `detect_proxy_features(model, X_train, protected: pd.Series, top_n: int = 5) -> pd.DataFrame`
- [ ] 5.9.2 — Use mutual information between each feature and the protected attribute — flag features with MI > threshold
- [ ] 5.9.3 — Department is a classic proxy in HR (correlated with gender via job-level distribution per pa-warehouse)
- [ ] 5.9.4 — If a flagged feature surfaces in SHAP top-5 (Epic 6), add a paragraph in the notebook explaining the proxy concern

### STORY 5.10 — Fairness notebook
- [ ] 5.10.1 — `notebooks/05_fairness_audit.ipynb` covers: compound attribute → 3-metric audit → calibration per group → 4/5 rule → reweighing comparison → ExponentiatedGradient comparison → intervention-category audit → proxy detection
- [ ] 5.10.2 — Final summary: which mitigation strategy (baseline / reweighed / in-processed) is the *deployed* choice and why
- [ ] 5.10.3 — Cross-link from notebook to `docs/methodology.md` Chouldechova paragraph

### STORY 5.11 — Fairness regression test (🟢 Tier 2 r01 + r02 fix 2026-05-21 — literature-grounded thresholds)
**Why this story:** Once Story 5.6 (Reweighing) or 5.7 (ExponentiatedGradient) reduces disparity, what prevents it from regressing on every subsequent retrain? Without a test, future-Nico (or a future maintainer) retrains the model, fairness silently regresses, and nothing catches it until a reviewer notices. Add the test to CI; make fairness a first-class assertion.

- [ ] 5.11.1 — In `tests/test_fairness_regression.py`, write `test_calibration_disparity_below_literature_threshold`:
  - Load the trained chosen-mitigation model (baseline / reweighed / in-processed per Story 5.10.2)
  - Compute the per-group calibration error (ECE) across protected attribute groups
  - Assert: `max(group_ece) - min(group_ece) ≤ ABSOLUTE_ECE_DISPARITY_THRESHOLD`
- [ ] 5.11.2 — **🟢 Threshold from literature (r02 fix — NOT from own model output):** `ABSOLUTE_ECE_DISPARITY_THRESHOLD = 0.05` as default, cited from established HR fairness practice:
  - **Aequitas** (CMU Auton Lab) — recommends absolute ECE disparity <0.05 for "production-suitable" fairness; <0.02 for "high-trust deployment"
  - **Fairlearn documentation** — uses similar absolute thresholds in its examples
  - **Chouldechova 2017** — implicit threshold via the "fair-enough calibration" definition
  - Document in `src/retention/config.py`: `ABSOLUTE_ECE_DISPARITY_THRESHOLD = 0.05  # Aequitas + Fairlearn convention; see docs/methodology.md Fairness Thresholds section for citation`
  - **Why absolute, not relative:** A relative threshold (10% above own model's output) sets the bar from a moving target — if your model is bad-but-not-getting-worse, the test passes. Absolute thresholds set the bar from external standards — your model is either fair enough by literature consensus or it's not.
- [ ] 5.11.3 — Write `test_no_top_k_4_5_violation`:
  - Compute top-k (k=10%) selection rate per protected group
  - Assert: `min_group_rate / max_group_rate ≥ 0.80` (the 4/5 rule — EEOC legal standard, not arbitrary)
- [ ] 5.11.4 — Write `test_demographic_parity_within_threshold`:
  - Assert: `max(group_selection_rate) - min(group_selection_rate) ≤ 0.10` (Fairlearn's `DEMOGRAPHIC_PARITY_THRESHOLD` default)
- [ ] 5.11.5 — Add all three tests to CI (`.github/workflows/test.yml`) — they run on every push
- [ ] 5.11.6 — **What you'll learn:** Fairness mitigation isn't a one-time accomplishment — it's a *property* of the model that must be re-verified on every train. Most public HR repos either skip fairness or treat it as a one-shot exercise. CI-enforced fairness with literature-grounded thresholds is the senior signal.
- [ ] 5.11.7 — Document the thresholds + their rationale in `docs/methodology.md → Fairness Thresholds`: *"ABSOLUTE_ECE_DISPARITY_THRESHOLD = 0.05 (Aequitas convention); 4/5 rule = 0.80 (EEOC); demographic parity = 0.10 (Fairlearn default). These are external standards, not relative to this model's output. If the deployed model fails these, fairness mitigation must continue — fairness is a property, not a milestone."*
- [ ] 5.11.8 — **🟢 Edge case handling (r02 fix — intervention tier collapse, see Epic 5.8):** if Epic 3's calibrated thresholds collapse the intervention tiers (e.g., `EV_breakeven > base_rate` or thresholds are <0.05 apart), document explicitly: *"Intervention tier separation is below meaningful threshold. Operational deployment would set tier boundaries from budget constraints, not from EV calibration."* Update Story 5.8.2 logic to handle empty tier case gracefully (collapse to 2-tier or 3-tier presentation; don't pretend 4 distinct tiers exist when they don't).

**Epic 5 Checkpoint:** Fairness section production-grade. Three mitigation strategies compared. Intervention-category audit is the differentiating artifact. **Minimum group size rule enforced; ExponentiatedGradient convergence pre-tested. Fairness regression tests live in CI with literature-grounded thresholds (Aequitas + EEOC + Fairlearn). Intervention tier edge cases handled gracefully.**

**Epic 5 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 5

| Risk | Mitigation | Where |
|---|---|---|
| 🟢 9 — Fairness instability on small groups (formalized rule) | `MIN_GROUP_SIZE_AUDIT = 30` threshold + collapse-or-flag logic + tests | Story 5.1.5 |
| 🟢 9 — Fairness instability (ExponentiatedGradient convergence) | Pre-test fit on small subset before Story 5.7.1 | Story 5.7.0 |
