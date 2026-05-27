# Epic 3 — Evaluation Rigor

**Prey:** rp-prey-004 (create when Epic 2 ships)
**Status:** ⏸ Blocked on Epic 2
**Effort:** 2–3 days
**Public artifact at end:** Calibration + precision@k + EV sweep + nested CV notebook

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 3 detail only.

---

*The differentiator. G2 found zero public HR repos with this depth. This is where the project stops looking like a tutorial.*

**Why this epic:** AUC-only evaluation is the strongest junior-tier signal. AUC-PR + precision@k + calibration + EV + nested CV is what a Visier senior IC produces. Every story here closes a gap G2 documented.

**Epic 3 Definition of Done:**
- `src/retention/evaluation/metrics.py` covers AUC-PR, precision@k (k=5%, 10%, 20%), lift, Brier
- `src/retention/evaluation/calibration.py` produces reliability diagrams + calibration error metrics
- `src/retention/evaluation/expected_value.py` implements EV framing with `p_eff ∈ [0.1, 0.9]` sensitivity sweep + breakeven plot
- `src/retention/evaluation/nested_cv.py` runs nested 5×5 CV
- `notebooks/03_evaluation_rigor.ipynb` ships the evaluation narrative
- All metrics tested (`pytest tests/test_evaluation.py`)

### STORY 3.1 — AUC-PR + precision@k + lift
- [ ] 3.1.1 — In `src/retention/evaluation/metrics.py`, write:
  - `auc_pr(y_true, y_proba) -> float` (sklearn's `average_precision_score`)
  - `precision_at_k(y_true, y_proba, k_pct: float) -> float` — top k% by predicted probability, then precision
  - `recall_at_k(y_true, y_proba, k_pct: float) -> float`
  - `lift_at_k(y_true, y_proba, k_pct: float) -> float` — precision_at_k / base_rate
- [ ] 3.1.2 — Tests for each — known-input known-output cases
- [ ] 3.1.3 — **What you'll learn:** precision@k is what HR teams actually use ("I have budget to talk to 50 people, who are the top 50?"). AUC-PR doesn't ask that question; precision@k does.

### STORY 3.2 — Calibration metrics + reliability diagrams
- [ ] 3.2.1 — In `src/retention/evaluation/calibration.py`, write:
  - `brier_score(y_true, y_proba) -> float`
  - `expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float` (ECE)
  - `reliability_diagram(y_true, y_proba, n_bins: int = 10) -> matplotlib.figure.Figure`
- [ ] 3.2.2 — Reliability diagram: 10 equal-probability bins, plot predicted vs observed, 45° reference line
- [ ] 3.2.3 — Per-cohort, per-model reliability diagrams saved to `reports/figures/calibration_*.png`
- [ ] 3.2.4 — **What you'll learn:** A model saying "this employee has 78% attrition probability" only means something if 78% of employees with that prediction actually leave. Calibration is what makes the probability decision-actionable.

### STORY 3.3 — Threshold calibration as controlled experiment (R1 [FLIP-RISK] win + A/B framing)
**Why this story:** This is the project's controlled-experimentation moment. We sweep operating thresholds, measure precision/recall/cost at each, and pick the operating point empirically. README frames this as the **A/B testing equivalent for risk scoring** — the discipline of "treat each candidate threshold as a treatment arm, compare arms on a clean metric, pick the empirically-winning arm." This closes the "A/B testing / experimentation" ATS keyword for Senior PA + Product Analyst JDs without writing any A/B-specific code — it's a framing of work the model already does.

- [ ] 3.3.1 — In `src/retention/models/threshold.py`, write `optimize_threshold(y_true, y_proba, criterion: str = 'f1') -> float`
- [ ] 3.3.2 — Sweep thresholds from 0.05 to 0.95 in 0.01 steps, optimize for F1 (default), F2 (recall-weighted), or EV (Epic 3.4)
- [ ] 3.3.3 — Plot threshold sweep: x-axis threshold, y-axis F1 + precision + recall — **frame as "treatment arm comparison" in the chart caption**
- [ ] 3.3.4 — Save figure to `reports/figures/threshold_sweep.png`
- [ ] 3.3.5 — Document the choice in `docs/methodology.md` with both framings:
  - *"Why we calibrate the threshold instead of resampling with SMOTE."* (R1 [FLIP-RISK] verdict — SMOTE distorts calibration)
  - *"Threshold calibration as a controlled experiment — each candidate threshold is a treatment arm; we measure precision/recall/cost on a held-out test set; the chosen threshold is the empirically-winning arm. This is the A/B testing equivalent for risk scoring models, applied to a single deployed system rather than two competing systems."*
- [ ] 3.3.6 — **What you'll learn:** experimentation vocabulary (treatment arms, controlled comparison, empirical operating point selection) is what Senior Product Analytics + Senior PA Scientist JDs ask for. You're already doing it — naming it correctly is the win. The deeper point: every threshold-selection process in ML is implicitly an experiment; most practitioners just don't name it that way.

### STORY 3.4 — Expected Value framing with sensitivity sweep
- [ ] 3.4.1 — In `src/retention/evaluation/expected_value.py`, define the EV formula:
  ```
  EV = TP × benefit(retention) - FP × cost(intervention) - FN × cost(unprevented_loss)
     = TP × (avg_replacement_cost × p_eff) - FP × cost_intervention - FN × avg_replacement_cost
  ```
- [ ] 3.4.2 — Parameters with defaults + citations: `avg_replacement_cost = 1.5 × annual_salary` (SHRM 2024), `cost_intervention = $2000` (mgr + HRBP time), `p_eff = 0.30` (effectiveness probability — what fraction of flagged-and-treated employees we actually retain)
- [ ] 3.4.3 — `ev_at_threshold(y_true, y_proba, threshold, p_eff, replacement_cost, intervention_cost) -> float`
- [ ] 3.4.4 — `p_eff_sensitivity_sweep(y_true, y_proba, threshold, p_eff_range=(0.1, 0.9)) -> pd.DataFrame` — EV at 9 values of p_eff
- [ ] 3.4.5 — `breakeven_p_eff(y_true, y_proba, threshold) -> float` — the p_eff at which EV crosses zero
- [ ] 3.4.6 — Visualize: EV curve over p_eff range, vertical line at breakeven
- [ ] 3.4.7 — Save to `reports/figures/ev_sensitivity.png`
- [ ] 3.4.8 — **What you'll learn:** R3 named this "the honest extension of saadhna25's E[loss] embryo." Most public repos either skip EV or pick one fake number and report it. The sensitivity sweep is what makes the framing defensible — *we don't know p_eff exactly, here's the range over which the model still pays off*.

### STORY 3.5 — Nested cross-validation
- [ ] 3.5.1 — In `src/retention/evaluation/nested_cv.py`, implement nested 5×5 CV (outer for performance estimate, inner for hyperparameter tuning)
- [ ] 3.5.2 — Use `sklearn.model_selection.StratifiedKFold` for both loops
- [ ] 3.5.3 — Inner loop: hyperparameter search via `GridSearchCV` or `RandomizedSearchCV` with `scoring='average_precision'`
- [ ] 3.5.4 — Outer loop: report mean ± std AUC-PR across 5 folds
- [ ] 3.5.5 — Apply to GBM only (LR has few hyperparameters worth tuning; EBM's defaults are strong)
- [ ] 3.5.6 — Document the difference between flat-CV (overstates) and nested-CV (correct) in `docs/methodology.md`
- [ ] 3.5.7 — **What you'll learn:** Without nested CV, you tune hyperparameters on the same data you evaluate, and the reported performance is optimistically biased. G2 finding: most public repos do flat CV. This is the easy senior signal.

### STORY 3.6 — Evaluation notebook
- [ ] 3.6.1 — `notebooks/03_evaluation_rigor.ipynb` ties everything: nested CV → calibrated GBM → threshold sweep → EV sensitivity → breakeven
- [ ] 3.6.2 — Final summary table: 6 model×cohort cells × metrics (AUC-PR, precision@10%, ECE, optimal threshold, EV at default p_eff, breakeven p_eff)
- [ ] 3.6.3 — Pick the cohort + model combination that best balances AUC-PR + calibration. Document the choice with a paragraph in the notebook ("we proceed with hybrid-cohort GBM because…")
- [ ] 3.6.4 — Save the chosen model + threshold to `reports/models/champion.pkl`

### STORY 3.7 — Mutation testing for evaluation logic (🟢 r02 hunt fix 2026-05-21 — staff-IC test-quality signal)
**Why this story:** Pytest-cov measures lines executed by tests. Mutation testing (`mutmut`) measures whether tests would CATCH bugs. For a "production-grade" claim, mutation testing is the actual senior signal — most public HR repos have coverage % without mutation testing. Adding it on the evaluation module (the highest-stakes correctness code) closes the gap.

- [ ] 3.7.1 — Add `mutmut = "^2.5"` to dev dependencies in `pyproject.toml`
- [ ] 3.7.2 — Configure `mutmut` in `pyproject.toml`:
  ```toml
  [tool.mutmut]
  paths_to_mutate = "src/retention/evaluation/"
  runner = "uv run pytest tests/test_evaluation.py -x -q"
  ```
- [ ] 3.7.3 — Run `uv run mutmut run` on `src/retention/evaluation/` — captures mutations that pass tests (i.e., bugs that wouldn't be caught)
- [ ] 3.7.4 — Inspect mutation survivors: `uv run mutmut results` + `uv run mutmut show <id>`
- [ ] 3.7.5 — For each survivor: either (a) add a test that kills it, or (b) document why the mutation is acceptable (e.g., equivalent mutation)
- [ ] 3.7.6 — Target: <5 surviving mutations on the evaluation module after iteration
- [ ] 3.7.7 — Add `make mutation-test` Makefile target
- [ ] 3.7.8 — Document the result in `docs/methodology.md → Test Quality`: *"Mutation testing on src/retention/evaluation/: N mutations generated, K killed by tests, M survived (M < 5). Surviving mutations documented as acceptable (equivalent mutations or low-risk code paths)."*
- [ ] 3.7.9 — **What you'll learn:** Coverage % is a proxy for test quality; mutation testing measures it directly. A test suite with 90% coverage but 60% mutation kill rate is letting half its bugs through. Senior practitioners care about the latter; junior practitioners report the former.

**Epic 3 Checkpoint:** Evaluation rigor in place. The "champion" model selected with documented reasoning. **Mutation testing on evaluation module passes.** Ready for survival arm.

**Epic 3 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 3

*No new mitigations in Epic 3 — Story 3.3 already carries the A/B testing controlled-experiment framing (added in earlier work). Risks 1, 2, 3, 5, 6, 7, 8, 10 mitigated in Epics 0–2; Risk 9 lands in Epic 5; Risks 11, 12, 14 land later.*
