# Epic 7 — Causal Slice (Optional)

**Prey:** rp-prey-008 (only create if Epic 7 is actually started)
**Status:** ⏸ OPTIONAL — explicitly skippable
**Effort:** 2–3 days
**Public artifact at end:** Causal notebook with Rung 1/Rung 2 boundary (only if executed)

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 7 detail only.

---

> **🔵 Risk 14 — Clean-skip rule (READ THIS BEFORE STARTING EPIC 7):**
>
> Epic 7 is optional and starting-half-built is worse than skipping cleanly. **Before you start any Story below**, decide explicitly:
>
> - **If skipping (recommended unless energy is high at Epic 7 time):**
>   1. Do NOT create `src/retention/causal/` subpackage (Epic 0 Story 0.2.2 already excludes it from scaffold)
>   2. Do NOT add `dowhy` or `econml` deps via `uv add` (they were pinned in Epic 0 Story 0.1.7 behind `[causal]` extra — never install the extra)
>   3. Do NOT create `notebooks/07_causal_slice.ipynb` (it's listed in repo structure but only as future placeholder)
>   4. Add to README → Limitations: *"Causal inference deferred — see future work. The R3 campaign documented the DoWhy + EconML pattern for HR retention; an implementation is out of Wave 1 scope. The intervention-category audit (Epic 5 Story 5.8) handles the operational fairness concern in the meantime."*
>   5. Skip to Epic 8 cleanly
>
> - **If executing:** create `src/retention/causal/` subpackage NOW (`__init__.py` only), install causal extras (`uv add dowhy econml`), then proceed with Stories 7.1–7.4. **Halfway through is the worst possible state — commit to finishing if you start.**

---

*The cleanest single integration gap per G3. SHAP + DoWhy with explicit Rung 1/Rung 2 boundary.*

**Why this epic:** Optional means *the project can ship without it*. But: G3 confirmed no public HR repo combines SHAP + DoWhy with epistemic boundary discipline. If you have the energy, this epic is the moat.

**Epic 7 Definition of Done:**
- One scoped treatment chosen (e.g., `comp_change → voluntary_attrition` using existing pa-warehouse data — no overtime needed)
- DoWhy backdoor ATE estimated
- EconML Double ML refutation
- Explicit Rung 2 framing: "this estimates the average causal effect *of this one treatment*, not the model's overall behavior"
- `notebooks/07_causal_slice.ipynb` ships as a separate notebook (not blocking other epics)

### STORY 7.1 — Treatment scoping
- [ ] 7.1.1 — Pick treatment: `received_comp_change_l12m` (binary) → `voluntary_attrition_within_12m` (binary)
- [ ] 7.1.2 — Rationale: comp_change is the most policy-relevant binary HR intervention available without schema changes
- [ ] 7.1.3 — Document confounders: tenure, performance_tier, department, manager — all measured, all adjusted

### STORY 7.2 — DoWhy backdoor ATE
- [ ] 7.2.1 — `from dowhy import CausalModel`
- [ ] 7.2.2 — Define the causal graph (DAG) using the confounders identified
- [ ] 7.2.3 — Estimate ATE via backdoor adjustment with linear regression
- [ ] 7.2.4 — Report point estimate + confidence interval
- [ ] 7.2.5 — Run DoWhy's `refute_estimate` with placebo treatment + random common cause

### STORY 7.3 — EconML Double ML refutation
- [ ] 7.3.1 — `from econml.dml import CausalForestDML`
- [ ] 7.3.2 — Use the same confounders
- [ ] 7.3.3 — Compare ATE point estimates from DoWhy linear vs EconML CausalForestDML
- [ ] 7.3.4 — If they agree (within CI), reinforce the conclusion; if they disagree, investigate

### STORY 7.4 — Causal notebook with Rung 1/Rung 2 boundary
- [ ] 7.4.1 — `notebooks/07_causal_slice.ipynb` starts with a section titled "Switching from Rung 1 to Rung 2"
- [ ] 7.4.2 — Explain: "Epics 1-6 estimate P(attrition | features) — Rung 1, associational. This notebook estimates E[attrition | do(comp_change=1)] - E[attrition | do(comp_change=0)] — Rung 2, interventional, conditional on our DAG being correct."
- [ ] 7.4.3 — Caveat clearly: "If our DAG misses a confounder (unobserved heterogeneity in employee preferences, market conditions, manager quality), the ATE is biased."
- [ ] 7.4.4 — Final paragraph: "What this tells HR: if you give an employee a comp increase, the *causal* effect on their 12-month attrition probability is [X percentage points]. The prediction model in Epic 3 tells you *who to give it to*; this analysis tells you *whether it works on average*."

**Epic 7 Checkpoint:** Optional but high-leverage causal slice complete. **OR cleanly skipped with limitations doc updated.**

**Epic 7 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 7

| Risk | Mitigation | Where |
|---|---|---|
| 🔵 14 — Causal half-built (worse than skip) | Explicit clean-skip rule at top of Epic 7 file | Epic header |
