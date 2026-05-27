# Epic 6 — Interpretability Section

**Prey:** rp-prey-007 (create when Epic 2 ships — Epic 6 parallelizable with Epic 3/4/5)
**Status:** ⏸ Blocked on Epic 2 **AND on paw-prey-005 mutability metadata delivery** (Tier 1 r02 fix 2026-05-21 — hunt review surfaced that DiCE Story 6.5 needs `mutable: bool` from FEATURE_CATALOG, which is populated from paw-prey-005)
**Effort:** 4–5 days
**Public artifact at end:** SHAP + ALE + DiCE + EBM intrinsic notebook + adversarial-SHAP caveat documented

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 6 detail only.

---

*The R4 + G3 senior stack. Universal gaps filled. Adversarial SHAP caveat documented.*

**Why this epic:** A senior reviewer reads the interpretability section asking "does this person understand what they're showing?" The answer is in the Rung 1 captions, the ALE-vs-PDP choice, the EBM-as-challenger framing, and whether DiCE counterfactuals respect actionability.

**Epic 6 Definition of Done:**
- `src/retention/explainability/` modules complete
- EBM-vs-GBM interpretability comparison (intrinsic vs post-hoc)
- SHAP beeswarm with Rung 1 caption template
- SHAP waterfall plots for 3 specific cases (high-risk, low-risk, edge case)
- ALE plots for top-3 correlated features (using package selected in Epic 0 Story 0.1.11 — alibi OR dalex)
- DiCE counterfactuals on 2 employees respecting mutability metadata
- Per-group calibration reused from Epic 5
- Adversarial SHAP (Slack 2020) caveat in `docs/methodology.md`
- `notebooks/06_interpretability.ipynb` ships

### STORY 6.1 — EBM intrinsic interpretation
- [ ] 6.1.1 — From the EBM model trained in Story 2.4, extract feature shape functions
- [ ] 6.1.2 — Plot: top-10 features with their EBM partial-dependence shapes (the GA² component)
- [ ] 6.1.3 — Save to `reports/figures/ebm_shapes.png`
- [ ] 6.1.4 — Side-by-side with GBM SHAP global importance for the same features — does the ranking agree?
- [ ] 6.1.5 — **Disagreement protocol (Tier 2 r01 + r02 fix 2026-05-21 — escape hatch added):** If top-3 SHAP features ≠ top-3 EBM features, **do not silently pick one.** Follow this protocol:
  1. **Quantify the disagreement:** compute rank correlation (Spearman) between SHAP and EBM top-10 rankings. Document the value.
  2. **If rank correlation > 0.7:** general agreement, document the minor differences, proceed with SHAP for downstream (Stories 6.2-6.7) but cite EBM as the corroborating signal.
  3. **If rank correlation 0.4–0.7:** real disagreement. Investigate WHICH features disagree and WHY (likely correlated feature pairs — Story 6.4 ALE plots will help). Document the investigation. Pick the interpretation with the stronger theoretical defense (EBM is intrinsic and faithful; SHAP can be misled by correlated features per Story 6.6 adversarial-SHAP caveat).
  4. **If rank correlation < 0.4:** major disagreement. **This is a model integrity signal — stop and investigate before Epic 6 ships.** Likely causes: training instability, feature correlation pathology, EBM hyperparameter mismatch.
  5. **🟢 Escape hatch (r02 fix 2026-05-21):** if investigation in step 4 cannot resolve the disagreement within **4 hours of focused effort**, document the irreducible disagreement explicitly. Proceed with EBM as primary interpretation (intrinsic-faithful per R4) and note the SHAP limitation in `README → Interpretability summary`: *"SHAP and EBM disagreed on top features (Spearman ρ = X). Investigation surfaced feature correlation pathology in [features Y, Z] consistent with adversarial-SHAP risk (Slack 2020). We proceed with EBM's interpretation as primary and document SHAP's divergence as a known limitation. ALE plots (Story 6.4) handle the correlated-feature pair directly."* This protects against the "stuck investigating forever" failure mode.
  6. **Threshold provenance (r02 fix):** the 0.7 / 0.4 rank-correlation thresholds are **illustrative defaults**, not literature-cited values. Adjust based on observed distribution if your specific run produces an unusual pattern. The protocol structure matters more than the exact numbers — what matters is that disagreement is detected, quantified, investigated, and either resolved or honestly documented.
- [ ] 6.1.6 — Document the chosen interpretation + protocol step that resolved disagreement (or the escape-hatch outcome) in `docs/methodology.md` → "Interpretability cross-validation"
- [ ] 6.1.7 — **What you'll learn:** Senior practitioners never present a single interpretation as ground truth — they cross-validate. SHAP and EBM disagreeing is informative, not embarrassing. The protocol IS the signal — including its escape hatch for irreducible cases.

### STORY 6.2 — SHAP global (beeswarm) with Rung 1 captions
- [ ] 6.2.1 — In `src/retention/explainability/shap_explain.py`, write `compute_shap_values(model, X) -> shap.Explanation`
- [ ] 6.2.2 — Use `shap.TreeExplainer` for XGBoost (faster + exact)
- [ ] 6.2.3 — Use `shap.LinearExplainer` for LR (correctness — `galafis` variant-correctness pattern)
- [ ] 6.2.4 — Compute SHAP values on a sample of 500 test rows (full set is wasteful)
- [ ] 6.2.5 — Plot beeswarm: top-10 features, dots colored by feature value, x-axis SHAP value
- [ ] 6.2.6 — Caption template (Rung 1 discipline, per R3):
  > "The model weighted [feature] heavily in attrition predictions. Higher [feature] values pushed predictions toward 'will leave' (positive SHAP). **This is associational, not causal — the model is explaining its own predictions, not the data-generating process. We have not estimated what happens to attrition if we intervene on [feature].**"
- [ ] 6.2.7 — Save to `reports/figures/shap_beeswarm.png`

### STORY 6.3 — SHAP local (waterfall) on 3 cases
- [ ] 6.3.1 — Pick 3 employees from the test set:
  - Highest predicted probability (high-risk case)
  - Lowest predicted probability among actual leavers (false negative case)
  - Edge case near the decision threshold
- [ ] 6.3.2 — Generate waterfall plots for each
- [ ] 6.3.3 — Save to `reports/figures/shap_waterfall_case_{1,2,3}.png`
- [ ] 6.3.4 — Annotate each with a paragraph: "Why this employee is flagged" (Rung 1 caption, no intervention claim)

### STORY 6.4 — ALE plots (top-3 correlated features)
- [ ] 6.4.1 — From the correlation matrix in Story 1.7, pick the top-3 most-correlated feature pairs
- [ ] 6.4.2 — **Use the ALE package selected in Epic 0 Story 0.1.11:** either `from alibi.explainers import ALE` OR `import dalex` (whichever survived the install verification)
- [ ] 6.4.3 — Plot 1D ALE for each of the top-5 important features (per SHAP)
- [ ] 6.4.4 — Plot 2D ALE for one chosen interaction (the most-correlated pair)
- [ ] 6.4.5 — Save to `reports/figures/ale_top_features.png` and `reports/figures/ale_2d_interaction.png`
- [ ] 6.4.6 — **What you'll learn:** PDP (partial dependence) averages over feature distributions and is misleading when features are correlated. ALE (Accumulated Local Effects) is the correlation-robust alternative — it conditions on local feature value rather than averaging globally. G3: 12/13 repos use PDP. Switching to ALE is a 30-minute senior-signal upgrade.

### STORY 6.5 — DiCE counterfactuals with actionability constraint
- [ ] 6.5.1 — In `src/retention/explainability/counterfactuals.py`, write `generate_counterfactuals(model, employee_row, n: int = 3, features_to_vary: list[str]) -> pd.DataFrame`
- [ ] 6.5.2 — `features_to_vary` = `get_mutable_features()` from the catalog — DiCE only varies features marked `mutable=True`. **This is the actionability constraint.**
- [ ] 6.5.3 — `import dice_ml; d = dice_ml.Dice(...)`
- [ ] 6.5.4 — Generate 3 counterfactuals per employee, sparsity-optimized (fewest features changed)
- [ ] 6.5.5 — Apply to the same 2 employees as Story 6.3 (or pick 2 different)
- [ ] 6.5.6 — Save to `reports/figures/dice_counterfactuals.png` (table with employee → original features → counterfactual changes → predicted probability)
- [ ] 6.5.7 — Caption: *"Counterfactual: if these changes happened, the model would predict 'will stay'. These are actionable changes only — gender, age, and tenure are excluded by design. The counterfactual is a *what-if for the model*, not a causal claim about real intervention effect."*
- [ ] 6.5.8 — **What you'll learn:** DiCE counterfactuals without the mutability constraint produce nonsense ("become younger by 10 years"). The mutability catalog from Story 1.2 pays off here.

### STORY 6.6 — Adversarial SHAP caveat (Slack 2020)
- [ ] 6.6.1 — In `docs/methodology.md`, add a section "Adversarial SHAP and the limits of post-hoc explanation"
- [ ] 6.6.2 — Cite Slack et al. 2020 "Fooling LIME and SHAP" — adversarial models can hide bias from SHAP by exploiting correlated features
- [ ] 6.6.3 — Document our defenses: EBM as the intrinsic-interpretation cross-check (Story 6.1), ALE plots that handle correlated features (Story 6.4), proxy detection (Story 5.9)
- [ ] 6.6.4 — Conclusion paragraph: "SHAP is necessary but not sufficient for fairness assurance. We use it as one of three corroborating signals."

### STORY 6.7 — SHAP-fairness linkage paragraph
- [ ] 6.7.1 — In the notebook, write a paragraph connecting SHAP top-features to the proxy detection from Story 5.9
- [ ] 6.7.2 — Example: "Department appears as the #3 SHAP feature globally. Our proxy detector (`src/retention/fairness/proxy.py`) flagged it as correlated with gender (MI=0.X) — meaning the model may be using department as a gender proxy. We do not drop the feature, but we report this transparently and recommend monitoring the intervention-category audit (Story 5.8) for disparate impact."

### STORY 6.8 — Interpretability notebook
- [ ] 6.8.1 — `notebooks/06_interpretability.ipynb` covers everything in this Epic
- [ ] 6.8.2 — Final summary: how SHAP + ALE + DiCE + EBM together form a *cross-validated* interpretation, not a single black-box explanation
- [ ] 6.8.3 — Final paragraph: cite the Rudin critique briefly — "We default to EBM where intrinsic interpretation suffices, and only escalate to GBM+SHAP when EBM's performance is meaningfully lower. In our experiment (Epic 2), EBM lost AUC-PR by X — we chose GBM and accept the post-hoc explanation tradeoff with the caveats above."

**Epic 6 Checkpoint:** Interpretability section bar-clearing. Universal gaps filled. SHAP-EBM disagreement protocol followed (with escape hatch if irreducible).

**Epic 6 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 6

| Risk | Mitigation | Where |
|---|---|---|
| 🟡 5 — alibi swap execution (if Story 0.1.11 triggered) | Story 6.4.2 references Epic 0 outcome and uses dalex if needed | Story 6.4.2 |
