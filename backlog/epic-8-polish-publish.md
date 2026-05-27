# Epic 8 — Polish, Publish, Distribute

**Prey:** rp-prey-009 (create when Epics 1–6 ship)
**Status:** ⏸ Blocked on Epics 1–6 (Epic 7 optional)
**Effort:** 4–5 days
**Public artifact at end:** README + Model Card + dashboard integration + Substack post + LinkedIn video

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 8 detail only.

---

*The Loop 4 of rp. Most public repos die here — the code exists but the presentation doesn't catch up.*

**Why this epic:** G1's #1 finding: presentation is the differentiator more than code depth. The same modeling work, presented at "tutorial" quality, is invisible. Presented at "centerpiece" quality, it lands interviews.

**Epic 8 Definition of Done:**
- README polished: governance disclaimer, forbidden uses, methodology, results, fairness, interpretability, limitations, references
- Data Card finalized
- Model Card created
- Architecture diagram (Mermaid or excalidraw)
- Predictions written back to BigQuery
- pa-warehouse Looker Studio Page 5 surfaces predictions
- Substack post draft
- LinkedIn demo video script
- Final code review against the Bar Test
- **Documentation reconciliation pass complete (Story 8.11) — every doc claim verified against code**

### STORY 8.1 — README polish
- [ ] 8.1.1 — Open the README skeleton from Story 0.5 and fill every section
- [ ] 8.1.2 — Top: one-line pitch + live dashboard link + SYNTHETIC banner + paired blog post link
- [ ] 8.1.3 — **Governance disclaimer (Jott2121 voice):**
  > "This model is designed for *decision support*, not *decision making*. It surfaces patterns from synthetic data for educational and methodological demonstration. It is not validated for, and must not be used for, real HR decisions including hiring, firing, performance evaluation, compensation determination, or individual surveillance."
- [ ] 8.1.4 — **Forbidden uses:**
  > "This system is not approved for use in: (a) employment decisions covered by the U.S. Equal Employment Opportunity Commission (EEOC), (b) automated employment decision tools as defined by NYC Local Law 144 (AEDT), (c) high-risk AI systems as defined by EU AI Act Annex III §4. No bias audit has been conducted for production use. Real-world deployment requires re-validation on real data with consultation from legal counsel."
- [ ] 8.1.5 — Methodology section: drop in the 6 Architectural Decision paragraphs from rp-prey-001 (DL non-choice, hypothesis reframe, threshold over SMOTE, calibration parity, MLflow, controlled-experiment framing). Link to `docs/methodology.md` for depth.
- [ ] 8.1.6 — Results: 6 model×cohort comparison table + champion model summary + SurvSHAP(t) chart embed + fairness summary
- [ ] 8.1.7 — Limitations: adversarial SHAP, p_eff uncertainty, synthetic data external validity, IBM-style overfit risk. **Include Story 0.1.10/0.1.11 fallback notes if triggered, Epic 7 skip note if skipped, Story 5.1.5 min-group-size exclusions.**
- [ ] 8.1.7.1 — **🟢 Future Work section (Tier 2 r01 + r02 fix 2026-05-21 — hunt review):** Add a "Future Work" subsection between Limitations and How-to-reproduce. Items:
  - *"**Drift monitoring** — once predictions are live, model degradation needs detection. Production deployment would integrate Evidently AI or NannyML to track feature drift, prediction drift, and calibration drift over time. Out of Wave 1 scope; the local MLflow registry tracks training-time metrics but not deployment-time drift."*
  - *"**SCD2 / shadow-mode writeback** — see `docs/architecture.md` Limitations. The portfolio uses `if_exists='replace'`; production would preserve prediction history."*
  - *"**Containerization (Docker)** — see `docs/architecture.md` Limitations. The portfolio is laptop-runnable via `uv` + `Makefile`. Production-grade reproducibility would ship a Dockerfile."*
  - *"**Real-data validation** — this project ships on synthetic data. Real-world deployment requires re-validation on real HR data with consultation from legal counsel (EEOC, AEDT, EU AI Act Annex III)."*
  - *"**Causal inference (if Epic 7 was skipped)** — the R3 campaign documented DoWhy + EconML patterns for HR retention. An implementation is future work."*
  - *"**A/B testing of model versions** — Story 3.3 frames threshold selection as a controlled experiment. A production deployment would A/B test model versions over time, not just thresholds."*

- [ ] 8.1.7.2 — **🟡 Interview-readiness acknowledgment (r02 hunt fix Tier 2.5.A — strategic gap from devil's advocate):** Add a "What This Project Is and Isn't" subsection near the top of README, between Problem framing and Methodology:
  > *"**What this project is:** a portfolio piece demonstrating production-grade methodology for retention prediction — dual-cohort comparative experiment, R2-recipe fairness audit, SurvSHAP(t) time-varying interpretability, MLflow-tracked experiments. It's the technical depth signal for senior People Analytics roles.*
  >
  > *"**What this project is NOT:** a substitute for live SQL screening, behavioral interview preparation, case study practice, or take-home modeling exercises. People Analytics hiring at the senior tier typically involves all of these in sequence. This portfolio addresses one stage (portfolio-shape signal); other stages require separate preparation. If you're evaluating this for hiring, please pair the repo review with whichever screening stages match your team's process — this artifact alone is not the whole picture."*
  - **Why this honesty matters:** prevents the "the portfolio didn't work" misattribution if Nico fails downstream stages. Reviewers also appreciate the framing — it shows awareness of the broader hiring process.
- [ ] 8.1.8 — References: R1-R4 + G1-G3 + named papers (Pearl, Lundberg-Lee, Rudin, Chouldechova, Slack, Grinsztajn, Shwartz-Ziv, Borisov, Rubenstein)

### STORY 8.2 — Model Card
- [ ] 8.2.1 — Create `docs/model_card.md` using Google's Model Card format
- [ ] 8.2.2 — Sections: intended use, factors (subpopulations), metrics (overall + per group), evaluation data, training data, quantitative analyses, ethical considerations, caveats and recommendations
- [ ] 8.2.3 — Fill from Epics 1-6 outputs

### STORY 8.3 — Data Card finalization
- [ ] 8.3.1 — Open `docs/data_card.md` from Story 1.6 — polish, add the temporal split decision, add the dual-cohort design rationale
- [ ] 8.3.2 — Add a "Versioning" section: which CSV snapshot was used for this experiment, regeneration cadence

### STORY 8.4 — Architecture diagram
- [ ] 8.4.1 — In `docs/architecture.md`, embed a Mermaid diagram showing: `pa-warehouse BigQuery → pandas-gbq → preprocessing → 6 models × 2 cohorts → champion selection → fairness/interpretability/causal arms → predictions → marts.v_attrition_predictions → Looker Studio Page 5`
- [ ] 8.4.2 — Verify the Mermaid renders on GitHub

### STORY 8.5 — Predictions write-back to BigQuery
- [ ] 8.5.1 — In `src/retention/writeback/bigquery.py`, write `write_predictions(predictions: pd.DataFrame, table_name: str = 'marts.v_attrition_predictions') -> None`
- [ ] 8.5.2 — Schema: `employee_id`, `prediction_date`, `predicted_probability`, `predicted_tier` (immediate/quarterly/passive/none), `champion_model`, `champion_cohort`, `feature_set_version`
- [ ] 8.5.3 — Use `pandas-gbq.to_gbq` with `if_exists='replace'` for portfolio simplicity (production would use SCD2)
- [ ] 8.5.4 — Add `make predict` Makefile target

- [ ] 8.5.5 — **🟢 Document SCD2/shadow-mode + Docker deferrals in docs/architecture.md → Limitations (Tier 2 r01 + r02 fix 2026-05-21):**
  - **SCD2/shadow-mode writeback paragraph:** *"The predictions writeback uses `if_exists='replace'` (full-table replacement on every run) for portfolio simplicity. Production deployment would use SCD2 (Slowly Changing Dimension Type 2) — write to `marts.v_attrition_predictions_staging`, validate, then atomic-swap. Or shadow mode — write predictions to a parallel table while the existing one stays live, compare for drift, promote when verified. The portfolio approach is auditable (every run produces a complete snapshot) but loses prediction history across runs. SCD2/shadow preserves history at the cost of more complex writeback logic and additional dataset permissions."*
  - **Containerization paragraph (r02 fix Tier 2.5.C):** *"The project ships as a `uv`-managed Python project running on the user's local machine. We have NOT provided a Dockerfile or containerized reproducibility path. A production-grade portfolio would include `docker run`-able reproduction; the absence here is a known limitation. Reviewers cloning the repo need Python 3.11 + `uv` + a BigQuery service account. A Dockerfile is future work — out of Wave 1 scope. The `Makefile` + lock file + pre-commit hooks are the reproducibility layer for the laptop-based audience."*
  - Cross-reference both from README → Limitations section
  - **Why these matter:** Reviewers grep for deployment-pattern awareness. Mentioning the production pattern (even while deferring it) is the senior signal; using a portfolio shortcut without acknowledgment is the junior signal. Same logic for Docker — acknowledged absence beats unacknowledged absence.

### STORY 8.6 — pa-warehouse Looker Studio Page 5
- [ ] 8.6.1 — In the pa-warehouse Looker Studio dashboard, add a new page "Page 5: Retention Predictions"
- [ ] 8.6.2 — Charts: predicted probability distribution histogram + intervention-tier counts + top-20 highest-risk employees table (with fairness disclaimer banner) + per-department aggregate risk
- [ ] 8.6.3 — Add a fairness disclaimer banner at the top of Page 5
- [ ] 8.6.4 — Test live BigQuery connection works after the write-back

### STORY 8.7 — Substack post draft
- [ ] 8.7.1 — Create `reports/distribution/substack_post_draft.md`
- [ ] 8.7.2 — Outline: hook (the dual-cohort question) → setup (the pa-warehouse pipeline) → result (which cohort won, by how much) → the interpretability moment (one SHAP/SurvSHAP screenshot with Rung 1 caption) → the fairness moment (intervention-category audit screenshot) → call to action (the repo link)
- [ ] 8.7.3 — Word target: 1200-1800 words
- [ ] 8.7.4 — **What you'll learn:** A paired blog post triples a portfolio repo's visibility per G1. The post is not a code dump — it's the *narrative the recruiter doesn't have time to extract from the README*.

### STORY 8.8 — LinkedIn demo video script
- [ ] 8.8.1 — Create `reports/distribution/linkedin_video_script.md`
- [ ] 8.8.2 — Target 90 seconds. Sections (15s each): hook → problem → approach → one chart (SurvSHAP(t) or fairness audit) → one finding → call to action with repo URL
- [ ] 8.8.3 — Plan recording: screen capture of the Looker dashboard + voiceover; or a Loom-style face-on-cam segment

### STORY 8.9 — Bar Test code review
- [ ] 8.9.1 — Self-review against the Bar Test: "Would this notebook survive code review at Visier/Lattice/Workday PA team?"
- [ ] 8.9.2 — Specific checks:
  - [ ] No SMOTE in the codebase
  - [ ] AUC-PR reported (not just AUC-ROC)
  - [ ] Calibration plots overall + per group
  - [ ] EV with sensitivity sweep
  - [ ] SHAP captions with Rung 1 discipline
  - [ ] ALE used over PDP for correlated features
  - [ ] EBM tested as challenger
  - [ ] DiCE counterfactuals respect mutability constraint
  - [ ] SurvSHAP(t) shipped (OR fallback documented)
  - [ ] Compound fairness attribute audited (with min-group-size rule enforced)
  - [ ] Calibration parity as primary fairness metric
  - [ ] Chouldechova paragraph
  - [ ] Top-k 4/5 rule audit
  - [ ] Intervention-category demographic audit
  - [ ] Department-as-proxy paragraph
  - [ ] Adversarial SHAP caveat
  - [ ] Data Card + Model Card
  - [ ] Governance disclaimer + forbidden uses
  - [ ] DL non-choice paragraph + 3 citations (Grinsztajn / Shwartz-Ziv / Borisov)
  - [ ] MLflow experiment tracking + UI screenshot
  - [ ] Threshold-as-controlled-experiment framing
  - [ ] Reproducibility smoke test passing
  - [ ] Schema contract validation test passing
  - [ ] Leakage tests passing
  - [ ] Tests + lint + CI green
- [ ] 8.9.3 — If any check fails, fix before announcing
- [ ] 8.9.4 — Cross-reference each check against G1/G2/G3 anti-patterns — name which one each item closes

### STORY 8.10.0 — Pre-public security scan (🟡 Tier 2 fix 2026-05-21 — hunt review)
**Why this story:** Best-in-class practice before any first public push: scan for committed credentials, secrets, internal project IDs, personal info. ~5 minutes; catches embarrassing leaks before they become permanent in git history.

- [ ] 8.10.0.1 — Run `uv run trufflehog filesystem .` (install via `uv add --dev trufflehog` if not present) — scans for high-entropy strings, AWS keys, GCP service account keys, GitHub tokens, etc.
- [ ] 8.10.0.2 — Manually grep for: `AKIA` (AWS access key prefix), `ya29` (Google OAuth token prefix), `ANTHROPIC_API_KEY`, `pa-warehouse-prod` (BigQuery project ID — should be in env vars or a placeholder, not committed)
- [ ] 8.10.0.3 — Grep for personal info: full email addresses, phone numbers, real names in comments
- [ ] 8.10.0.4 — Verify `.gitignore` properly excludes: `.env`, `.env.local`, `~/.dbt/profiles.yml`, `mlruns/`, `data/raw/*.csv` (if any contain real data), `*.pkl` with model artifacts containing training data
- [ ] 8.10.0.5 — If any finding: rewrite history (`git filter-branch` or BFG Repo-Cleaner) BEFORE first push. Once pushed, secrets are compromised — rotate them.
- [ ] 8.10.0.6 — **What you'll learn:** Public push is a one-way door for secrets. The 5-minute scan is the cheapest insurance against a leak that costs hours to rotate and an embarrassing audit trail in your repo's commit history.

### STORY 8.10 — Public announce
- [ ] 8.10.1 — Push final commit, tag `v1.0.0`
- [ ] 8.10.2 — Publish Substack post
- [ ] 8.10.3 — Record + publish LinkedIn video
- [ ] 8.10.4 — Update `career/portfolio.md`, `career/skills.md` (Survival Analysis, Fairlearn, SHAP, DiCE, scikit-survival, MLflow to the lists), `SESSION_STATE.md`
- [ ] 8.10.5 — Update `MEMORY.md` with the project completion entry
- [ ] 8.10.6 — Write the project summary in `LORE.md`

- [ ] 8.10.6.5 — **🟢 Pre-publish external review (r02 hunt fix 2026-05-21 — Tier 2.5.B):**
  - **The "Bar Test" (Story 8.9) is self-assessment.** Self-assessment is the weakest validation. The Bar Test invokes "senior reviewer at Visier/Lattice/Workday" as the standard but Nico doesn't actually know that standard from the inside.
  - **Before Story 8.10.1 (push final commit, tag v1.0.0):** get one informal external review from someone in the target tier.
  - Who to ask: LinkedIn connection at Visier / Lattice / Workday / GitLab / Personio / Revolut / Figma's PA team. Even an indirect connection (1st-degree → 2nd-degree introduction) works. 30 minutes of their time.
  - What to ask: *"I'm preparing to publish a People Analytics retention model portfolio piece. Could you spend 30 minutes reviewing the README + one notebook of your choice and flagging anything that wouldn't pass your team's internal code review?"*
  - Apply their feedback BEFORE the public push
  - If no connection is available within a reasonable window: post the repo (private/draft mode) to relevant Slack communities (Locally Optimistic, People Analytics community) and request a 30-min review there
  - **Why this matters:** the public ship is a one-way door for first impressions. One external review beats the most rigorous self-Bar-Test, because reviewers SEE THINGS YOU MISS.
  - Document the reviewer + their feedback (anonymized if needed) in `docs/methodology.md → External Review` — adds another senior signal (you sought and applied feedback, not just shipped from your own head)

- [ ] 8.10.7 — **🟢 GitHub repo metadata setup (Tier 2 fix 2026-05-21 — hunt review):**
  - **Repo description (Settings → About):** *"Senior People Analytics retention model — dual-cohort controlled experiment, R2 fairness audit, SurvSHAP(t) time-varying interpretability, MLflow experiment tracking. Research-grounded (R1-R4 + G1-G3, 7 sessions). Synthetic data."*
  - **Repo topics (Settings → About → Topics):** `people-analytics`, `hr-data-science`, `attrition-prediction`, `machine-learning`, `fairness`, `interpretability`, `mlflow`, `xgboost`, `lightgbm`, `survival-analysis`, `shap`, `dice-ml`, `fairlearn`, `dbt`, `bigquery`, `python`
  - **Social preview image (Settings → Social preview):** generate a 1280×640 PNG from the SurvSHAP(t) chart OR the MLflow UI screenshot. Tool: `figma` or `excalidraw`. The social preview is what appears when someone shares the repo URL on LinkedIn / Slack / Twitter — currently defaults to a generic GitHub avatar.
  - **About sidebar (Settings → About → Website):** link to live Looker Studio dashboard (Page 5) — the primary visual artifact
  - **About sidebar (Settings → About → Topics):** confirm topics applied (above)
  - **Pin to GitHub profile (profile page → Customize your pins):** add this repo alongside `pa-warehouse` and `workforce-analytics`. The three-pin combo tells the story: warehouse → analytics → ML.
  - **(Optional) Enable Discussions** for inquiries — only if you want public interaction. Default: skip.
  - **(Optional) GitHub Pages from `docs/`** — turns the model card / data card into a browsable mini-site. Defer to post-launch unless you have ~30 min spare.
  - **Why this matters:** A reviewer's first 5-second impression of the repo comes from the metadata strip (description + topics + social preview), not the code. Good metadata signals polish; default metadata signals "abandoned-side-project." Same code, different reception.

### STORY 8.11 — Documentation reconciliation pass (🟢 Risk 12 mitigation)
**Why this story:** 6 docs (README, methodology.md, data_card.md, model_card.md, integration_contract.md, architecture.md) over 6 weeks of build. Code evolved; docs drifted. Final reconciliation catches drift before shipping.

- [ ] 8.11.1 — Walk through `README.md` claim-by-claim — verify each against actual code/output
- [ ] 8.11.2 — Walk through `docs/methodology.md` — verify every "we chose X because Y" against actual implementation. Particularly check: SMOTE absence, AUC-PR usage, calibration parity choice, threshold-as-experiment framing, MLflow setup, DL non-choice + citations
- [ ] 8.11.3 — Walk through `docs/data_card.md` — verify dataset stats match what's actually in `data/processed/`
- [ ] 8.11.4 — Walk through `docs/model_card.md` — verify reported metrics match `reports/figures/` and MLflow runs
- [ ] 8.11.5 — Walk through `docs/integration_contract.md` — verify schema matches current `marts.v_attrition_features` (re-run Story 1.1.8 contract test)
- [ ] 8.11.6 — Walk through `docs/architecture.md` — verify the data flow diagram matches actual file paths in `src/`
- [ ] 8.11.7 — Fix every drift instance found. Re-commit. Re-run CI.
- [ ] 8.11.8 — Add ongoing rule (carries forward to future projects): at every `/kill` (epic close), update affected doc sections. "Doc updated?" item in each future epic's Definition of Done.

**Epic 8 Checkpoint:** Shipped. Visible. Distributed. **All docs reconciled against code.** **External review applied (Story 8.10.6.5).** **GitHub repo metadata set (Story 8.10.7).** **Pre-publish security scan clean (Story 8.10.0).**

**Epic 8 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — this is the final epic close, so the "energy switch" decision becomes the post-rp pivot decision (next portfolio piece, sustained applications, rest).

---

## Mitigations landed in Epic 8

| Risk | Mitigation | Where |
|---|---|---|
| 🟢 12 — Documentation drift | Final reconciliation pass — walk every doc against code | Story 8.11 |
