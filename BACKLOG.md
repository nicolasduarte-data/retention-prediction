# retention-prediction — Project Backlog (hrds/)

Production-grade retention prediction model for senior People Analytics + Data Scientist roles. Reads from **pa-warehouse** (`marts.v_attrition_features` in BigQuery), ships as a separate public GitHub repo, integrates predictions back into the pa-warehouse Looker Studio dashboard (Page 5).

Target buyer: Visier, Lattice, Workday, Personio, HiBob, Deel, tech-company People Analytics teams.

**The thesis:** Most public HR retention portfolios are IBM tutorial notebooks with single-model + accuracy + no fairness + no calibration. The campaign synthesis (`research/sessions/rp-prey-001-portfolio-craft-plan.md` lines 472–494) identified the specific gap structure. This project **ships the canonical HR exemplar the libraries themselves don't ship**.

**Teaching mode:** This is a hands-on build. Every story includes a "Why this story" rationale + "What you'll learn" note so the technique sticks, not just the artifact.

---

## 📂 File Structure (restructured 2026-05-20)

This is the **project-level** BACKLOG — definitions, decisions, research, and the per-epic index. **Detailed story-level content lives in `backlog/epic-N-*.md` files.** Always load this file first, then load the specific epic file when working that epic.

```
projects/hrds/retention-prediction/
├── BACKLOG.md                          ← you are here (project-level only)
├── CLAUDE.md                           ← project conventions
└── backlog/
    ├── epic-0-project-setup.md         ← Epic 0 detail (rp-prey-001)
    ├── epic-1-data-layer.md            ← Epic 1 detail (rp-prey-002)
    ├── epic-2-baseline-models.md       ← Epic 2 detail (rp-prey-003)
    ├── epic-3-evaluation-rigor.md      ← Epic 3 detail (rp-prey-004)
    ├── epic-4-survival-analysis.md     ← Epic 4 detail (rp-prey-005)
    ├── epic-5-fairness-audit.md        ← Epic 5 detail (rp-prey-006)
    ├── epic-6-interpretability.md      ← Epic 6 detail (rp-prey-007)
    ├── epic-7-causal-slice.md          ← Epic 7 detail (rp-prey-008, optional)
    └── epic-8-polish-publish.md        ← Epic 8 detail (rp-prey-009)
```

**Why split:** the monolithic BACKLOG hit 35k tokens — exceeded read limits, made Edits brittle, slowed context windows. Per-epic files solve this. See `## File Structure Decision` at the bottom of this file.

---

## 🔄 Loops Overview — Shipping Layer (added 2026-05-21 r03 hunt response)

**Restructured from "9 epics" to "4 main loops (6 prey ships total)"** following the pa-warehouse shipping pattern that demonstrably worked for this workspace. Each loop ends with a working public artifact + a LinkedIn moment + an independently shippable portfolio piece. Per-epic files (`backlog/epic-N-*.md`) become the detailed implementation reference per loop, not separate units of work.

**Why this restructure:** r03 hunt review found the 9-epic plan was over-engineered — 25-34 day estimate was actually 50-80 days realistic, with high abandonment risk past Loop 3. Pa-warehouse shipped in 4 loops in ~3-4 weeks (the pattern works). Each loop is small enough to finish and ship before fatigue catches up.

### Loop ship plan

| Loop | Prey | Effort | Public artifact at end |
|---|---|---|---|
| **Loop 1 — Walking Skeleton** | rp-prey-001 (✅ Ready) | **3-5 days** | Public repo + CI green + first end-to-end pipeline (load → train one GBM → report AUC-PR) + README skeleton at v0.1 |
| **Loop 2 — Full Comparison + Eval** | rp-prey-002 | ~1 week | LR/GBM/EBM × 2 cohorts comparison + MLflow UI + evaluation notebook (calibration + threshold + EV + nested CV) |
| **Loop 3a — Fairness** *(highest-leverage differentiator — ship first within Loop 3)* | rp-prey-003 | ~1 week | Fairness notebook + audit matrix + Fairlearn mitigation comparison + intervention-category audit (R2-recipe complete) |
| **Loop 3b — Survival** | rp-prey-004 | ~1 week | Survival notebook + SurvSHAP(t) chart + IBS report (G3 zero-repo differentiator) |
| **Loop 3c — Interpretability** | rp-prey-005 | ~1 week | Interpretability notebook + SHAP + ALE + DiCE + EBM intrinsic |
| **Loop 4 — Polish + Publish** | rp-prey-006 | 1-2 weeks | v1.0 launch — README polished + Substack post + LinkedIn video + dashboard integration + Looker Studio Page 5 |

**Total realistic effort:** 6-8 weeks. **First ship: 3-5 days.**

### Causal slice (Epic 7) — CUT from v1.0 (r03 hunt finding)

Per r03 hunt review (scope discipline finding) + Epic 7's "optional" framing from initial design: **causal slice is CUT from v1.0**. It moves to the v1.1 roadmap, added only if a reviewer of v1.0 specifically asks about causal reasoning. This frees Loop 4 to focus on polish + publish without the DoWhy + EconML complexity. `backlog/epic-7-causal-slice.md` remains in the repo as a future-work spec.

### Loop-to-epic mapping (detail reference)

Each loop draws from one or more per-epic files in `backlog/`. The per-epic files are the **implementation reference**, not separate ship units:

| Loop | Detailed reference (per-epic files) |
|---|---|
| **Loop 1** | `epic-0` (full) + `epic-1` (Stories 1.1 + 1.2 + 1.4 + 1.7 minimal) + `epic-2` (Stories 2.1 + 2.3 only, hybrid cohort) + `epic-3` (Story 3.1 only) + `epic-8` (Story 8.1 README skeleton only) |
| **Loop 2** | `epic-1` remaining (Stories 1.3, 1.5 full 4-layer, 1.6) + `epic-2` full (LR + EBM + comparison + MLflow + EBM OHE sensitivity + repro smoke) + `epic-3` full (calibration + threshold + EV + nested CV + mutation testing) + JD re-validation (Story 2.10) |
| **Loop 3a** | `epic-5` full (fairness audit) |
| **Loop 3b** | `epic-4` full (survival + SurvSHAP(t)) |
| **Loop 3c** | `epic-6` full (interpretability cross-validated) |
| **Loop 4** | `epic-8` remaining (everything except Story 8.1 covered in Loop 1) |
| **v1.1 roadmap** | `epic-7-causal-slice.md` — held for v1.1 if reviewer feedback requests it |

### Early-exit per loop (ship & stop discipline)

Each loop boundary is a legitimate "ship + stop + measure feedback" point:

| Stopped after | Version | Signal level |
|---|---|---|
| Loop 1 | v0.1 | Skeleton + tracer-bullet pipeline. Junior signal but real. Sets up the rest. |
| Loop 2 | v0.5 | 3-model comparison + evaluation rigor. Mid-senior signal. **First strong ship — competent ML practitioner.** |
| Loop 3a | v0.6 | Adds R2-recipe fairness audit. Strong senior ship. |
| Loop 3b | v0.7 | Adds survival arm + SurvSHAP(t). Senior signal complete. |
| Loop 3c | v0.8 | Adds interpretability deep-dive. Senior signal layered. |
| Loop 4 | v1.0 | Polished + published + distributed. Bar Test target. |

**Decision at each `/kill`:** ship the current version as-is (release tag + LinkedIn) and measure feedback OR continue immediately to next loop. Default = **ship-and-measure unless next loop's gap is THE thing a real reviewer asked for**. Apply the Epic Close Protocol's energy-switch decision (`## 🛑 Epic Close Protocol` below).

---

## Definition of Done (Project-Level)

A senior People Analytics IC at Visier/Lattice/Workday reviews this repo and says "this person knows what they're doing." Concretely:

- [ ] Public GitHub repo with proper `src/` layout (not notebook-only)
- [ ] `Makefile` or `justfile` with reproducibility commands: `make data`, `make train`, `make evaluate`, `make fairness`, `make report`, `make mlflow-ui`, `make repro`
- [ ] Pre-commit hooks: ruff + black + mypy + nbstripout + notebook-logic-check
- [ ] CI: GitHub Actions running tests + linting on push
- [ ] Pinned dependencies via `uv.lock` or `poetry.lock` (not just `requirements.txt`)
- [ ] **Dual-cohort experiment:** HRIS-only vs hybrid (HRIS+survey) models, apples-to-apples, honest reporting of which won
- [ ] **Three models compared:** logistic regression (interpretable baseline) → gradient boosting (XGBoost or LightGBM) → EBM (InterpretML). Comparison table with AUC-PR, calibration, interpretability score.
- [ ] **Survival arm:** Cox PH baseline + Random Survival Forest with C-index + Integrated Brier Score + calibration at 3/6/12-month horizons
- [ ] **SurvSHAP(t)** time-varying feature importance chart (G3 confirmed: zero HR repos publish this) — OR documented fallback per Risk 1 mitigation
- [ ] **Evaluation rigor:** AUC-PR + precision@k + calibration plots + Expected Value framing with sensitivity sweep + breakeven plot + nested CV
- [ ] **Threshold calibration over SMOTE** (R1 [FLIP-RISK] verdict — document the choice) — framed as controlled experiment / A/B testing equivalent
- [ ] **MLflow experiment tracking** with screenshot artifact (Risk 11 mitigation: fresh-clone reproduction docs included)
- [ ] **Fairness audit:** Reweighing on compound attribute (gender × age × tenure-band) + 3-metric audit (demographic parity + equalized odds + **calibration parity** as primary) + Chouldechova impossibility theorem paragraph + top-k 4/5 rule audit + intervention-category demographic audit + Fairlearn ExponentiatedGradient in-processing mitigation — with minimum-group-size rule enforced
- [ ] **Interpretability section:** SHAP beeswarm with **Rung 1 caption discipline** + SHAP waterfall (3 cases) + ALE plots (top-3 correlated features) + DiCE counterfactuals (2 employees, actionability constraint) + per-group calibration + adversarial SHAP caveat in methodology doc
- [ ] **Optional causal slice:** SHAP + DoWhy backdoor ATE + EconML Double ML on one scoped treatment, separate notebook — OR cleanly skipped per Risk 14
- [ ] **Data Card** at `docs/data_card.md` (universal gap per G3)
- [ ] **Model Card** at `docs/model_card.md` (universal gap per G3)
- [ ] **Integration contract** at `docs/integration_contract.md` (Risk 2 mitigation — schema validated in CI)
- [ ] **README** with: problem framing → governance disclaimer ("decision support, not decision making") → forbidden uses (EEOC + AEDT + EU AI Act Annex III) → methodology (incl. DL non-choice paragraph with citations + hypothesis reframe + threshold-as-experiment framing) → results → fairness → interpretability → limitations
- [ ] **Predictions written back to BigQuery** (`marts.v_attrition_predictions`) and surfaced in pa-warehouse Looker Studio dashboard as Page 5
- [ ] **Distribution:** Substack post draft + LinkedIn demo video script (G1's "10x effort" markers)
- [ ] **Documentation reconciliation pass (Story 8.11)** — every doc claim verified against code
- [ ] **Bar test:** Would this notebook survive code review at Visier/Lattice/Workday PA team? Self-answer yes with specific receipts.

---

## Research Foundation

This BACKLOG is the implementation arm of the rp-prey-001-portfolio-craft campaign. All seven sessions complete 2026-05-19. Build spec lives in the campaign summary.

- **Campaign plan + summary:** [`research/sessions/rp-prey-001-portfolio-craft-plan.md`](../../../research/sessions/rp-prey-001-portfolio-craft-plan.md) — read lines 472–494 (the build spec) before every story
- **R1 — HR Attrition Modeling state of the practice:** [`research/sessions/2026-05-12_employee-attrition-prediction-state-of-the-art/`](../../../research/sessions/2026-05-12_employee-attrition-prediction-state-of-the-art/)
- **R2 — Algorithmic Fairness in HR:** *(session folder name TBD — check `research/sessions/2026-05-1X_*`)*
- **R3 — Causal vs Predictive Framing:** *(session folder name TBD)*
- **R4 — Model Interpretability Theory:** *(session folder name TBD)*
- **G1 — Reference Portfolio Survey:** [`research/sessions/2026-05-12_rp-portfolio-survey/output.md`](../../../research/sessions/2026-05-12_rp-portfolio-survey/output.md)
- **G2 — HR Attrition + Synthetic HR Data Reference Implementations:** *(session folder name TBD)*
- **G3 — Fairness + Interpretability Tooling in HR Practice:** [`research/sessions/2026-05-19_rp-fairness-xai-tooling-survey/output.md`](../../../research/sessions/2026-05-19_rp-fairness-xai-tooling-survey/output.md)
- **Gap analysis (pa-warehouse implementation):** [`projects/hrds/pa-warehouse/docs/campaign-summary-build-gap.md`](../pa-warehouse/docs/campaign-summary-build-gap.md)

---

## Preconditions (pa-warehouse Dependencies)

This project **cannot start coding past Epic 0 until pa-warehouse delivers** the following:

| Precondition | pa-warehouse Action | Status | Owner |
|---|---|---|---|
| `marts.v_attrition_features` mart | Story 4.7 of paw-prey-004 | ✅ Delivered 2026-05-19 | (done) |
| Age column (`age_at_window_close`) in mart | Action 1 from gap analysis | ✅ Delivered 2026-05-22 (paw-prey-005) — verified live via schema dump 2026-05-27 | (done) |
| Column-level mutability metadata in `_core.yml` | Action 3 from gap analysis | ✅ Delivered 2026-05-22 (paw-prey-005) | (done) |
| BigQuery service account credentials | Reuse from pa-warehouse | ✅ Available | (done) |
| `marts.v_attrition_predictions` write permissions | Loop 6 prep — service account `BigQuery Data Editor` on `marts` dataset | ⏳ Pending → ship during rp Epic 8 (Story 8.5) | rp Epic 8 |
| **Survey signal columns in `marts.v_attrition_features`** (engagement score, eNPS, manager-relationship score, etc.) | **NEW — gap surfaced 2026-05-27 by schema discovery spike (Story 0.5.0). Mart currently has 10 columns, zero survey signal.** Required for the HRIS-only-vs-hybrid controlled experiment per `## Hypothesis Update` below | ⏳ Pending → **paw-prey-NNN to be created before `/mark rp-prey-002` (Loop 2)** | paw-prey-NNN |

**Work that can start before preconditions land:** Epic 0 (repo init + dependencies + project structure) — covered by `backlog/epic-0-project-setup.md`. Loop 1 (rp-prey-001 walking skeleton) trains XGBoost on the 7 effective features now in the mart and runs end-to-end. Loop 2 (rp-prey-002 full comparison) **cannot** run the controlled HRIS-only-vs-hybrid experiment until the survey signal columns land via the new paw-prey above.

**Critical-path blocker for Loop 2:** new paw-prey to extend `marts.v_attrition_features` with survey signal columns ships → unblocks the hybrid arm of rp-prey-002 → unblocks the central hypothesis (HRIS-only vs hybrid). Without this, Loop 2 either degrades to "with-demo vs without-demo" reframe OR Loop 2 stalls. Decision 2026-05-27 (Nico): **Option 1 — ship the paw-prey, keep the original portfolio thesis intact.**

**Prey numbering convention (revised 2026-05-20 — Option B):** The original seed prey `work/queue/rp-prey-001-talent-retention-hard-data.md` was **overwritten with Epic 0 scope** to keep numbering consistent with paw-prey / wa-prey / audit-prey (prey-NNN = work unit NNN, not seed). Epic 0 → rp-prey-001 (Ready), Epic 1 → rp-prey-002, Epic 2 → rp-prey-003, etc. Original ADEN-inspired hypothesis preserved in `## Hypothesis Update` below; the seed prey file itself no longer holds that content.

---

## Hypothesis Update (Important — Read Before Writing Code)

The seed prey's hypothesis: *"A retention model on hard operational signals outperforms a survey-based model both in prediction accuracy and in actionability."*

**R1 finding:** Not supported. Rubenstein 2017 meta-analysis found 17 validated attrition predictors; only 2 live in pure HRIS data. Engagement/survey features are empirically necessary at the precision tier. HRIS-only models hit a precision ceiling.

**New hypothesis (the controlled experiment):**

> "We test whether HRIS-only behavioral features alone can match hybrid HRIS+survey models on precision-at-k, AUC-PR, and calibration. The literature says no — but we test it honestly. The result, whichever way it lands, is the analytical contribution."

This reframe is **stronger as portfolio narrative** — it shows a practitioner who can be wrong and update on evidence. Senior hiring managers grep for exactly this.

---

## Stack Decisions (Resolved)

| Decision | Choice | Rationale |
|---|---|---|
| **Language + version** | Python 3.11 | Matches pa-warehouse and workforce-analytics. Modern enough for `lifelines`/`scikit-survival` deps without bleeding-edge fragility. |
| **Dependency manager** | `uv` (preferred) or `poetry` | Lock file is mandatory for reproducibility. `uv` is faster + native to modern Python tooling; `poetry` is more conservative. Pick one and stick. |
| **Classification stack** | scikit-learn (LR) + XGBoost (GBM) + InterpretML (EBM) | R1: GBM density 4.0 winner over RF. EBM closes a G3 universal gap (12/13 repos don't test it). LR as interpretable baseline. |
| **No deep learning (Risk 1 documented)** | Tree-based methods only on tabular data | Grinsztajn 2022 / Shwartz-Ziv 2022 / Borisov 2022 — settled science: GBM beats DL on tabular at our sample size. README documents the non-choice with citations. |
| **Survival stack** | `scikit-survival` (RSF + Cox PH) + `survshap` (for SurvSHAP(t), with documented fallback per Story 0.1.10) | R3 settled stack. Only `Naresh1401` repo reports IBS today — easy bar to clear. SurvSHAP(t) is G3's highest-leverage single gap (zero HR repos use it). |
| **Class imbalance treatment** | **Threshold calibration, NOT SMOTE** | R1 [FLIP-RISK] verdict. SMOTE distorts calibration. Use class weights + threshold sweep + cost-sensitive evaluation. Document this choice prominently. Framed as controlled experiment (Story 3.3). |
| **Evaluation primary metrics** | AUC-PR + precision@k + Brier + reliability diagrams + EV with sensitivity sweep | R1: AUC-only is misleading at HR base rates (8–15% attrition). G2 found zero public repos do this. |
| **Cross-validation** | Nested CV (5×5 outer-inner) — `hannesbuchner` pattern | G2 finding: most public repos use single train/test split and overstate generalization. Nested CV is the credible signal. |
| **Fairness library** | Fairlearn (primary) + `aif360` only if needed | G3: Fairlearn is the modern choice. AIF360 still works but its API is heavy. Use Fairlearn's `MetricFrame` for the audit, `ExponentiatedGradient` for in-processing. |
| **Fairness audit pattern** | 3-metric (demographic parity + equalized odds + **calibration parity** primary) + Chouldechova impossibility theorem paragraph + top-k 4/5 audit + intervention-category demographic audit | R2 minimum-viable senior stack. The intervention-category audit is an explicit R2 open gap that G3 confirmed unclaimed in public repos. **Minimum-group-size rule enforced per Story 5.1.5.** |
| **Interpretability stack** | SHAP (TreeExplainer + LinearExplainer per `galafis` variant-correctness) + `dice-ml` (counterfactuals) + ALE library (alibi OR dalex per Story 0.1.11 verification) + InterpretML (EBM intrinsic) + `survshap` (SurvSHAP(t) with fallback) | G3 found 12/13 repos miss ALE; 13/13 miss DiCE; 13/13 miss SurvSHAP(t). Each is a confirmed differentiation lever. |
| **Causal stack (optional)** | DoWhy (backdoor ATE) + EconML (Double ML refutation) | R3 + G3: HR-context causal usage exists at portfolio-demo level (`ayu5h4`, `olivia3395`). The SHAP+DoWhy *with explicit Rung 1/Rung 2 boundary* is unclaimed. **Epic 7 is clean-skippable per Risk 14.** |
| **Data source** | BigQuery via `pandas-gbq` for development; CSV snapshot for reproducibility | Reads `marts.v_attrition_features` from pa-warehouse with `ORDER BY employee_id` for determinism. CSV export from BigQuery committed to repo under `data/processed/` for offline reproducibility (gitignored if >50MB). |
| **Notebooks vs scripts** | `src/` modules for everything + thin notebooks for narrative | G1 anti-pattern: single-notebook repos. Modules are testable, notebooks are narrative. The notebooks `import` from `src/`. **Notebook-logic-lint hook enforces (Story 0.3.7) — Risk 10 mitigation.** |
| **Testing framework** | `pytest` + `pytest-cov` (target 70%+ coverage on `src/`) + integration tests + data quality tests + reproducibility smoke test | Production-grade signal. Most public HR repos have zero tests. **Risks 6 + 7 mitigated across Stories 2.6 + 2.8.** |
| **Experiment tracking** | **MLflow (local file store, no hosted server)** | Some Senior PA Scientist / Data Scientist JDs (Visier, Lattice tier) explicitly ask for it. ~1h add across Epic 2 — wrap 6 training cells in `mlflow.start_run()`, screenshot the UI for the README. Closes an ATS keyword without infra overhead. Risk 11 mitigated via Story 2.7.9 (fresh-clone reproduction docs). |
| **Linting + formatting** | `ruff` (lint + format) + `mypy --strict` on `src/` + `nbstripout` on notebook commits + custom notebook-logic-check | Pre-commit hooks enforce. |
| **CI** | GitHub Actions: lint + test + (optional) lightweight smoke run of training pipeline on tiny synthetic subset + reproducibility test + schema contract test | Catches regressions before they ship. |
| **Docs framework** | Plain markdown in `docs/` + dbt-style data card + Google's Model Card format + final reconciliation pass (Story 8.11) | Resist the temptation to add MkDocs/Sphinx — the audience reads markdown, not generated docs sites. **Risk 12 mitigation: every doc claim verified against code at Epic 8.** |
| **Charts** | matplotlib for static portfolio figures + Plotly for any interactive demo + Looker Studio for the integrated dashboard | matplotlib's lack of interactivity is a feature for portfolio repos — figures are *for the reader*, not exploratory. |
| **Random seed convention** | `SEED = 42` global constant, set in every notebook + script via `set_global_seed()`. Document in README. | Reproducibility is a senior signal. Same convention as workforce-analytics + pa-warehouse. **Risk 6 mitigation: full smoke test at Story 2.8.** |

---

## Research Findings Applied (Campaign Summary Synthesis)

The campaign summary's recommendations, translated into this project's design:

**From G1 (Portfolio Survey):**
- *"Anti-pattern confirmed: IBM dataset + accuracy metric + single notebook + no distribution = invisible to hiring managers at Visier, Lattice, Workday."* → This project uses pa-warehouse's custom benchmark-aligned synthetic data (not IBM), evaluates with AUC-PR + precision@k + calibration (not accuracy), uses `src/` + thin notebooks (not single notebook), ships paired Substack post + LinkedIn demo video.
- *"10x effort markers confirmed: calibration, precision@k, HRIS-only vs hybrid controlled experiment, survival analysis variant, intervention cost framing, and a blog post are the differentiators."* → Every one of those is a story in the per-epic files.

**From R1 (Attrition Modeling):**
- Gradient boosting is the default (density 4.0 vs 3.33 RF). LR as interpretable baseline. EBM as the 2024+ challenger.
- Threshold calibration over SMOTE for class imbalance.
- AUC-PR + precision@k over AUC-ROC for HR base rates.
- Engagement/survey features are empirically necessary — hence the dual-cohort honest test.

**From R2 (Fairness):**
- Compound attribute (gender × age × tenure-band) — requires pa-warehouse Action 1 (age column) → paw-prey-005.
- Calibration parity as the primary fairness metric.
- Chouldechova impossibility theorem paragraph stating the choice and why.
- Intervention-category demographic audit — R2's explicit open gap.
- Fairlearn `ExponentiatedGradient` in-processing (`ctriz` pattern, the only public repo using it).
- Reweighing on compound attribute as pre-processing baseline.

**From R3 (Causal vs Predictive):**
- Rung 1 narration discipline: every SHAP caption says "the model weighted X heavily — associational, not causal."
- Survival analysis as more honest framing than binary classification for retention specifically.
- Optional causal slice with explicit Rung 2 scoping to one treatment.

**From R4 (Interpretability Theory):**
- SHAP TreeExplainer for tree models, LinearExplainer for LR — variant-correctness matters.
- ALE plots required (not PDP) because HR features share variance.
- Adversarial SHAP caveat (Slack 2020) in methodology doc.
- EBM as the interpretable-by-design challenger before defaulting to post-hoc explanation.

**From G2 (Reference Implementations):**
- Repo layout copies `aroraneel` skeleton + `galafis` `src/explainability/` + `src/fairness/` modules.
- `Naresh1401` Fairlearn 3-metric pattern is the floor; we go beyond it.
- IBS reporting is the floor differentiator (only `Naresh1401` does it today in public repos).

**From G3 (Fairness + Interpretability Tooling):**
- Universal gaps to fill: SurvSHAP(t), DiCE counterfactuals, ALE plots, EBM-vs-GBM comparison, per-group calibration parity, Chouldechova paragraph, intervention-category demographic audit, Data Card / Model Card, adversarial SHAP caveat. **Each is a story in the per-epic files.**
- `Jott2121` governance language for README disclaimers.

---

## Repo Structure (Planned)

```
retention-prediction/
├── .github/
│   └── workflows/
│       ├── lint.yml              # ruff + mypy + nbstripout check
│       └── test.yml              # pytest + smoke train + repro test + schema contract test
├── .pre-commit-config.yaml       # ruff + black + mypy + nbstripout + notebook-logic-check
├── .gitignore                    # excludes credentials, large CSVs, .ipynb_checkpoints, mlruns/
├── Makefile                      # make data / train / evaluate / fairness / report / mlflow-ui / repro
├── pyproject.toml                # project metadata + tool config (ruff/mypy)
├── uv.lock                       # dependency lock (or poetry.lock)
├── README.md                     # the load-bearing artifact for hiring managers
├── LICENSE                       # MIT
├── scripts/
│   └── check_notebook_logic.py   # pre-commit hook blocking def/class in notebooks (Risk 10)
│
├── src/
│   └── retention/                # importable package
│       ├── __init__.py
│       ├── config.py             # SEED, paths, hyperparameters, set_global_seed()
│       ├── data/
│       │   ├── load.py           # BigQuery → pandas (pandas-gbq) + CSV fallback
│       │   ├── split.py          # train/val/test with temporal awareness
│       │   └── schema.py         # pydantic / dataclass schema for v_attrition_features
│       ├── features/
│       │   ├── catalog.py        # feature catalog with mutability metadata
│       │   ├── transforms.py     # encoders, scalers, survey decay
│       │   └── cohorts.py        # HRIS-only vs hybrid filter logic
│       ├── models/
│       │   ├── lr.py             # logistic regression baseline
│       │   ├── gbm.py            # XGBoost wrapper
│       │   ├── ebm.py            # InterpretML EBM wrapper
│       │   ├── survival.py       # Cox PH + RSF
│       │   ├── threshold.py      # threshold calibration utilities (NOT SMOTE)
│       │   └── tracking.py       # MLflow log_run() wrapper
│       ├── evaluation/
│       │   ├── metrics.py        # AUC-PR, precision@k, Brier, lift, survival metrics
│       │   ├── calibration.py    # reliability diagrams, calibration error
│       │   ├── expected_value.py # EV framing + sensitivity sweep + breakeven
│       │   ├── nested_cv.py      # nested cross-validation harness
│       │   └── repro.py          # reproducibility smoke test harness (Story 2.8)
│       ├── fairness/
│       │   ├── audit.py          # 3-metric audit using Fairlearn MetricFrame
│       │   ├── compound.py       # gender × age × tenure-band attribute (+ min-group-size rule)
│       │   ├── mitigation.py     # Reweighing + ExponentiatedGradient
│       │   ├── intervention.py   # intervention-category demographic audit
│       │   └── proxy.py          # department-as-proxy detector
│       ├── explainability/
│       │   ├── shap_explain.py   # TreeExplainer + LinearExplainer + Rung 1 captions
│       │   ├── ale.py            # ALE plots via alibi OR dalex (Story 0.1.11 outcome)
│       │   ├── counterfactuals.py # DiCE with actionability constraint
│       │   ├── survshap.py       # SurvSHAP(t) time-varying chart (with fallback per Story 4.5.1)
│       │   └── ebm_intrinsic.py  # EBM intrinsic interpretation
│       ├── causal/               # OPTIONAL — only create if Epic 7 runs (Risk 14)
│       │   ├── dowhy_ate.py      # backdoor ATE
│       │   └── econml_dml.py     # Double ML refutation
│       └── writeback/
│           └── bigquery.py       # write predictions to marts.v_attrition_predictions
│
├── notebooks/                    # narrative-driven analysis, imports from src/
│   ├── 00_environment_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines_lr_gbm_ebm.ipynb
│   ├── 03_evaluation_rigor.ipynb
│   ├── 04_survival_arm.ipynb
│   ├── 05_fairness_audit.ipynb
│   ├── 06_interpretability.ipynb
│   ├── 07_causal_slice.ipynb     # optional — only if Epic 7 runs
│   └── 08_final_report.ipynb     # the load-bearing notebook for portfolio reviewers
│
├── tests/
│   ├── conftest.py               # pytest fixtures (small synthetic data)
│   ├── test_smoke.py
│   ├── test_data_load.py
│   ├── test_integration_contract.py  # schema validation (Risk 2)
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_integration.py       # full pipeline smoke (Risk 7)
│   ├── test_data_quality.py      # prediction sanity + no-leakage-at-predict (Risk 7)
│   ├── test_no_leakage.py        # leakage MUST-PASS gate (Risk 3)
│   ├── test_reproducibility.py   # full smoke test (Risk 6)
│   ├── test_evaluation.py
│   ├── test_fairness.py
│   └── test_explainability.py
│
├── docs/
│   ├── data_card.md              # synthetic generation methodology, benchmarks, intended use, forbidden use
│   ├── model_card.md             # Google Model Card format
│   ├── methodology.md            # threshold-over-SMOTE, AUC-PR-over-AUC-ROC, adversarial SHAP caveat, DL non-choice, etc.
│   ├── fairness_recipe.md        # how the 3-metric audit + Chouldechova paragraph were built
│   ├── architecture.md           # data flow diagram pa-warehouse → rp → predictions
│   └── integration_contract.md   # cross-project schema contract with pa-warehouse (Risk 2)
│
├── reports/
│   ├── figures/                  # all generated plots, committed for README inclusion
│   │   ├── calibration_overall.png
│   │   ├── calibration_per_group.png
│   │   ├── shap_beeswarm.png
│   │   ├── shap_waterfall_case_1.png
│   │   ├── ale_top_features.png
│   │   ├── survshap_time_varying.png
│   │   ├── dice_counterfactuals.png
│   │   ├── precision_at_k.png
│   │   ├── ev_sensitivity.png
│   │   ├── threshold_sweep.png
│   │   ├── fairness_audit_matrix.png
│   │   └── mlflow_experiment_view.png
│   └── distribution/
│       ├── substack_post_draft.md
│       └── linkedin_video_script.md
│
└── data/
    ├── processed/                # CSV exports from BigQuery for offline reproducibility
    │   └── .gitkeep              # CSVs ignored if >50MB, otherwise committed
    └── README.md                 # how to refresh from pa-warehouse
```

---

## Detailed Epic Reference (per-loop content map)

**Note (2026-05-21 r03 hunt restructure):** the table below is the **detail reference**, not the shipping plan. Actual ship units are the **Loops** (see Loops Overview above) — each loop draws from multiple epic files. Prey numbering follows loops, not epics. This table preserves the per-epic structure as implementation documentation.

| Loop | Epic file | Loop scope from this file | Public artifact contribution |
|---|---|---|---|
| Loop 1 | [`backlog/epic-0-project-setup.md`](backlog/epic-0-project-setup.md) | All stories (0.1–0.6) | Project skeleton + CI + integration contract |
| Loop 1 | [`backlog/epic-1-data-layer.md`](backlog/epic-1-data-layer.md) | Stories 1.1, 1.2 (full FeatureSpec incl. leakage_audited), 1.4, 1.7 minimal | Data loader + catalog + temporal split + exploration notebook |
| Loop 1 | [`backlog/epic-2-baseline-models.md`](backlog/epic-2-baseline-models.md) | Stories 2.1, 2.3 only (hybrid cohort) | Preprocessing + one trained GBM |
| Loop 1 | [`backlog/epic-3-evaluation-rigor.md`](backlog/epic-3-evaluation-rigor.md) | Story 3.1 (AUC-PR only) | First metric report |
| Loop 1 | [`backlog/epic-8-polish-publish.md`](backlog/epic-8-polish-publish.md) | Story 8.1 (skeleton only, v0.1 banner) | README at v0.1 + roadmap |
| Loop 2 | `epic-1` remaining | Stories 1.3 (cohorts), 1.5 (full 4-layer leakage), 1.6 (Data Card) | Dual-cohort + leakage MUST-PASS gate + Data Card |
| Loop 2 | `epic-2` remaining | Stories 2.2 (LR), 2.4 (EBM), 2.5 (comparison), 2.6 (tests), 2.7 (MLflow), 2.8 (repro smoke), 2.9 (EBM OHE sensitivity), 2.10 (JD re-validation) | 3-model × 2-cohort comparison + MLflow + repro |
| Loop 2 | `epic-3` remaining | Stories 3.2 (calibration), 3.3 (threshold), 3.4 (EV), 3.5 (nested CV), 3.6 (eval nb), 3.7 (mutation testing) | Evaluation rigor notebook |
| Loop 3a | [`backlog/epic-5-fairness-audit.md`](backlog/epic-5-fairness-audit.md) | All stories (5.1–5.11) | Fairness notebook + audit matrix + Fairlearn mitigation comparison |
| Loop 3b | [`backlog/epic-4-survival-analysis.md`](backlog/epic-4-survival-analysis.md) | All stories (4.1–4.6) | Survival notebook + SurvSHAP(t) chart + IBS report |
| Loop 3c | [`backlog/epic-6-interpretability.md`](backlog/epic-6-interpretability.md) | All stories (6.1–6.8) | Interpretability notebook (SHAP + ALE + DiCE + EBM intrinsic) |
| Loop 4 | `epic-8` remaining | Stories 8.1 (full polish), 8.2 (Model Card), 8.3 (Data Card final), 8.4 (architecture diagram), 8.5 (writeback + SCD2 note), 8.6 (Looker Page 5), 8.7 (Substack), 8.8 (LinkedIn video), 8.9 (Bar Test), 8.10 (announce + security scan + external review + GitHub metadata), 8.11 (doc reconciliation) | v1.0 launch — full polished release |
| **v1.1 roadmap (CUT from v1.0)** | [`backlog/epic-7-causal-slice.md`](backlog/epic-7-causal-slice.md) | All stories (7.1–7.4) — **CUT** | Causal notebook — only built if v1.0 reviewers ask for it |

**Total realistic effort:** 6-8 weeks across 4 loops (6 prey ships). **First ship: 3-5 days (Loop 1).** Each loop ships independently — early-exit at any boundary still produces a coherent portfolio piece.

---

## 📢 LinkedIn Ship Cadence (revised 2026-05-21 — loop restructure)

**One LinkedIn post per loop ship (6 posts total over 6-8 weeks).** Spread out across the timeline so audience doesn't saturate. The pa-warehouse pattern: one LinkedIn moment per loop ship, paced ~1 per week. That cadence works.

| Loop | Post angle | Trigger |
|---|---|---|
| **Loop 1** — *"Walking skeleton live — repo + CI + first model"* | First ship moment. Sets expectations: production-grade scaffolding is the foundation. Quick scan-able artifact (the repo). | Loop 1 / rp-prey-001 closes |
| **Loop 2** — *"3-model × 2-cohort comparison + MLflow tracking + evaluation rigor"* | The first analytical contribution. Visible UI screenshot + comparison table. Mid-senior signal. | Loop 2 / rp-prey-002 closes |
| **Loop 3a** — *"Fairness audit + intervention-category demographic analysis (R2 recipe)"* | The R2 differentiating signal. Senior HR / People Analytics audience grep for this exactly. | Loop 3a / rp-prey-003 closes |
| **Loop 3b** — *"Survival arm + SurvSHAP(t) — zero public HR repos do this"* | The G3 highest-leverage differentiator. Senior PA Scientist tier signal. | Loop 3b / rp-prey-004 closes |
| **Loop 3c** — *"Interpretability cross-validated — SHAP + ALE + DiCE + EBM"* | Senior reviewer's "does this person understand interpretation" test. Pairs with Loop 3a fairness. | Loop 3c / rp-prey-005 closes |
| **Loop 4** — *"v1.0 launch"* (paired with Substack post + LinkedIn demo video) | Full release. Canonical artifact. Bar Test pass. | Loop 4 / rp-prey-006 closes |

**Pacing:** ~1 post per week across 6-8 weeks. Sustainable cadence — feeds the algorithm without saturating audience. Each post is a complete artifact (its own loop's public deliverable), not an "update" on a build still in progress.

**Why 6 posts now, not 3:** the loops pattern gives natural pacing. 3 posts crammed at Epic 2 / 5 / 8 didn't account for Loops 3a-3b-3c each being shippable. With each loop shipping ~1 week apart, posts naturally space out instead of clustering. The pacing is what makes 6 posts work where the original "9 epic posts" wouldn't.

---

## 🛑 Epic Close Protocol (r02 hunt fix 2026-05-21 — Tier 2.C centralized)

**Every epic's Definition of Done references this protocol. At `/kill` (epic close), explicitly evaluate:**

- [ ] **Result review:** Did this epic ship its public artifact? If not, why? Document gaps in the epic's progress.md.
- [ ] **Doc update?** Are the relevant docs (README, methodology.md, data_card.md, model_card.md, integration_contract.md, architecture.md) updated to reflect what shipped? If not, log doc-debt for Epic 8 Story 8.11.
- [ ] **Dependency check:** `uv pip list --outdated` — any breaking changes in the 3 fragile deps (`survshap`, `dice-ml`, `alibi`/`dalex`)? If yes, document in next epic's prep notes.
- [ ] **🟡 Energy switch decision (r02 fix — explicit prompt, not passive structure):** Evaluate explicitly — what's the next move?
  - [ ] Continue rp — the next epic is the right move RIGHT NOW
  - [ ] Switch to `fc-prey-001` (LLM Retention Engine, parallel weekend track) — rp got dry, need variety
  - [ ] Switch to `paw-prey-005` if not yet shipped — clear the next downstream blocker
  - [ ] Switch to `audit-prey-006` if Nestlé Sprint 4 just shipped — operational priority
  - [ ] Pause — apply to 2–3 new roles (see SESSION_STATE.md cadence trigger), rest, return next week
  - **Document the decision + rationale in progress.md before opening the next prey.** The structural protection works only when the decision is conscious.
- [ ] **Application cadence check:** Has the SESSION_STATE.md cadence trigger fired? If due, run it now before opening the next prey.

**Why this protocol exists:** Per-epic shipping IS the structural mitigation against ADHD long-execution abandonment. But structure is passive — Nico must consciously evaluate at each boundary, not default to "next epic." This protocol makes the evaluation explicit.

---

## 🎯 Early-Exit Thresholds (Risk 13 mitigation)

**Each epic boundary is a legitimate "ship-and-stop" point.** If energy/time runs out, you have a coherent portfolio piece at any of these checkpoints — no half-built abandonment.

| Stopped after | Portfolio shape | Buyer signal |
|---|---|---|
| Epic 0 | Public repo with skeleton + CI green | "I set up production-grade Python projects" — junior signal only |
| Epic 1 | + Data layer with dual-cohort + Data Card | "I think rigorously about data engineering for ML" — entry-mid signal |
| Epic 2 | + 3-model comparison (LR + GBM + EBM) + MLflow tracking | "I compare baselines apples-to-apples and track experiments" — solid mid signal |
| Epic 3 | + Evaluation rigor (AUC-PR, calibration, EV, nested CV) | **First strong ship.** "I evaluate models the way Visier seniors do." Mid-senior signal. |
| Epic 4 | + Survival arm + SurvSHAP(t) | "I understand retention as time-to-event, not just classification." Senior signal. |
| Epic 5 | + Fairness audit (Fairlearn + Chouldechova + intervention audit) | **Strong senior ship.** R2-recipe-complete. This is the Visier/Lattice tier signal. |
| Epic 6 | + Interpretability (SHAP + ALE + DiCE + EBM intrinsic) | Senior signal complete. R4 + G3 levers all pulled. |
| Epic 7 | + Causal slice (optional) | Top-tier signal. Combination is unclaimed in any public HR repo. |
| Epic 8 | + Polished README + Substack + LinkedIn video + dashboard integration | **Full deliverable.** Bar Test should pass. |

**Recommended ship-points if scope-cutting:**
- **Minimum viable senior portfolio piece:** stop after Epic 5. ~14–17 days. R1 + R2 + G1 + G2 all addressed.
- **Strong senior portfolio piece:** stop after Epic 6. ~18–22 days. Add interpretability.
- **Distinctive portfolio piece:** ship Epic 8 with optional Epic 7. ~25–34 days. Full Bar Test pass.

**ADHD pattern note:** Per-epic shipping cadence + early-exit framing is the structural protection against the "in-progress-forever" portfolio piece. Each epic → its own prey → its own /kill → its own LinkedIn post → momentum maintained.

---

## ⚠️ Risk Register

Production-grade safeguards across the build. Each risk has a documented mitigation in the per-epic files (column "Where").

### 🔴 Critical (address before Epic 1 starts)

| # | Risk | Likelihood | Impact | Mitigation | Where |
|---|---|---|---|---|---|
| 1 | `survshap` install fragility | Medium-High | Critical (loses SurvSHAP(t) differentiator) | Install gate + documented fallback path | `backlog/epic-0-project-setup.md` Story 0.1.10; `backlog/epic-4-survival-analysis.md` Story 4.5.1 |
| 2 | pa-warehouse contract drift — silent breakage | Medium | Critical (wasted modeling on wrong data) | Schema validation test in CI | `backlog/epic-1-data-layer.md` Story 1.1.8 |
| 3 | Temporal leakage — model false-positive | High | Critical (deploys to noise; credibility-killing) | MUST-PASS test gate blocking Epic 2 | `backlog/epic-1-data-layer.md` Story 1.5.5 |
| 4 | Epic 1 **AND Epic 5 AND Epic 6** blocked on pa-warehouse age column + mutability metadata + writeback IAM | Certain (already blocked) | Critical (modeling AND fairness audit AND DiCE counterfactuals AND Epic 8 writeback all blocked) | Explicit paw-prey-005 created — now covers age column, mutability metadata (consumed by Epic 6 DiCE Story 6.5), Snowflake port, AND writeback IAM verification (added/corrected 2026-05-21 per hunt review r01 + r02 fixes) | `work/queue/paw-prey-005-deliver-rp-preconditions.md` |

### 🟡 High (address by Epic 2)

| # | Risk | Likelihood | Impact | Mitigation | Where |
|---|---|---|---|---|---|
| 5 | `alibi` dependency weight (pulls TF transitive) | Medium | Moderate (slow CI, fresh-clone friction) | Install verification + dalex swap | `backlog/epic-0-project-setup.md` Story 0.1.11; `backlog/epic-6-interpretability.md` Story 6.4.2 |
| 6 | Reproducibility — SEED set but not propagated | Medium | Moderate (numbers different on reviewer machine) | Global seed scaffolding + full repro smoke test | `backlog/epic-0-project-setup.md` Story 0.2.9; `backlog/epic-2-baseline-models.md` Story 2.8 |
| 7 | Test coverage measures wrong thing | Certain | Moderate-High (broken model, green tests) | Integration + data quality + sanity tests | `backlog/epic-2-baseline-models.md` Stories 2.6.5–2.6.7 |
| 8 | BigQuery cost spiral during dev | Low/High (one bug) | Moderate (real money + lockout) | Dry-run cost estimation + CSV snapshot strategy | `backlog/epic-1-data-layer.md` Story 1.1.9 |

### 🟢 Medium (address by Epic 5–8)

| # | Risk | Likelihood | Impact | Mitigation | Where |
|---|---|---|---|---|---|
| 9 | Fairness audit instability on small groups | High | Moderate (unstable numbers = credibility hit) | Minimum-group-size rule + EG pre-test | `backlog/epic-5-fairness-audit.md` Stories 5.1.5 + 5.7.0 |
| 10 | Notebook logic drift from src/ | Medium-High | Moderate (logic untested) | Pre-commit hook blocking def/class in notebooks | `backlog/epic-0-project-setup.md` Story 0.3.7 |
| 11 | MLflow UI fails from fresh clone | Certain (by design) | Low-Moderate (reviewer confusion) | Fresh-clone reproduction docs | `backlog/epic-2-baseline-models.md` Story 2.7.9 |
| 12 | Documentation drift over the build | High | Moderate (claim-vs-code mismatch caught by reviewer) | Final reconciliation pass + per-epic doc-update rule | `backlog/epic-8-polish-publish.md` Story 8.11 |
| 15 | Fragile academic deps (`survshap`, `dice-ml`, `alibi`/`dalex`) deprecate mid-build | Medium | Low-Moderate (one breaking release stalls 1–2 stories) | Pre-epic dependency check: `uv pip list --outdated` at start of every epic execution; track upstream breaking changes for the 3 fragile packages specifically | Per-epic Definition of Done — add line: *"`uv pip list --outdated` run; no breaking changes since last epic"* |

### 🔵 Low (accept or defer)

| # | Risk | Mitigation strategy |
|---|---|---|
| 13 | Long execution → abandonment (25–34 days) | Per-epic shipping + Early-Exit Thresholds (above) make any stop-point a legitimate portfolio piece. Structurally mitigated by design. |
| 14 | Causal slice half-built (worse than skipping) | Explicit clean-skip rule at top of `backlog/epic-7-causal-slice.md`. Decide YES/NO before starting Story 7.1. |

---

## Bar Test (The Final Read)

The single test that matters: **Would this notebook survive code review at Visier/Lattice/Workday PA team?**

Self-answer with receipts. If yes:
- *Yes — because we ship calibration plots per group (G3 universal gap), SurvSHAP(t) for time-varying importance (G3 highest-leverage gap), DiCE with actionability constraint (G3 universal gap), ALE over PDP for correlated features (G3 universal gap), EBM as the intrinsic challenger (G3 universal gap), the Chouldechova paragraph (R2 minimum-viable senior stack), the intervention-category demographic audit (R2's explicit open gap), the dual-cohort controlled experiment (G2's "10x effort" marker), nested CV (G2 anti-pattern closer), threshold calibration over SMOTE framed as A/B testing (R1 [FLIP-RISK] verdict), MLflow experiment tracking with screenshot (modern MLOps signal), governance disclaimer with EEOC/AEDT/EU AI Act citations (R2 legal frame), paired blog + LinkedIn video (G1 distribution), and `src/` modular layout with tests + CI (G1 anti-pattern closer). Additionally: DL non-choice with literature citations (tool-fit judgment signal), R1-driven hypothesis reframe (intellectual honesty signal), reproducibility smoke test in CI (production-grade signal), schema contract test (cross-project robustness signal).*

If any of those isn't true at ship time, Epic 8 isn't done.

---

## File Structure Decision

**Restructured 2026-05-20** from monolithic BACKLOG (35,557 tokens, exceeded read limits) into project-level BACKLOG.md + per-epic `backlog/epic-N-*.md` files.

**Rationale:**
- Monolithic file was too large to read in full → forced grep/offset navigation → slowed audits, increased Edit fragility
- Per-epic files fit in single Read calls (~3–5k tokens each)
- Matches the "one prey per epic" shipping pattern — file-per-ship-unit
- Project-level decisions (stack, hypothesis, research) only need to be read once at session start
- Per-epic content loaded on demand when working that epic

**Convention going forward:** any project BACKLOG that exceeds ~10k tokens should adopt this two-layer pattern. Candidates to retrofit when they resume active expansion: `pa-warehouse/BACKLOG.md`, `nestle-audit/BACKLOG.md`.

---

## Cross-Reference

- **Strategic context (why this project):** [`career/jobspy-analysis-2026-05-12.md`](../../../career/jobspy-analysis-2026-05-12.md)
- **Research foundation (build spec):** [`research/sessions/rp-prey-001-portfolio-craft-plan.md`](../../../research/sessions/rp-prey-001-portfolio-craft-plan.md) — read lines 472-494 (Campaign Summary) before every story
- **Implementation gap analysis (pa-warehouse prep):** [`projects/hrds/pa-warehouse/docs/campaign-summary-build-gap.md`](../pa-warehouse/docs/campaign-summary-build-gap.md)
- **Workspace conventions:** [`projects/hrds/CLAUDE.md`](../CLAUDE.md)
- **Active Epic 0 prey:** [`work/queue/rp-prey-001-talent-retention-hard-data.md`](../../../work/queue/rp-prey-001-talent-retention-hard-data.md) — rewritten 2026-05-20 (Option B) from seed-concept to Epic 0 scope. Original ADEN-inspired hypothesis preserved above in `## Hypothesis Update`.
- **Blocker prey for Epic 1+:** [`work/queue/paw-prey-005-deliver-rp-preconditions.md`](../../../work/queue/paw-prey-005-deliver-rp-preconditions.md) — pa-warehouse delivers age + mutability metadata.
- **Per-epic detail files:** see `backlog/` directory (linked in Phase/Epic Overview above).
