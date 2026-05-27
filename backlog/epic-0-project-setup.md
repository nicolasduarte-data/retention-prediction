# Epic 0 — Project Setup + Integration Contract

**Prey:** `work/queue/rp-prey-001-talent-retention-hard-data.md`
**Status:** Ready
**Effort:** 1–2 days
**Public artifact at end:** Public repo with structure + CI passing + README skeleton + integration contract

> **Project-level context:** See `../BACKLOG.md` for stack decisions, hypothesis, research findings, repo structure, phase overview, risk register. This file covers Epic 0 detail only.

---

*Tracer bullet for the dev environment. Public repo by end of day 2. Production-grade scaffolding before any modeling.*

**Why this epic:** Most public HR data science repos fail the first 10 seconds of senior code review because the repo *itself* looks amateurish (no tests, no CI, no lock file, no pre-commit). This epic earns the credibility to be read.

**Epic 0 Definition of Done:**
- GitHub repo created, labeled "Setup phase — modeling stories in progress"
- `uv` (or `poetry`) lock file committed, `make install` works on a fresh clone
- Pre-commit hooks installed and passing locally
- CI green on first push: lint + (empty) tests pass
- `src/retention/` skeleton exists with `__init__.py` files
- `notebooks/` exists with one Jupyter notebook that imports from `src/retention/` and runs without error
- `docs/`, `reports/figures/`, `data/processed/` directories exist (with `.gitkeep` where empty)
- README skeleton with sections planned, SYNTHETIC banner, link to pa-warehouse
- Integration contract documented in `docs/architecture.md` — the schema rp expects from `marts.v_attrition_features`
- **All fragility gates pass:** survshap installs OR fallback documented (Story 0.1.10), alibi doesn't bloat install OR swapped to dalex (Story 0.1.11), notebook-logic-lint hook in place (Story 0.3.7)

### STORY 0.1 — Repo init + Python environment
**Why this story:** A reproducible Python environment is the foundation. Without a lock file, "it worked on my machine" becomes the rejection reason.

- [ ] 0.1.1 — Create local folder `projects/hrds/retention-prediction/`
- [ ] 0.1.2 — `git init`, create public GitHub repo `retention-prediction`, set remote
- [ ] 0.1.3 — Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or use Poetry — pick one)
- [ ] 0.1.4 — `uv init` (or `poetry init`) — create `pyproject.toml`
- [ ] 0.1.5 — Add core dependencies (pinned to ranges):
  ```
  python = "^3.11"
  pandas = "^2.0"
  numpy = "^1.24"
  scikit-learn = "^1.4"
  xgboost = "^2.0"
  lightgbm = "^4.0"
  interpret = "^0.5"            # EBM
  scikit-survival = "^0.22"
  shap = "^0.45"
  survshap = "^0.4"             # SurvSHAP(t) — verify install in Story 0.1.10
  dice-ml = "^0.11"
  alibi = "^0.9"                # verify in Story 0.1.11 (may swap to dalex)
  fairlearn = "^0.10"
  matplotlib = "^3.8"
  seaborn = "^0.13"
  plotly = "^5.20"
  google-cloud-bigquery = "^3.20"
  pandas-gbq = "^0.22"
  jupyterlab = "^4.0"
  mlflow = "^2.12"              # experiment tracking — wired into training cells in Story 2.7
  ```
- [ ] 0.1.6 — Add dev dependencies:
  ```
  pytest = "^8.0"
  pytest-cov = "^4.1"
  ruff = "^0.4"
  mypy = "^1.10"
  nbstripout = "^0.7"
  pre-commit = "^3.7"
  nbmake = "^1.5"               # for Risk 10 mitigation — notebook end-to-end CI
  ```
- [ ] 0.1.7 — Add optional causal deps (gated behind `[causal]` extra):
  ```
  dowhy = "^0.11"
  econml = "^0.15"
  ```
- [ ] 0.1.8 — Run `uv sync` (or `poetry install`) — verify lock file generated
- [ ] 0.1.9 — Commit `pyproject.toml` + lock file. **What you'll learn:** the difference between dependency declaration (`pyproject.toml`) and dependency resolution (`uv.lock`/`poetry.lock`).

- [ ] 0.1.10 — **🔴 survshap install gate (Risk 1 mitigation):**
  - Run `uv pip install survshap` in a fresh shell
  - Verify: `python -c "import survshap; print(survshap.__version__)"` succeeds
  - Verify: 5-line smoke script — import + dummy RSF + `survshap.SurvSHAP(model)` — no errors
  - **If install OR smoke fails:** drop `survshap` from `pyproject.toml`. Document in `docs/methodology.md` Limitations: *"SurvSHAP(t) deferred — install fragility on Python 3.11. Survival feature importance computed via standard SHAP wrapped on RSF prediction-at-horizon (Story 4.5 fallback)."* Update Story 4.5.1 to use fallback path.
  - **Why now, not Epic 4:** Catching this in Epic 0 (when the modeling pipeline doesn't exist yet) means Epic 4 has a planned fallback. Discovering in Epic 4 means scrambling under time pressure.

- [ ] 0.1.11 — **🟡 alibi install verification (Risk 5 mitigation):**
  - Time `uv add alibi` install in fresh env
  - Inspect: `uv tree | grep -i tensorflow` — confirm no TF transitive dep
  - **If install >5min OR pulls TensorFlow:** swap to `dalex` (lighter ALE alternative). Update `pyproject.toml`, update Story 6.4 (ALE plots) to use dalex API.
  - Document the choice in `docs/methodology.md`

### STORY 0.2 — Directory structure + scaffolding
**Why this story:** The `src/` layout is the canonical Python project pattern. Senior reviewers grep for it in 5 seconds.

- [ ] 0.2.1 — Create `src/retention/` package with `__init__.py`
- [ ] 0.2.2 — Create subpackages: `data/`, `features/`, `models/`, `evaluation/`, `fairness/`, `explainability/`, `writeback/` — each with `__init__.py`. **Exclude `causal/` for now** — only create if Epic 7 actually starts (Risk 14 — clean-skip discipline).
- [ ] 0.2.3 — Create `src/retention/config.py` with: `SEED = 42`, `PROJECT_ROOT = Path(__file__).parents[2]`, `DATA_DIR`, `REPORTS_DIR`, paths for BigQuery project ID and dataset names
- [ ] 0.2.4 — Configure `pyproject.toml` `[tool.setuptools.packages.find]` to find `src/retention/`
- [ ] 0.2.5 — Create `notebooks/` directory with a stub notebook `00_environment_check.ipynb` that imports from `src.retention` and prints the SEED
- [ ] 0.2.6 — Create `tests/conftest.py` (empty for now, will house fixtures later)
- [ ] 0.2.7 — Create `docs/`, `reports/figures/`, `reports/distribution/`, `data/processed/` — `.gitkeep` files in each
- [ ] 0.2.8 — Verify `uv run python -c "from retention import config; print(config.SEED)"` returns 42

- [ ] 0.2.9 — **🟡 Reproducibility + visual-design scaffold (Risk 6 prep + r02 Tier 2 fix 2026-05-21 — color-blind palette must be set in Epic 0, not Epic 8):**
  - In `src/retention/config.py`, add `set_global_seed()` function that sets `random.seed(SEED)`, `np.random.seed(SEED)`, `os.environ['PYTHONHASHSEED'] = str(SEED)`, and a comment for future `torch.manual_seed` (if added)
  - All notebooks call `from retention.config import set_global_seed; set_global_seed()` in cell 1
  - Document the convention in README "Reproducibility" section
  - **🟢 Visual design defaults (r02 fix — set NOW, not at Epic 8):**
    - Add to `src/retention/config.py`:
      ```python
      # Visual design — set once in Epic 0, used throughout Epics 1-8
      import matplotlib
      import seaborn as sns
      MATPLOTLIB_FIGSIZE = (10, 6)
      SEABORN_PALETTE = "colorblind"   # 8-color palette safe for deuteranopia/protanopia/tritanopia
      SEABORN_CONTEXT = "talk"          # readable at presentation scale
      SEABORN_STYLE = "whitegrid"

      def configure_plot_style():
          sns.set_theme(context=SEABORN_CONTEXT, style=SEABORN_STYLE, palette=SEABORN_PALETTE)
          matplotlib.rcParams['figure.figsize'] = MATPLOTLIB_FIGSIZE
          matplotlib.rcParams['savefig.dpi'] = 150
          matplotlib.rcParams['axes.spines.top'] = False
          matplotlib.rcParams['axes.spines.right'] = False
      ```
    - All notebooks call `from retention.config import configure_plot_style; configure_plot_style()` in cell 1 alongside `set_global_seed()`
    - **Why now, not Epic 8:** Every figure rendered in Epics 1-7 will use whatever palette is configured. Retrofitting in Epic 8 means re-rendering 12+ figures (calibration plots, ALE plots, fairness matrix, SHAP beeswarm, threshold sweep, EV sensitivity, survshap chart, etc.). Setting it in Epic 0 means every figure is colorblind-safe by default and presentation-grade from the first render.
    - **Why colorblind palette specifically:** ~5% of male reviewers have red-green colorblindness; rainbow defaults render some of your fairness comparison charts illegible to them. Senior reviewers notice when charts are accessible vs default-rainbow.

### STORY 0.3 — Pre-commit hooks + linting
**Why this story:** Pre-commit hooks catch garbage before it reaches GitHub. They're a senior-practitioner signal that costs nothing once set up.

- [ ] 0.3.1 — Create `.pre-commit-config.yaml` with hooks:
  ```yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.4.4
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.10.0
      hooks:
        - id: mypy
          args: [--strict, --ignore-missing-imports]
          files: ^src/
    - repo: https://github.com/kynan/nbstripout
      rev: 0.7.1
      hooks:
        - id: nbstripout
    - repo: local
      hooks:
        - id: notebook-logic-check
          name: Block function/class defs in notebooks
          entry: python scripts/check_notebook_logic.py
          language: system
          files: \.ipynb$
  ```
- [ ] 0.3.2 — Configure `ruff` in `pyproject.toml`: line-length 100, select all standard checks, exclude `notebooks/` from strict rules
- [ ] 0.3.3 — Configure `mypy` in `pyproject.toml`: `strict = true`, `ignore_missing_imports = true` (third-party stubs incomplete), Python target 3.11
- [ ] 0.3.4 — Install hooks: `uv run pre-commit install`
- [ ] 0.3.5 — Verify hooks run: `uv run pre-commit run --all-files` — fix any errors before first commit
- [ ] 0.3.6 — **What you'll learn:** `nbstripout` strips outputs from notebooks on commit — keeps the repo small and avoids merge conflicts on cell outputs. `ruff` is 10–100× faster than flake8+isort+pyupgrade combined.

- [ ] 0.3.7 — **🟢 Notebook-logic-lint hook (Risk 10 mitigation):**
  - Create `scripts/check_notebook_logic.py`: parses each `.ipynb`, scans code cells for top-level `def ` or `class ` patterns
  - If found: print "❌ Function/class definition in notebook [X] cell [Y] — move to src/retention/ and import" and exit 1
  - Acceptable in notebooks: imports, function calls, assignments, markdown narration, plot generation
  - Test with a notebook containing a `def foo(): pass` — confirm hook blocks
  - **Why:** notebook-defined logic bypasses mypy + pytest. Without this hook, "I'll just test it in the notebook" becomes a structural integrity hole.

- [ ] 0.3.8 — **🟢 Continuous secret scanning pre-commit hook (r02 Tier 2 fix 2026-05-21 — not just pre-push):**
  - Add `detect-secrets` to `.pre-commit-config.yaml`:
    ```yaml
    - repo: https://github.com/Yelp/detect-secrets
      rev: v1.5.0
      hooks:
        - id: detect-secrets
          args: ['--baseline', '.secrets.baseline']
    ```
  - Initialize baseline: `uv run detect-secrets scan > .secrets.baseline`
  - Commit `.secrets.baseline` (it documents what's known-safe; future scans diff against it)
  - **Why pre-commit, not just pre-push (Story 8.10.0):** secrets can be committed at any point during the build. A copy-pasted API key in a notebook, an env file accidentally tracked — caught at commit time, not 4 weeks later at ship time. Pre-commit is the proactive defense; Story 8.10.0 trufflehog scan is the final pre-push check.
  - Story 8.10.0 still runs at Epic 8 as the final verification — defense in depth.

### STORY 0.4 — GitHub Actions CI
**Why this story:** CI catches regressions before they ship. Even a smoke test signals "this person operates production-style."

- [ ] 0.4.1 — Create `.github/workflows/lint.yml`: runs `uv run pre-commit run --all-files` on push and PR
- [ ] 0.4.2 — Create `.github/workflows/test.yml`: runs `uv run pytest --cov=src --cov-report=term-missing` on push and PR
- [ ] 0.4.3 — Add minimum placeholder test in `tests/test_smoke.py`: `def test_imports(): from retention import config; assert config.SEED == 42`
- [ ] 0.4.4 — Verify both workflows green on first push
- [ ] 0.4.5 — Add CI status badges to README skeleton

### STORY 0.5.0 — Schema discovery spike (🟡 Tier 2 fix 2026-05-21 — hunt review)
**Why this story:** Best-in-class engineers don't write integration contracts from spec — they query the actual mart, dump the real schema, then write the contract. This prevents Story 0.5.3 from documenting a schema that's drifted before Epic 0 even started. ~10 minutes to do; saves hours of "why is the contract test failing" debugging in Epic 1.

- [ ] 0.5.0.1 — Run `bq show --schema --format=prettyjson pa-warehouse-prod:marts.v_attrition_features` (or `SELECT * FROM marts.v_attrition_features LIMIT 0` via pandas-gbq — free, schema only, zero data scanned)
- [ ] 0.5.0.2 — Dump column names + types + nullability to `docs/schema-discovery.md` (temporary scratchpad — can delete after Story 0.5.3 finalizes the contract)
- [ ] 0.5.0.3 — Cross-check against the BACKLOG's spec (Stories 1.2.2 lists expected features) and pa-warehouse's `_core.yml`
- [ ] 0.5.0.4 — Flag any drift before writing Story 0.5.3 — e.g., column renamed, type changed, new NULL pattern, expected column missing
- [ ] 0.5.0.5 — Use the dumped schema as source of truth for Story 0.5.3 — the contract documents reality, not aspiration
- [ ] 0.5.0.6 — **What you'll learn:** Reality-check the spec before documenting it. The spec is what was agreed in the campaign sessions (2026-04 / 05); the schema is what's actually deployed in BigQuery today. They may differ. The 10-minute spike is the cheapest insurance you can buy.

### STORY 0.5 — README skeleton + integration contract
**Why this story:** The README is the load-bearing portfolio artifact. Sketching its sections now makes filling them later mechanical, not a creative ask under fatigue.

- [ ] 0.5.1 — Create `README.md` skeleton with section headers:
  - One-line pitch
  - SYNTHETIC banner (mirror pa-warehouse's tone)
  - Live dashboard link (placeholder — fills in Epic 8)
  - Problem framing
  - **Governance disclaimer** (`Jott2121` voice — "decision support, not decision making")
  - **Forbidden uses** (EEOC + AEDT + EU AI Act Annex III — pull exact citations in Epic 8)
  - Methodology *(placeholder for Decision 1 — DL non-choice, Decision 2 — hypothesis reframe, Decision 3 — threshold over SMOTE, Decision 5 — MLflow, Decision 6 — controlled experiment framing; see rp-prey-001 prey for paragraphs)*
  - Results (placeholder)
  - Fairness audit summary *(placeholder for Decision 4)*
  - Interpretability summary (placeholder)
  - Limitations *(include survshap/alibi swap notes if Story 0.1.10/0.1.11 triggered)*
  - How to reproduce
  - License (MIT)
  - References (R1–R4 + G1–G3 + key papers + Grinsztajn / Shwartz-Ziv / Borisov / Rubenstein from rp-prey Architectural Decisions)
- [ ] 0.5.2 — Create `docs/architecture.md` describing data flow: `pa-warehouse BigQuery → pandas-gbq → src/retention/data/load.py → train/eval pipeline → marts.v_attrition_predictions write-back → pa-warehouse Looker Studio Page 5`
- [ ] 0.5.3 — Create `docs/integration_contract.md` documenting the **expected schema** of `marts.v_attrition_features`. List every column rp expects with name, type, source, mutability flag, intended use. This is the cross-project contract — if pa-warehouse changes the schema, this file changes first.
- [ ] 0.5.4 — Cross-link from pa-warehouse `BACKLOG.md` Loop 6 → this `integration_contract.md` (the prep work for Loop 6's write-back).

### STORY 0.6 — Makefile (or justfile) + initial commands
**Why this story:** Reproducibility goes from "read the README" to "run `make train`." Most public repos lack this.

- [ ] 0.6.1 — Create `Makefile` with targets (placeholders until Epic 1+):
  ```makefile
  .PHONY: install lint test data train evaluate fairness explain causal report mlflow-ui repro clean

  install:
  	uv sync

  lint:
  	uv run pre-commit run --all-files

  test:
  	uv run pytest --cov=src --cov-report=term-missing

  data:
  	uv run python -m retention.data.load

  train:
  	@echo "Epic 2 — coming soon"

  evaluate:
  	@echo "Epic 3 — coming soon"

  fairness:
  	@echo "Epic 5 — coming soon"

  explain:
  	@echo "Epic 6 — coming soon"

  causal:
  	@echo "Epic 7 — optional, see BACKLOG"

  report:
  	@echo "Epic 8 — coming soon"

  mlflow-ui:
  	@echo "Epic 2 — coming soon (Story 2.7)"

  repro:
  	@echo "Epic 2 — coming soon (Story 2.8 — reproducibility smoke test)"

  clean:
  	rm -rf .pytest_cache .ruff_cache .mypy_cache
  	find . -type d -name __pycache__ -exec rm -rf {} +
  ```
- [ ] 0.6.2 — Verify `make install`, `make lint`, `make test` all succeed
- [ ] 0.6.3 — **What you'll learn:** `make` targets give reviewers a one-line "how to reproduce" — much stronger signal than a wall of bash commands in the README.

**Epic 0 Checkpoint:** Public repo + CI green + skeleton + integration contract + fragility gates passed. Modeling work unblocked. **Visual design defaults set (colorblind palette via Story 0.2.9). Continuous secret scanning hook installed (Story 0.3.8).**

**Epic 0 close protocol:** follow `BACKLOG.md → 🛑 Epic Close Protocol` — including the energy-switch decision and application-cadence check.

---

## Mitigations landed in Epic 0

| Risk | Mitigation | Where |
|---|---|---|
| 🔴 1 — survshap install fragility | Install gate + fallback plan | Story 0.1.10 |
| 🟡 5 — alibi dep weight | Install verification + dalex swap | Story 0.1.11 |
| 🟡 6 — Reproducibility (prep) | `set_global_seed()` scaffolding | Story 0.2.9 |
| 🟢 10 — Notebook logic drift | Pre-commit hook blocking def/class in notebooks | Story 0.3.7 |
| 🔵 14 — Causal clean-skip (prep) | Exclude `causal/` subpackage from initial scaffold | Story 0.2.2 |
