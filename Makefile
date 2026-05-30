# Makefile — retention-prediction
#
# Story 0.6 — Reproducibility goes from "read the README" to "run `make train`".
# Most public HR repos lack this. Even when most targets are placeholders today,
# the Makefile documents the surface the project promises by v1.0.
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/) on PATH
#   - On Windows: `scoop install make` (Git Bash doesn't ship with make)
#   - On macOS:   `brew install make` if not already present
#   - On Linux:   make ships with build-essential
#   - On CI:      ubuntu-latest ships make by default

.PHONY: help install lint test data train evaluate fairness explain report \
        mlflow-ui repro coverage mutation-test clean

# Default target — show available commands.
help:
	@echo "retention-prediction — Makefile targets"
	@echo ""
	@echo "  Setup:"
	@echo "    install     uv sync — install all deps (works today)"
	@echo "    clean       remove .pytest_cache, .ruff_cache, .mypy_cache, __pycache__"
	@echo ""
	@echo "  Quality gates (work today):"
	@echo "    lint           uv run pre-commit run --all-files"
	@echo "    test           uv run pytest with coverage"
	@echo "    coverage       uv run pytest --cov + open htmlcov/index.html"
	@echo "    repro          PYTHONHASHSEED=42 uv run pytest (deterministic hash)"
	@echo "    mutation-test  mutmut 2.x on src/retention/evaluation/ (Story 3.7)"
	@echo ""
	@echo "  Pipeline (ship as later stories land):"
	@echo "    data        Epic 1 (Loop 1 — Story 1.1) — BigQuery loader"
	@echo "    train       Epic 2 (Loop 1+2)            — Model training"
	@echo "    evaluate    Epic 3 (Loop 1+2)            — Evaluation rigor"
	@echo "    fairness    Epic 5 (Loop 3a)             — Fairness audit"
	@echo "    explain     Epic 6 (Loop 3c)             — SHAP / ALE / DiCE"
	@echo "    report      Epic 8 (Loop 4)              — Publish + writeback"
	@echo "    mlflow-ui   Epic 2 (Loop 2 — Story 2.7)  — Local MLflow UI"

# --- Setup ---

install:
	uv sync

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

# --- Quality gates (work today) ---

lint:
	uv run pre-commit run --all-files --show-diff-on-failure

test:
	uv run pytest --cov=src --cov-report=term-missing

coverage:
	uv run pytest --cov=src --cov-report=html
	@echo ""
	@echo "Open htmlcov/index.html to view the coverage report."

# Story 2.8 reproducibility smoke test (Loop 2) lives here in full.
# v0.1 partial: PYTHONHASHSEED is the missing piece set_global_seed() can't deliver
# at runtime (see src/retention/config.py:set_global_seed docstring).
repro:
	PYTHONHASHSEED=42 uv run pytest --cov=src --cov-report=term-missing

# Story 3.7 — mutation testing.
# Runs mutmut 2.x on src/retention/evaluation/ (configured in pyproject.toml
# under [tool.mutmut]).  Target: fewer than 5 surviving mutants.
# Takes ~15-30 minutes on first run; results cached in .mutmut-cache/.
# Interpret output: "Survived N" after `mutmut results` = mutations our tests
# did not detect.  Use `uv run mutmut show <ID>` to inspect each survivor.
mutation-test:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run mutmut run
	@echo ""
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run mutmut results
	@echo ""
	@echo "Target: < 5 surviving mutants. See docs/methodology.md -> Test Quality."
	@echo "Inspect survivors with: uv run mutmut show <ID>"

# --- Pipeline (placeholders until the relevant story lands) ---

data:
	@echo "Epic 1 — coming soon (Story 1.1 BigQuery loader)"

train:
	uv run python scripts/generate_comparison_notebook.py
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb
	@echo ""
	@echo "Training complete. mlruns/ populated with 6 runs (LR/GBM/EBM × hris_only/hybrid)."
	@echo "Run 'make mlflow-ui' to open the comparison view."

evaluate:
	uv run python scripts/generate_evaluation_notebook.py
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/03_evaluation_rigor.ipynb
	@echo ""
	@echo "Evaluation complete. Champion -> reports/models/champion.pkl;"
	@echo "registered as rp-champion/Production in mlruns/."
	@echo "Run 'make mlflow-ui', then open the Models tab to inspect the registry."

fairness:
	@echo "Epic 5 — coming soon (Loop 3a — Fairlearn audit + compound attribute + intervention category)"

explain:
	@echo "Epic 6 — coming soon (Loop 3c — SHAP + ALE + DiCE + EBM intrinsic)"

report:
	@echo "Epic 8 — coming soon (Loop 4 — README polish + Substack + LinkedIn + Looker Page 5 + writeback)"

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns
