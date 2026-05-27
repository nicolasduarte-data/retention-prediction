"""Project-wide constants + reproducibility / visual-design scaffolding.

This module is imported by every notebook and every script in the project.
Two functions are the load-bearing scaffolding:

- `set_global_seed()` — seeds `random` and `numpy` for the current process,
  and writes `PYTHONHASHSEED` to `os.environ` for downstream subprocesses.
  See the function docstring for the *exact* reproducibility envelope.
- `configure_plot_style()` — sets seaborn's `colorblind` palette + matplotlib
  defaults so every figure rendered anywhere in the project is colorblind-safe
  and presentation-grade *by default* (no retrofit at Epic 8).

Story 0.2.3 + 0.2.9 — Risk 6 (reproducibility) + r02 Tier 2 fix 2026-05-21
(colorblind palette must be set in Epic 0, not Epic 8 — otherwise every
Epic 1-7 figure has to be re-rendered).

**Import-cost note**: `matplotlib` and `seaborn` are lazy-imported inside
`configure_plot_style()` so that consumers of this module that don't render
figures (tests, data loaders, model training scripts) don't pay the ~4s
matplotlib import cost. See `/feast r01` Tier 2 #2.

**Configuration override**: BigQuery project/dataset names default to the
canonical pa-warehouse values but accept env-var override (`BQ_PROJECT_ID`,
`BQ_DATASET_MARTS`, etc.) so a reviewer cloning the repo can point at their
own GCP project without editing source. See `/feast r01` Tier 2 #7.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- #
# Constants                                                              #
# --------------------------------------------------------------------- #

SEED: int = 42
"""Global random seed. Matches workforce-analytics + pa-warehouse convention."""

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the retention-prediction project root."""

DATA_DIR: Path = PROJECT_ROOT / "data"
"""Local data directory (raw + processed). Most data lives in BigQuery; this
holds CSV snapshots for offline reproducibility (see BACKLOG Decision: 'Data source')."""

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
"""Figures + distribution artifacts (model card screenshots, MLflow exports, etc.)."""

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
"""Thin narrative notebooks that import from `retention.*`."""

# --------------------------------------------------------------------- #
# BigQuery integration contract (env-overridable)                        #
# --------------------------------------------------------------------- #

BQ_PROJECT_ID: str = os.getenv("BQ_PROJECT_ID", "pa-warehouse-prod")
"""GCP project hosting the pa-warehouse mart layer.
Override with `export BQ_PROJECT_ID=my-project` to point at a different GCP project."""

BQ_DATASET_MARTS: str = os.getenv("BQ_DATASET_MARTS", "marts")
"""Dataset name for production marts (read source for features, write target for predictions)."""

BQ_DATASET_STAGING: str = os.getenv("BQ_DATASET_STAGING", "staging")
"""Dataset name for staging / dbt intermediate models (rarely read by rp)."""

BQ_TABLE_ATTRITION_FEATURES: str = os.getenv("BQ_TABLE_ATTRITION_FEATURES", "v_attrition_features")
"""Feature mart consumed by the loader (see `docs/integration_contract.md`)."""

BQ_TABLE_ATTRITION_PREDICTIONS: str = os.getenv(
    "BQ_TABLE_ATTRITION_PREDICTIONS", "v_attrition_predictions"
)
"""Write-back target for Epic 8 — predictions surfaced in pa-warehouse Looker Studio Page 5."""

# --------------------------------------------------------------------- #
# Visual design defaults (Story 0.2.9)                                   #
# --------------------------------------------------------------------- #

MATPLOTLIB_FIGSIZE: tuple[float, float] = (10.0, 6.0)
SEABORN_PALETTE: str = "colorblind"
"""8-color palette safe for deuteranopia / protanopia / tritanopia. ~5% of male
reviewers have red-green colorblindness; rainbow defaults render some fairness
comparison charts illegible to them. Set here so every figure across Loops 1-4
ships colorblind-safe by default."""
SEABORN_CONTEXT: str = "talk"
SEABORN_STYLE: str = "whitegrid"


# --------------------------------------------------------------------- #
# Scaffolding functions (called from every notebook cell 1)              #
# --------------------------------------------------------------------- #


def set_global_seed(seed: int = SEED) -> None:
    """Seed `random` + `numpy` for the current process; export `PYTHONHASHSEED` for subprocesses.

    Reproducibility envelope (read this — the function does less than its name suggests):

    **Current process (works as expected):**
      - `random.seed(seed)` — Python's `random` module is deterministic across re-seeds.
      - `np.random.seed(seed)` — NumPy's global RNG is deterministic across re-seeds.

    **Subprocesses (works, but matters only when you spawn them):**
      - `os.environ['PYTHONHASHSEED'] = str(seed)` — child processes (e.g. `multiprocessing.Pool`,
        xgboost's internal `n_jobs>1` workers, `subprocess.run`) inherit this env var and use it.
        So `hash()` and set/dict iteration order *in the child process* are deterministic.

    **What this function does NOT do:**
      - It does **not** make `hash(str)` deterministic in the *current* process. `PYTHONHASHSEED`
        is read once at interpreter startup; writing it via `os.environ` after that has no effect
        on the current process's hash randomization. To get deterministic `hash(str)` in your
        notebook or test, the *invoker* must set the env var before `uv run` / `python` starts:
        `PYTHONHASHSEED=42 uv run pytest tests/`. The project's Makefile target `repro` should
        do this once Story 0.6 ships.
      - It does **not** seed `torch.manual_seed()` — no deep learning on tabular HR data per the
        Grinsztajn 2022 / Shwartz-Ziv 2022 / Borisov 2022 decision (see BACKLOG `## Stack Decisions`).
        Add it here if Epic 7 (causal, v1.1 only) ever needs it.
      - It does **not** verify that downstream libraries (sklearn, xgboost, lightgbm) actually
        accept `random_state=SEED`. Each model wrapper in `retention.models` is responsible for
        passing the seed through. Story 2.8 reproducibility smoke test verifies end-to-end.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_plot_style() -> None:
    """Apply colorblind palette + matplotlib defaults globally.

    Idempotent — safe to call multiple times. Sets `sns.set_theme(...)` first
    (which applies palette + style + context) then overrides specific matplotlib
    rcParams for figsize, savefig dpi, and despine.

    `matplotlib` and `seaborn` are imported lazily here (not at module scope)
    so that `from retention import config` stays cheap (~50ms) for consumers
    that never call this function. See `/feast r01` Tier 2 #2.
    """
    import matplotlib
    import seaborn as sns

    sns.set_theme(
        context=SEABORN_CONTEXT,
        style=SEABORN_STYLE,
        palette=SEABORN_PALETTE,
    )
    matplotlib.rcParams["figure.figsize"] = MATPLOTLIB_FIGSIZE
    matplotlib.rcParams["savefig.dpi"] = 150
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
