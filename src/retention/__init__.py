"""retention — Behavioral-signal-based retention prediction package.

Loop 1 (walking skeleton) ships the minimum viable end-to-end pipeline:
BigQuery load → temporal split → XGBoost on the hybrid cohort → AUC-PR.

Subpackages:
- `data`         — loaders, splits, contract validation
- `features`     — feature catalog (FeatureSpec) + preprocessing pipeline
- `models`       — model wrappers (LR / XGBoost / EBM / RSF / Cox)
- `evaluation`   — metrics, calibration, threshold sweep, EV
- `fairness`     — Fairlearn audit + compound-attribute analysis
- `explainability` — SHAP / ALE / DiCE / SurvSHAP(t)
- `writeback`    — predictions back to pa-warehouse marts

`causal/` is intentionally excluded — Epic 7 is CUT from v1.0 (v1.1 roadmap only).

Module-level `config` exposes constants (SEED, paths) and the two scaffolding
functions every notebook calls in cell 1: `set_global_seed()` and
`configure_plot_style()`.
"""

from importlib.metadata import version as _version

from retention import config

__all__ = ["config"]
# Single source of truth: pyproject.toml [project].version. Reading via
# importlib.metadata guarantees __version__ never drifts from the distribution
# version a fresh-clone reviewer sees. See `/feast r02` Tier 1 #9.
__version__ = _version("retention-prediction")
