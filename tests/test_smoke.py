"""Story 0.4.3 — minimum smoke tests so CI has something real to run.

These tests are deliberately *behavior-checks*, not presence-checks: they
exercise the scaffolding functions and assert on their observable effects, so a
regression in `set_global_seed` or `configure_plot_style` would fail CI rather
than silently slip through. See `/feast r01` Tier 2 #4 for the rationale.

Loop 2 (Story 2.6 + 2.8) expands this into the full reproducibility test
suite with model-wrapper + integration + data-quality tests.
"""

from __future__ import annotations

import random

import numpy as np
import seaborn as sns


def test_retention_imports() -> None:
    """The `retention` package must be importable from the editable install,
    and the canonical SEED constant must be 42."""
    from retention import config

    assert config.SEED == 42


def test_set_global_seed_is_deterministic() -> None:
    """Calling `set_global_seed` twice with the same seed must produce identical
    `random.random()` and `np.random.rand(...)` sequences.

    This is the contract every model wrapper in `retention.models` relies on:
    re-seed before training → same numbers → same model → same metrics.
    """
    from retention import config

    config.set_global_seed(42)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5).tolist()

    config.set_global_seed(42)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5).tolist()

    assert a_py == b_py, "random.random() not deterministic after set_global_seed"
    assert a_np == b_np, "np.random.rand() not deterministic after set_global_seed"


def test_set_global_seed_accepts_non_default_seed() -> None:
    """`set_global_seed(seed=99)` must seed differently than the default 42 —
    proves the seed argument actually propagates and isn't hardcoded internally."""
    from retention import config

    config.set_global_seed(42)
    a = random.random()

    config.set_global_seed(99)
    b = random.random()

    assert a != b, "set_global_seed ignored the seed argument (both seeds gave same value)"


def test_configure_plot_style_sets_colorblind_palette() -> None:
    """`configure_plot_style` must apply the seaborn `colorblind` palette globally.

    A regression here would mean every fairness comparison chart in Loops 3a/3c
    silently ships with the default `deep` palette (some shades indistinguishable
    to ~5% of male reviewers with red-green colorblindness). Catch it at unit-test
    time, not at Epic 8 polish time.
    """
    from retention import config

    config.configure_plot_style()
    current = sns.color_palette()
    expected = sns.color_palette("colorblind")
    assert list(current) == list(expected), (
        f"colorblind palette not active after configure_plot_style; "
        f"got first 2 colors: {list(current)[:2]}"
    )


def test_paths_resolve_under_project_root() -> None:
    """`PROJECT_ROOT`, `DATA_DIR`, `REPORTS_DIR`, `NOTEBOOKS_DIR` must all
    resolve to directories under the project root, and the project root itself
    must contain the canonical `pyproject.toml`."""
    from retention import config

    assert (config.PROJECT_ROOT / "pyproject.toml").exists(), (
        f"PROJECT_ROOT misresolved to {config.PROJECT_ROOT}"
    )
    for d in (config.DATA_DIR, config.REPORTS_DIR, config.NOTEBOOKS_DIR):
        assert config.PROJECT_ROOT in d.parents or d == config.PROJECT_ROOT, (
            f"{d} not under PROJECT_ROOT={config.PROJECT_ROOT}"
        )


def test_bq_constants_respect_env_override(monkeypatch) -> None:
    """`BQ_PROJECT_ID` and friends must default to the canonical pa-warehouse
    values but accept env-var override, so a reviewer cloning the repo can
    point at their own GCP project without editing source."""
    # Default value (no env var)
    monkeypatch.delenv("BQ_PROJECT_ID", raising=False)
    # Re-import to pick up the cleared env
    import importlib

    from retention import config

    importlib.reload(config)
    assert config.BQ_PROJECT_ID == "pa-warehouse-prod"

    # Env-var override
    monkeypatch.setenv("BQ_PROJECT_ID", "my-test-project")
    importlib.reload(config)
    assert config.BQ_PROJECT_ID == "my-test-project"

    # Reset cleanly for downstream tests
    monkeypatch.delenv("BQ_PROJECT_ID", raising=False)
    importlib.reload(config)
