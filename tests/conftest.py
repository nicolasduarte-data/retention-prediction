"""Shared pytest fixtures for the retention-prediction test suite.

Story 2.6 (Loop 2) adds the `synthetic_attrition_df` fixture used by both
the integration smoke test (2.6.5) and the data-quality tests (2.6.6/2.6.7).

Why a shared fixture rather than duplicating in each test file?

- **Single source of schema truth.** The fixture produces a DataFrame that
  matches every column in FEATURE_CATALOG. If the catalog adds a column,
  one fixture update covers all tests that depend on it.
- **Consistency.** The integration test and the data-quality tests exercise
  the same 100-row population. Results are directly comparable across tests.
- **Seed isolation.** ``np.random.default_rng(0)`` — seed 0 is reserved for
  fixtures; ``config.SEED`` (42) is reserved for model training. The two are
  intentionally different so the fixture isn't accidentally tuned to the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_attrition_df() -> pd.DataFrame:
    """100-row synthetic DataFrame with all FEATURE_CATALOG columns.

    Designed to be a valid input to the full pipeline:
    ``load → temporal_split → split_cohorts → extract_X_y → train → predict``.

    Schema matches ``docs/integration_contract.md``:
    - ``employee_id``               — unique string identifier
    - ``snapshot_date``             — date spread over 4 quarters (ensures
                                      temporal_split gets ≥ 15 rows per split)
    - 7 HRIS feature columns        — appear in both cohorts
    - 3 survey feature columns      — appear in hybrid cohort only
    - ``voluntary_exit_label``      — binary target, ~25% positive rate

    Why scope='session'?
        The fixture is read-only: no test mutates it. Session scope means it
        is built once and shared across the full test run — avoids re-building
        100 rows × 13 columns for every test function that requests it.

    Why seed=0 instead of config.SEED (42)?
        ``config.SEED`` is the project's model training seed. Using a different
        seed for test fixtures prevents the fixture from being accidentally
        calibrated to the model's random state — the tests should pass for any
        reasonable population, not just one that happens to produce nice metrics
        with the default seed.
    """
    rng = np.random.default_rng(0)
    n = 100

    # Spread rows over 4 calendar quarters so temporal_split always gets
    # at least 15 rows per split (25 rows × 4 quarters → train≈70, val≈15, test≈15).
    # Using quarterly dates (not daily) means ties are common; employee_id
    # acts as the deterministic tiebreak, matching the production contract.
    quarters = ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"]
    snapshot_dates = np.repeat(quarters, n // len(quarters))  # 25 per quarter

    return pd.DataFrame(
        {
            # ── Identifiers & temporal anchor ───────────────────────────────
            "employee_id": [f"EMP{i:03d}" for i in range(1, n + 1)],
            "snapshot_date": snapshot_dates,
            # ── HRIS features (cohort='both') ────────────────────────────────
            # tenure_months: continuous, right-skewed in production
            "tenure_months": rng.uniform(1.0, 60.0, n),
            # performance_tier: ordinal string — values mirror pa-warehouse enum
            "performance_tier": rng.choice(["2", "3", "4", "5"], n),
            # compa_ratio: key predictor — low values correlate with exit
            "compa_ratio": rng.uniform(0.7, 1.3, n),
            # is_critical_role: boolean — True means the employee fills a
            # role flagged as business-critical in the HRIS system
            "is_critical_role": rng.choice([True, False], n),
            # successor_count: integer — number of identified successors on file
            "successor_count": rng.integers(0, 4, n).astype(float),
            # age_at_window_close: numeric proxy for career stage
            "age_at_window_close": rng.uniform(22.0, 60.0, n),
            # gender: categorical — protected attribute in the fairness audit (Loop 3a)
            "gender": rng.choice(["M", "F", "NB"], n),
            # ── Survey features (cohort='hybrid_only') ───────────────────────
            # enps: Employee Net Promoter Score, range [-100, 100]
            "enps": rng.uniform(-100.0, 100.0, n),
            # engagement_score: 1–5 Likert composite
            "engagement_score": rng.uniform(1.0, 5.0, n),
            # manager_relationship_score: 1–5 Likert
            "manager_relationship_score": rng.uniform(1.0, 5.0, n),
            # ── Label ────────────────────────────────────────────────────────
            # ~25% positive rate — mirrors the production base rate (20–25%)
            # Low compa_ratio employees are more likely to exit (domain logic).
            "voluntary_exit_label": (
                rng.uniform(size=n)
                < (0.10 + 0.30 * (rng.uniform(0.7, 1.3, n) < 0.90).astype(float))
            ).astype(int),
        }
    )
