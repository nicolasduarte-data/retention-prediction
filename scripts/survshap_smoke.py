"""Story 0.1.10 — survshap install gate smoke test.

Runs the minimum survshap workflow: fit a tiny RSF, wrap it with the
explainer, and call SurvSHAP on a single observation. If any line raises,
the gate fails and survshap must be dropped from pyproject.toml with the
fallback documented in docs/methodology.md.

Re-runnable. Side effects only execute when invoked as a script
(`uv run python scripts/survshap_smoke.py`), not on import — see the
`__main__` guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import survshap
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def main() -> None:
    """Run the SurvSHAP install gate smoke. See module docstring."""
    print(f"survshap.__version__: {survshap.__version__}")

    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(rng.random((n, 5)), columns=[f"f{i}" for i in range(5)])
    T = rng.integers(1, 100, n).astype(float)
    E = rng.integers(0, 2, n).astype(bool)
    y = Surv.from_arrays(event=E, time=T)

    rsf = RandomSurvivalForest(n_estimators=5, random_state=42).fit(X, y)
    explainer = survshap.SurvivalModelExplainer(model=rsf, data=X, y=y)
    ssh = survshap.PredictSurvSHAP()
    ssh.fit(explainer, X.iloc[:1])
    print("survshap smoke OK — SurvSHAP(t) workflow runs end-to-end.")


if __name__ == "__main__":
    main()
