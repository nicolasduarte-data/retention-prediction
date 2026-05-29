"""Champion selection + persistence — Story 3.6.

Epic 3 built five evaluation lenses — discrimination (3.1), calibration (3.2),
threshold (3.3), expected value (3.4), unbiased generalisation (3.5, nested CV).
This module is where they converge into a single, defensible decision:

    *Of the six model × cohort cells, which one do we ship?*

A champion is not "the highest AUC-PR cell." That is the mistake this module
exists to avoid. A cell can win on discrimination while being badly
mis-calibrated — and every downstream HR action (the threshold, the
expected-value case) is computed from probabilities, so a cell whose
probabilities lie is unsafe to operate even if it ranks well. Selection here is
therefore a **gated rank**, not a sort.

────────────────────────────────────────────────────────────────────────────
THE SELECTION RULE — calibration gate, then discrimination rank
────────────────────────────────────────────────────────────────────────────
1. **Calibration gate.** A cell is *eligible* only if its Brier score is at or
   below the base-rate Brier, ``β·(1 − β)`` where β is the positive base rate.

   Why that number is the right bar: the best possible *constant* predictor
   outputs β for everyone, and its Brier is exactly ``β·(1 − β)`` (derivation in
   ``_base_rate_brier``). That constant predictor has zero discrimination — it
   ranks no-one. So a model scoring *below* ``β·(1 − β)`` is provably adding
   value beyond "predict the base rate": its lower squared error can only come
   from probabilities that move in the right direction. A model *above* that bar
   is worse than the trivial baseline at the one thing Brier measures, and we
   refuse to operate its probabilities no matter how well it ranks.

2. **Discrimination rank.** Among the eligible cells, the champion is the one
   with the highest AUC-PR — the primary metric across every loop, chosen
   because it is invariant to the true-negative flood that inflates AUC-ROC
   under imbalance.

3. **Honest fallback.** If *no* cell clears the gate, we do not pretend one did.
   We select the highest-AUC-PR cell overall, set ``passed_calibration_gate =
   False``, and say so in the rationale: the ranking is usable, the
   probabilities are not — calibrate before trusting any threshold or EV number.

This mirrors exactly the finding the comparison + calibration notebooks already
surfaced: GBM was the only family whose Brier (≈ 0.158) sat below the base-rate
Brier (≈ 0.153–0.162), while LR and EBM (Brier ≈ 0.235) failed the bar because
``class_weight='balanced'`` / ``compute_sample_weight`` inflate their
probabilities. The rule is the codification of that argument, not a new claim.

────────────────────────────────────────────────────────────────────────────
TWO DATACLASSES — the decision vs. the deployable thing
────────────────────────────────────────────────────────────────────────────
``ChampionSelection`` is the *decision*: numbers + rationale, no model object.
It is what ``select_champion`` returns and what the notebook prints. It is cheap
to build, trivial to test, and carries no pickled estimator.

``ChampionArtifact`` is the *deployable thing*: the fitted model + its operating
threshold + provenance (which selection produced it, validation vs. test
metrics, seed, timestamp). It is what ``persist_champion`` writes to
``reports/models/champion.pkl`` and what Loop 4's write-back will load.

Keeping them separate means the selection logic stays a pure function of a
metrics table — no model fitting, no I/O — and the persistence logic stays a
thin, well-typed pickle wrapper. Each is tested in isolation.

────────────────────────────────────────────────────────────────────────────
TEST-SET DISCIPLINE — the protocol this module assumes
────────────────────────────────────────────────────────────────────────────
Every metric that feeds ``select_champion`` is a **validation-set** number
(threshold and EV stories both promised "confirm on test once, at champion
selection"). The champion is selected on validation, then confirmed on the
held-out test set exactly once — those test numbers are stored on the artifact
as ``test_metrics`` beside the ``val_metrics`` they were chosen on, so the
optimism gap is visible in the persisted record, not hidden.

Story 3.6 — see ``backlog/epic-3-evaluation-rigor.md``.
"""

from __future__ import annotations

import dataclasses
import pickle  # noqa: S403 — we only ever unpickle our own champion.pkl (see load_champion)
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from retention import config

if TYPE_CHECKING:
    import pandas as pd


# The columns ``select_champion`` requires in the summary table. Declared as a
# constant (not buried in the function) so the notebook builder and the tests
# share one source of truth for the contract.
REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "cohort",
    "auc_pr",
    "brier",
    "ece",
    "threshold",
)

# Default on-disk location for the persisted champion. Absolute (resolved from
# the package), so it is independent of the caller's working directory — the
# same cwd-independence reason ``tracking.log_run`` pins its MLflow URI.
DEFAULT_CHAMPION_PATH: Path = config.REPORTS_DIR / "models" / "champion.pkl"


class SupportsPredictProba(Protocol):
    """Structural type for any fitted estimator the artifact can wrap.

    We deliberately type the champion's model by *behaviour*, not by class:
    the GBM champion is a ``RetentionModel``, but LR/EBM are sklearn
    ``Pipeline``s, and Loop 4 may wrap something else again. All they must share
    is a ``predict_proba`` that returns an array — that is the only method the
    artifact calls. Typing to this Protocol keeps ``ChampionArtifact`` decoupled
    from any one model class while still satisfying ``mypy --strict``.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        """Return class probabilities; column 1 is the positive class P(exit)."""
        ...


def _base_rate_brier(base_rate: float) -> float:
    """Brier score of the best constant predictor at ``base_rate`` β.

    The best constant predictor outputs β for every employee. Its Brier is the
    mean squared error of that constant against the 0/1 labels:

        a fraction β of labels are 1  → error (β − 1)²
        a fraction (1 − β) are 0      → error (β − 0)² = β²

        Brier = β·(1 − β)² + (1 − β)·β²
              = β·(1 − β)·[(1 − β) + β]
              = β·(1 − β)

    So the bar is just the variance of a Bernoulli(β). At β = 0.188 that is
    ≈ 0.153 — the "random baseline Brier" the calibration module quotes.
    """
    return base_rate * (1.0 - base_rate)


@dataclasses.dataclass(frozen=True)
class ChampionSelection:
    """The selection *decision* — numbers + rationale, no model object.

    Frozen because a decision is a record: once ``select_champion`` has made the
    call on a given metrics table, mutating it after the fact would desync the
    rationale from the numbers that justified it.
    """

    model_name: str
    cohort: str
    auc_pr: float
    brier: float
    ece: float
    threshold: float
    criterion: str
    passed_calibration_gate: bool
    base_rate_brier: float
    rationale: str

    def summary(self) -> str:
        """One-block human summary — what the notebook prints at selection time."""
        gate = "PASS" if self.passed_calibration_gate else "FAIL"
        return (
            f"Champion: {self.model_name} × {self.cohort}\n"
            f"  AUC-PR = {self.auc_pr:.3f}  |  "
            f"Brier = {self.brier:.3f} (gate ≤ {self.base_rate_brier:.3f}: {gate})  |  "
            f"ECE = {self.ece:.3f}\n"
            f"  Operating threshold = {self.threshold:.2f} "
            f"({self.criterion.upper()}-optimal, validation)\n"
            f"  {self.rationale}"
        )


def select_champion(
    summary: pd.DataFrame,
    *,
    base_rate: float,
    criterion: str = "f2",
) -> ChampionSelection:
    """Choose the champion cell from a six-row model × cohort metrics table.

    The rule is the calibration-gated rank documented at module scope: keep the
    cells whose Brier clears the base-rate bar, then take the highest AUC-PR
    among them; fall back to highest AUC-PR overall (flagged) if none clear it.

    Args:
        summary: One row per model × cohort cell. Must contain the columns in
            ``REQUIRED_SUMMARY_COLUMNS``: ``model``, ``cohort``, ``auc_pr``,
            ``brier``, ``ece``, ``threshold`` (the cell's own operating
            threshold). Extra columns (precision@k, EV, breakeven) are ignored
            by the rule but fine to carry for reporting.
        base_rate: Positive-class prevalence β on the split the metrics were
            computed on (validation). Drives the calibration gate
            ``brier ≤ β·(1 − β)``.
        criterion: The threshold criterion the ``threshold`` column was
            optimised under — stored on the result for provenance and the
            summary string. Default ``"f2"`` (the [FLIP-RISK] recall-weighted
            operating point the threshold story argues for).

    Returns:
        A ``ChampionSelection`` describing the winning cell and why it won.

    Raises:
        ValueError: if ``summary`` is empty, is missing a required column, or
            ``base_rate`` is not strictly inside (0, 1) — a degenerate base rate
            makes the gate ``β·(1 − β)`` meaningless.

    Teaching note — why ``idxmax`` is the right tie-break:
        On a plateau (two cells with identical AUC-PR), ``Series.idxmax``
        returns the *first* index label. The summary table is built in a fixed
        model × cohort order, so the tie-break is deterministic and reproducible
        — the same table always yields the same champion.
    """
    missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in summary.columns]
    if missing:
        raise ValueError(
            f"summary is missing required column(s) {missing}. "
            f"Expected at least {list(REQUIRED_SUMMARY_COLUMNS)}, got {list(summary.columns)}."
        )
    if summary.empty:
        raise ValueError("summary has no rows; cannot select a champion from an empty table.")
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}.")

    gate = _base_rate_brier(base_rate)

    # Eligible = clears the calibration gate. ``<=`` (not ``<``) so a cell that
    # exactly matches the trivial baseline is given the benefit of the doubt —
    # the rank step then decides among such ties on discrimination.
    eligible = summary[summary["brier"] <= gate]
    passed_gate = not eligible.empty
    pool = eligible if passed_gate else summary

    # Highest AUC-PR in the pool. ``idxmax`` → label of the max; ``.loc`` → row.
    best_label = pool["auc_pr"].idxmax()
    best = pool.loc[best_label]

    model_name = str(best["model"])
    cohort = str(best["cohort"])
    auc_pr = float(best["auc_pr"])
    brier = float(best["brier"])
    ece = float(best["ece"])
    threshold = float(best["threshold"])

    if passed_gate:
        n_eligible = int(len(eligible))
        rationale = (
            f"Selected on AUC-PR among the {n_eligible} cell(s) clearing the calibration "
            f"gate (Brier ≤ {gate:.3f}); its probabilities beat the base-rate baseline, so "
            f"the threshold and expected-value decisions built on them are trustworthy."
        )
    else:
        rationale = (
            f"NO cell cleared the calibration gate (Brier ≤ {gate:.3f}); selected on AUC-PR "
            f"alone. The ranking is usable, but the probabilities are worse than predicting "
            f"the base rate — calibrate (e.g. isotonic/Platt) before trusting any threshold "
            f"or EV figure."
        )

    return ChampionSelection(
        model_name=model_name,
        cohort=cohort,
        auc_pr=auc_pr,
        brier=brier,
        ece=ece,
        threshold=threshold,
        criterion=criterion,
        passed_calibration_gate=passed_gate,
        base_rate_brier=gate,
        rationale=rationale,
    )


@dataclasses.dataclass(frozen=True)
class ChampionArtifact:
    """The deployable champion: fitted model + operating threshold + provenance.

    This is what ``persist_champion`` pickles and Loop 4 loads. Beyond the model
    and threshold it records *how it was chosen* (``selection``) and *how it
    scored* on both the split it was selected on (``val_metrics``) and the
    held-out confirmation (``test_metrics``), so the persisted file is a
    self-describing record, not an opaque blob.

    Frozen for the same reason as the selection: a shipped artifact is a fixed
    point. Re-selecting produces a *new* artifact rather than mutating this one.
    """

    model: SupportsPredictProba
    model_name: str
    cohort: str
    threshold: float
    feature_names: list[str]
    seed: int
    selection: ChampionSelection
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    created_utc: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        """Positive-class probability P(exit) as a 1-D array.

        Delegates to the wrapped model and normalises the shape: estimators that
        return the sklearn ``(n, 2)`` matrix are reduced to column 1; estimators
        that already return a 1-D positive-class vector pass through unchanged.
        """
        proba = np.asarray(self.model.predict_proba(X))
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        """Binary flag decisions at the champion's operating threshold.

        ``>=`` matches the decision rule used throughout the threshold and EV
        modules, so a 1 here means exactly what it meant when the threshold was
        chosen: "flag this employee for a retention conversation."
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def summary(self) -> str:
        """Human-readable provenance block for logs and the notebook."""
        val = "  ".join(f"{k}={v:.3f}" for k, v in self.val_metrics.items())
        test = "  ".join(f"{k}={v:.3f}" for k, v in self.test_metrics.items())
        return (
            f"ChampionArtifact — {self.model_name} × {self.cohort} "
            f"(threshold={self.threshold:.2f}, seed={self.seed})\n"
            f"  features ({len(self.feature_names)}): {', '.join(self.feature_names)}\n"
            f"  validation: {val}\n"
            f"  test:       {test}\n"
            f"  created:    {self.created_utc}"
        )


def persist_champion(artifact: ChampionArtifact, path: Path | None = None) -> Path:
    """Pickle ``artifact`` to ``path`` (default ``reports/models/champion.pkl``).

    Args:
        artifact: The champion to persist.
        path: Destination. Defaults to ``DEFAULT_CHAMPION_PATH``. Parent
            directories are created if absent.

    Returns:
        The path written — convenient for logging and for the notebook to echo.

    Teaching note — why pickle (and why that is acceptable here):
        The artifact wraps a fitted XGBoost/sklearn estimator, which is itself
        only portable via pickle/cloudpickle. Pickle's deserialisation can run
        arbitrary code, so it is unsafe on *untrusted* files — but this file is
        produced by our own pipeline and read back by our own code, so the trust
        boundary is never crossed. ``load_champion`` documents the same contract.
    """
    target = path or DEFAULT_CHAMPION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        pickle.dump(artifact, fh)
    return target


def load_champion(path: Path | None = None) -> ChampionArtifact:
    """Load a champion previously written by ``persist_champion``.

    Args:
        path: Source. Defaults to ``DEFAULT_CHAMPION_PATH``.

    Returns:
        The deserialised ``ChampionArtifact``.

    Raises:
        FileNotFoundError: if ``path`` does not exist — with a hint to run the
            evaluation notebook that produces it.
        TypeError: if the unpickled object is not a ``ChampionArtifact`` (the
            file was overwritten by something else).

    Security note:
        Only ever point this at ``champion.pkl`` files produced by this project.
        Unpickling executes code embedded in the file; never load a champion you
        did not generate. See ``persist_champion`` for the trust-boundary rationale.
    """
    source = path or DEFAULT_CHAMPION_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"No champion at {source}. Run notebooks/03_evaluation_rigor.ipynb "
            f"(or `make evaluate`) to select and persist one."
        )
    with source.open("rb") as fh:
        obj = pickle.load(fh)  # noqa: S301 — trusted, self-produced file (see docstring)
    if not isinstance(obj, ChampionArtifact):
        raise TypeError(
            f"{source} did not contain a ChampionArtifact (got {type(obj).__name__}). "
            f"The file may have been overwritten — regenerate it via the evaluation notebook."
        )
    return obj


__all__ = [
    "DEFAULT_CHAMPION_PATH",
    "REQUIRED_SUMMARY_COLUMNS",
    "ChampionArtifact",
    "ChampionSelection",
    "SupportsPredictProba",
    "load_champion",
    "persist_champion",
    "select_champion",
]
