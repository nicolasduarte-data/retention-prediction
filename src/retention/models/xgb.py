"""XGBoost retention model wrapper — Story 2.3 (cohort-aware update — Loop 2).

Thin wrapper around XGBClassifier that:
  - Wires eval_metric='aucpr' so training monitors AUC-PR directly.
  - Accepts (X_train, y_train, X_val, y_val) to enable eval_set monitoring.
  - Accepts cohort='hris_only' | 'hybrid' to select the right preprocessor.
  - Exposes `predict_proba` for the evaluation layer.
  - Holds the fitted preprocessor + classifier so the notebook can pass raw
    DataFrames without manual transform calls.

Story 2.3 scope: v0.1 skeleton — no hyperparameter search (that's Story 3.3,
Loop 2). Default n_estimators=100 is intentionally conservative; the notebook
verifies AUC-PR is better than random on synthetic data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retention import config
from retention.features.preprocessing import build_preprocessor

if TYPE_CHECKING:
    import pandas as pd


class RetentionModel:
    """Retention probability classifier: preprocessor + XGBClassifier.

    The preprocessor is fitted on X_train only. The fitted objects are
    assembled into an sklearn Pipeline stored as `self.pipeline` so that
    predict_proba(X_raw) transforms before scoring without manual steps.

    Usage::

        model = RetentionModel()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        proba = model.predict_proba(X_test)   # shape (n, 2), col 1 = P(exit)

    Attributes:
        pipeline: sklearn Pipeline (preprocessor → xgb). Set after `fit()`.
    """

    def __init__(
        self,
        *,
        cohort: Literal["hris_only", "hybrid"] = "hybrid",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        random_state: int = config.SEED,
        verbosity: int = 0,
    ) -> None:
        self._cohort = cohort
        self._xgb_params: dict[str, object] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
            eval_metric="aucpr",
        )
        if scale_pos_weight is not None:
            self._xgb_params["scale_pos_weight"] = scale_pos_weight
        self.pipeline: Pipeline | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> RetentionModel:
        """Fit the preprocessor on X_train, then fit XGBClassifier.

        Fitting order:
          1. Preprocessor is fit_transform'd on X_train only (no val leakage).
          2. If X_val provided, transform it with the already-fitted preprocessor.
          3. XGBClassifier is fit with optional eval_set for monitoring.
          4. Assemble self.pipeline from the already-fitted steps so that
             subsequent predict_proba(X_raw) auto-transforms raw DataFrames.

        Args:
            X_train: Raw feature DataFrame (catalog feature columns).
            y_train: Binary target (1 = voluntary exit).
            X_val: Optional validation features for eval_set monitoring.
            y_val: Optional validation labels.

        Returns:
            self (fluent interface).
        """
        preprocessor = build_preprocessor(cohort=self._cohort)
        X_train_t = preprocessor.fit_transform(X_train)
        y_train_arr = y_train.to_numpy()

        xgb = XGBClassifier(**self._xgb_params)

        xgb_fit_kwargs: dict[str, object] = {"verbose": False}
        if X_val is not None and y_val is not None:
            X_val_t = preprocessor.transform(X_val)
            xgb_fit_kwargs["eval_set"] = [(X_val_t, y_val.to_numpy())]

        xgb.fit(X_train_t, y_train_arr, **xgb_fit_kwargs)

        # Assemble into a Pipeline whose steps are already fitted.
        # sklearn Pipeline.predict_proba calls transform() (not fit_transform())
        # on all but the last step — safe since preprocessor is already fitted.
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", xgb),
            ]
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        """Return class probability estimates.

        Args:
            X: Raw feature DataFrame (same schema as X_train).

        Returns:
            ndarray of shape (n_samples, 2). Column 1 is P(voluntary_exit=1).

        Raises:
            RuntimeError: if called before `fit()`.
        """
        if self.pipeline is None:
            raise RuntimeError("Call .fit() before .predict_proba().")
        return self.pipeline.predict_proba(X)  # type: ignore[no-any-return]

    def predict(self, X: pd.DataFrame, *, threshold: float = 0.5) -> np.ndarray:  # type: ignore[type-arg]
        """Return binary predictions at `threshold`.

        Default threshold=0.5 is a baseline — Loop 2 will tune per the
        precision/recall tradeoff the people analytics team specifies.
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
