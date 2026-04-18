"""
optuna_tuner.py -- Hyperparameter tuning for DirectionClassifier using Optuna.

Uses TimeSeriesSplit cross-validation (no shuffling, time-series safe).
The objective is mean ROC AUC across folds.

Search space (XGBoost / LightGBM):
  - n_estimators    : [100, 600]
  - max_depth       : [3, 10]
  - learning_rate   : [0.005, 0.20] (log scale)
  - subsample       : [0.5, 1.0]
  - colsample_bytree: [0.5, 1.0]

Usage:
    from models.optuna_tuner import OptunaTuner
    tuner = OptunaTuner(model_type="xgboost", n_trials=30)
    best_params, best_score = tuner.tune(X, y)
    clf = tuner.build_best_classifier()
    clf.fit(X_train, y_train)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from models.direction_classifier import DirectionClassifier

warnings.filterwarnings("ignore")


class OptunaTuner:
    """
    Bayesian hyperparameter optimization for direction classifiers.

    Parameters
    ----------
    model_type : "xgboost" | "lightgbm"
    n_trials : int
        Number of Optuna trials. 30 is a reasonable default for ~1300 rows.
    n_splits : int
        TimeSeriesSplit folds for the inner CV loop.
    random_state : int
    direction : "maximize" (we maximize AUC)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_trials: int = 30,
        n_splits: int = 4,
        random_state: int = 42,
    ):
        if model_type not in ("xgboost", "lightgbm"):
            raise ValueError("model_type must be 'xgboost' or 'lightgbm'")

        self.model_type   = model_type
        self.n_trials     = n_trials
        self.n_splits     = n_splits
        self.random_state = random_state

        self.best_params_: dict = {}
        self.best_score_:  float = 0.0
        self._study = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def tune(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Run Optuna optimization. Returns (best_params, best_auc).
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError as e:
            raise ImportError(
                "optuna is not installed. Run: pip install optuna"
            ) from e

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective = self._make_objective(X, y)

        self._study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            study_name=f"tuner_{self.model_type}",
        )
        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params_ = self._study.best_params
        self.best_score_  = float(self._study.best_value)
        return self.best_params_, self.best_score_

    def build_best_classifier(self) -> DirectionClassifier:
        """Build a fresh unfitted DirectionClassifier with the best params."""
        if not self.best_params_:
            raise RuntimeError("Call .tune() first.")
        return DirectionClassifier(
            model_type=self.model_type,
            n_estimators=int(self.best_params_["n_estimators"]),
            max_depth=int(self.best_params_["max_depth"]),
            learning_rate=float(self.best_params_["learning_rate"]),
            subsample=float(self.best_params_["subsample"]),
            colsample_bytree=float(self.best_params_["colsample_bytree"]),
            random_state=self.random_state,
        )

    def trial_history(self) -> pd.DataFrame:
        """Return all trials as a DataFrame for analysis."""
        if self._study is None:
            return pd.DataFrame()
        rows = []
        for t in self._study.trials:
            row = {"trial": t.number, "auc": t.value, **t.params}
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private: Optuna objective
    # ------------------------------------------------------------------

    def _make_objective(self, X: pd.DataFrame, y: pd.Series):
        """Closure capturing X, y for Optuna's objective signature."""

        def objective(trial) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            return self._cv_score(X, y, params)

        return objective

    def _cv_score(self, X: pd.DataFrame, y: pd.Series, params: dict) -> float:
        """Time-series CV AUC for a given parameter set."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        aucs = []

        for train_idx, test_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            if len(y_test.unique()) < 2:
                continue  # skip degenerate fold

            clf = DirectionClassifier(
                model_type=self.model_type,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=self.random_state,
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
            try:
                auc = roc_auc_score(y_test, proba)
            except Exception:
                auc = 0.5
            aucs.append(auc)

        return float(np.mean(aucs)) if aucs else 0.5
