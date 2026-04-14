"""
direction_classifier.py — Binary direction classifier (up/down next day).

Design rules:
  - Time-series aware: NEVER shuffle data, always train on past / test on future
  - Two model flavours: XGBoost (default) and LightGBM
  - Walk-forward cross-validation with TimeSeriesSplit
  - Outputs: class label {0,1} AND calibrated probability (used as signal strength)
  - No data leakage: features at row i use only data up to bar i
  - Target: target_direction_1d (1 = next-day close > today's close)

Why NOT deep learning here:
  - 1,300–2,000 rows per asset is too few for reliable neural net training
  - Gradient boosting consistently outperforms DL on tabular data at this scale
  - Faster iteration, interpretable feature importances
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class DirectionClassifier:
    """
    Predict next-day price direction (up=1 / flat-or-down=0).

    Parameters
    ----------
    model_type : 'xgboost' | 'lightgbm'
    n_estimators : int
    max_depth : int
    learning_rate : float
    scale_features : bool
        Gradient boosting is tree-based and invariant to feature scale,
        so scaling is OFF by default. Set True if you later add linear
        meta-learners on top.
    random_state : int
    """

    SUPPORTED_MODELS = ("xgboost", "lightgbm")

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_features: bool = False,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.model_type       = model_type
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_features   = scale_features
        self.random_state     = random_state

        self._model   = None
        self._scaler  = StandardScaler() if scale_features else None
        self.feature_names_: list = []
        self.is_fitted_: bool     = False

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    def _build_model(self):
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                eval_metric="logloss",
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            )
        else:  # lightgbm
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionClassifier":
        """
        Train on (X, y). X must be a DataFrame of features; y must be {0,1}.
        Stores feature names for later importance lookup.
        """
        self.feature_names_ = list(X.columns)
        X_arr = self._transform(X, fit=True)
        self._model = self._build_model()
        self._model.fit(X_arr, y.values)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return class labels {0, 1} aligned to X.index."""
        self._check_fitted()
        X_arr = self._transform(X, fit=False)
        preds = self._model.predict(X_arr)
        return pd.Series(preds, index=X.index, name="ml_signal")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return P(up) in [0, 1] aligned to X.index.
        Values > 0.5 → model leans long; < 0.5 → model leans short/flat.
        """
        self._check_fitted()
        X_arr = self._transform(X, fit=False)
        proba = self._model.predict_proba(X_arr)[:, 1]
        return pd.Series(proba, index=X.index, name="ml_proba")

    def feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted Series (descending)."""
        self._check_fitted()
        return (
            pd.Series(self._model.feature_importances_, index=self.feature_names_, name="importance")
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Walk-forward cross-validation (time-series safe)
    # ------------------------------------------------------------------

    def walk_forward_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """
        TimeSeriesSplit cross-validation.
        Each fold: train on all past bars, test on the next future block.
        No shuffling. No future data leaks into training.

        Returns
        -------
        dict with:
          fold_metrics : list of dicts (one per fold)
          mean_accuracy, mean_auc, mean_f1  (averages across folds)
          std_accuracy,  std_auc,  std_f1   (std across folds — stability check)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = self._clone_unfitted()
            fold_model.fit(X_train, y_train)

            y_pred  = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)

            auc = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5

            fm = {
                "fold":      fold + 1,
                "train_rows": len(train_idx),
                "test_rows":  len(test_idx),
                "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "auc":       round(auc, 4),
                "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
                "baseline":  round(float(y_test.mean()), 4),  # always-long baseline
                "train_start": str(X_train.index[0].date()),
                "train_end":   str(X_train.index[-1].date()),
                "test_start":  str(X_test.index[0].date()),
                "test_end":    str(X_test.index[-1].date()),
            }
            fold_metrics.append(fm)

        accs = [m["accuracy"] for m in fold_metrics]
        aucs = [m["auc"]      for m in fold_metrics]
        f1s  = [m["f1"]       for m in fold_metrics]

        return {
            "fold_metrics":    fold_metrics,
            "mean_accuracy":   round(float(np.mean(accs)), 4),
            "std_accuracy":    round(float(np.std(accs)), 4),
            "mean_auc":        round(float(np.mean(aucs)), 4),
            "std_auc":         round(float(np.std(aucs)), 4),
            "mean_f1":         round(float(np.mean(f1s)), 4),
            "std_f1":          round(float(np.std(f1s)), 4),
        }

    # ------------------------------------------------------------------
    # Train / test final split evaluation
    # ------------------------------------------------------------------

    def train_test_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_pct: float = 0.80,
    ) -> dict:
        """
        Single train/test split (80/20).
        Trains on first 80%, evaluates on last 20%.
        Returns metrics + fitted model ready for live prediction.
        """
        split = int(len(X) * train_pct)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.fit(X_train, y_train)
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        auc = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5

        return {
            "train_rows":   len(X_train),
            "test_rows":    len(X_test),
            "train_start":  str(X_train.index[0].date()),
            "train_end":    str(X_train.index[-1].date()),
            "test_start":   str(X_test.index[0].date()),
            "test_end":     str(X_test.index[-1].date()),
            "accuracy":     round(accuracy_score(y_test, y_pred), 4),
            "auc":          round(auc, 4),
            "f1":           round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision":    round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":       round(recall_score(y_test, y_pred, zero_division=0), 4),
            "baseline_acc": round(float(y_test.mean()), 4),  # always-predict-up accuracy
            "y_test":       y_test,
            "y_pred":       y_pred,
            "y_proba":      y_proba,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the fitted classifier to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "DirectionClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a DirectionClassifier: {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transform(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        arr = X.values
        if self._scaler is not None:
            if fit:
                arr = self._scaler.fit_transform(arr)
            else:
                arr = self._scaler.transform(arr)
        return arr

    def _clone_unfitted(self) -> "DirectionClassifier":
        """Return a new unfitted classifier with the same hyperparameters."""
        return DirectionClassifier(
            model_type=self.model_type,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_features=self.scale_features,
            random_state=self.random_state,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call .fit() or .train_test_evaluate() first.")
