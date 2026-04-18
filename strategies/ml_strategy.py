"""
ml_strategy.py -- Wraps a trained DirectionClassifier as a BaseStrategy.

Flow:
  df (OHLCV) -> FeatureEngineer.build_features() -> DirectionClassifier.predict_proba()
             -> threshold -> signal {0, 1}

No-lookahead guarantee:
  - build_features() only uses past data at each bar (rolling indicators)
  - The model was trained on a separate training set
  - Target columns are never passed to the model
  - The signal is shifted by 1 inside BacktestEngine before trade execution

Threshold effect:
  - Low threshold (0.50) = trade on every mild up signal  (high activity)
  - High threshold (0.60) = only trade high-confidence predictions (low activity, higher precision)
"""

import pandas as pd

from features.feature_engineer import FeatureEngineer
from models.direction_classifier import DirectionClassifier
from strategies.base_strategy import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    Trade signal derived from ML model probability.

    Entry:  P(up) >= long_threshold  -> signal = 1
    Exit:   P(up) <  exit_threshold  -> signal = 0
    Hold:   otherwise carry last signal (hysteresis avoids excessive churn)
    """

    def __init__(
        self,
        classifier: DirectionClassifier,
        long_threshold: float = 0.55,
        exit_threshold: float = 0.50,
        use_hysteresis: bool = True,
        multi_timeframe: bool = False,
    ):
        """
        Parameters
        ----------
        classifier      : Fitted DirectionClassifier (call .train_test_evaluate() first).
        long_threshold  : Enter long when P(up) >= this value.
        exit_threshold  : Exit when P(up) drops below this value.
                          Must be <= long_threshold.
        use_hysteresis  : If True, hold position between thresholds (reduces whipsaw).
                          If False, binary: signal = (proba >= long_threshold).
        multi_timeframe : Set True if the underlying classifier was trained
                          with multi-timeframe (weekly) features. Must match
                          the training setup or feature columns will mismatch.
        """
        if exit_threshold > long_threshold:
            raise ValueError("exit_threshold must be <= long_threshold")

        self.classifier      = classifier
        self.long_threshold  = long_threshold
        self.exit_threshold  = exit_threshold
        self.use_hysteresis  = use_hysteresis
        self.multi_timeframe = multi_timeframe
        self._fe             = FeatureEngineer(
            drop_warmup=False,
            multi_timeframe=multi_timeframe,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Build features -> get probabilities -> apply threshold.
        Returns pd.Series of {0, 1}, same index as df.
        Warmup rows (where features are NaN) default to flat (0).
        Vectorised: no row-by-row Python loop.
        """
        X, warmup_mask = self._prepare_features(df)
        proba = self.classifier.predict_proba(X)

        if self.use_hysteresis:
            # Forward-fill state machine: mark entry/exit events, ffill between them.
            # exit written first so enter takes priority if both fire (impossible in practice).
            events = pd.Series(float("nan"), index=df.index)
            events[proba <  self.exit_threshold]  = 0.0
            events[proba >= self.long_threshold]  = 1.0
            signal = events.ffill().fillna(0).astype(int)
        else:
            signal = (proba >= self.long_threshold).astype(int)

        signal[warmup_mask] = 0
        return signal.rename("ml_signal")

    def get_probabilities(self, df: pd.DataFrame) -> pd.Series:
        """Return raw P(up) probabilities (for debugging and calibration analysis)."""
        X, _ = self._prepare_features(df)
        return self.classifier.predict_proba(X)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame):
        """Build feature matrix from OHLCV df. Returns (X_filled, warmup_mask)."""
        df_feat = self._fe.build_features(df)

        # Use the classifier's stored feature_names_ if available — guarantees
        # exact column order and number match between training and inference.
        if getattr(self.classifier, "feature_names_", None):
            feat_cols = list(self.classifier.feature_names_)
            missing = [c for c in feat_cols if c not in df_feat.columns]
            if missing:
                raise ValueError(
                    f"MLStrategy: classifier expects features not produced by "
                    f"the FeatureEngineer (multi_timeframe={self.multi_timeframe}). "
                    f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
        else:
            feat_cols = self._fe.get_feature_names(df_feat)

        X           = df_feat[feat_cols].reindex(df.index)
        warmup_mask = X.isna().any(axis=1)
        return X.fillna(0), warmup_mask
