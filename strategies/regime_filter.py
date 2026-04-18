"""
regime_filter.py -- Volatility regime filter wrapper.

Wraps any BaseStrategy and suppresses its signals when the volatility regime
is unfavourable (e.g. high vol crash periods). The filter uses a rolling
percentile of realised volatility — an entirely past-data computation, so
no lookahead.

Three regime modes:
  - "low_vol_only"  : trade only when vol percentile <= upper_threshold
  - "high_vol_only" : trade only when vol percentile >= lower_threshold
  - "exclude_extreme" : trade only when vol is in [lower, upper] band
                       (skips both the calmest and the most violent regimes)

Why filter?
  Most rule-based strategies underperform during volatility regime shifts.
  Cutting trades during the worst regime improves Sharpe and reduces drawdown,
  even though it forfeits some upside.
"""

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class RegimeFilteredStrategy(BaseStrategy):
    """
    Wrap a base strategy so it only trades within an allowed volatility regime.

    Parameters
    ----------
    base_strategy : BaseStrategy
        Any strategy whose signals will be filtered.
    vol_window : int
        Rolling window for realised volatility (default 20 bars).
    percentile_window : int
        Lookback for the rolling percentile rank (default 252 bars = 1 year).
    mode : "low_vol_only" | "high_vol_only" | "exclude_extreme"
    lower_threshold : float in [0, 1]
        Lower percentile bound (used by high_vol_only and exclude_extreme).
    upper_threshold : float in [0, 1]
        Upper percentile bound (used by low_vol_only and exclude_extreme).
    """

    VALID_MODES = ("low_vol_only", "high_vol_only", "exclude_extreme")

    def __init__(
        self,
        base_strategy: BaseStrategy,
        vol_window: int = 20,
        percentile_window: int = 252,
        mode: str = "low_vol_only",
        lower_threshold: float = 0.20,
        upper_threshold: float = 0.80,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= lower < upper <= 1"
            )

        self.base_strategy   = base_strategy
        self.vol_window      = vol_window
        self.percentile_window = percentile_window
        self.mode            = mode
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute base signals, then mask out bars where the volatility regime
        is disallowed. Returns pd.Series of {0, 1} aligned to df.index.
        """
        base_sig = self.base_strategy.generate_signals(df)
        regime_ok = self._compute_regime_mask(df)

        # Mask: where regime_ok is False, force flat (0)
        filtered = base_sig.where(regime_ok, 0).astype(int)
        return filtered.rename(f"regime_filtered_{self.mode}")

    def get_regime_mask(self, df: pd.DataFrame) -> pd.Series:
        """Public helper: return the boolean regime mask used by this filter."""
        return self._compute_regime_mask(df)

    def get_vol_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Public helper: return the rolling vol percentile (for plotting)."""
        ret = np.log(df["close"] / df["close"].shift(1))
        vol = ret.rolling(self.vol_window).std() * np.sqrt(252)
        return self._rolling_percentile(vol)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_regime_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute boolean Series: True where current bar is in an ALLOWED
        volatility regime.  Uses past data only.
        """
        # Realised volatility (log returns, annualised)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol = log_ret.rolling(self.vol_window).std() * np.sqrt(252)

        # Rolling percentile rank of current vol vs past N bars
        pct = self._rolling_percentile(vol)

        if self.mode == "low_vol_only":
            mask = pct <= self.upper_threshold
        elif self.mode == "high_vol_only":
            mask = pct >= self.lower_threshold
        else:  # exclude_extreme
            mask = (pct >= self.lower_threshold) & (pct <= self.upper_threshold)

        # Warmup bars (NaN percentile) -> default to disallowed (False)
        return mask.fillna(False)

    def _rolling_percentile(self, series: pd.Series) -> pd.Series:
        """
        For each bar, compute the percentile rank of the current value
        within the trailing `percentile_window` values (excluding the current bar).
        Returns NaN where insufficient history.
        """
        return series.rolling(
            self.percentile_window, min_periods=30
        ).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
