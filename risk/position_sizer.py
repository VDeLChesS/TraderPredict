"""
position_sizer.py -- Compute per-bar position size as a fraction of portfolio.

Three modes:
  fixed          : constant fraction (1.0 = all-in, 0.5 = half position)
  volatility_scaled : inversely proportional to recent realised volatility
  atr_scaled     : inversely proportional to ATR (accounts for high/low range)
  kelly          : optimal fraction from historical win/loss statistics

All methods return values in [0.0, 1.0] (fraction of portfolio to deploy).
"""

import numpy as np
import pandas as pd
import ta


class PositionSizer:
    """Static methods -- no state, call directly."""

    @staticmethod
    def fixed(signals: pd.Series, fraction: float = 1.0) -> pd.Series:
        """
        Every bar uses the same fraction.
        fraction=1.0 is the default (full position, same as no sizing).
        """
        return pd.Series(fraction, index=signals.index, name="position_size")

    @staticmethod
    def volatility_scaled(
        df: pd.DataFrame,
        lookback: int = 20,
        target_risk_pct: float = 0.02,
        min_size: float = 0.1,
        max_size: float = 1.0,
    ) -> pd.Series:
        """
        Size = target_risk / realised_volatility.

        In calm markets (low vol) -> larger positions.
        In volatile markets (high vol) -> smaller positions.

        target_risk_pct: daily risk budget as fraction of portfolio (0.02 = 2%).
        Capped to [min_size, max_size] to prevent extreme allocations.
        """
        daily_vol = df["close"].pct_change().rolling(lookback).std()
        # Avoid division by zero: clamp vol to a sensible floor
        daily_vol = daily_vol.clip(lower=1e-6)
        raw_size = target_risk_pct / daily_vol
        return raw_size.clip(min_size, max_size).fillna(min_size).rename("position_size")

    @staticmethod
    def atr_scaled(
        df: pd.DataFrame,
        atr_window: int = 14,
        risk_per_trade: float = 0.02,
        min_size: float = 0.1,
        max_size: float = 1.0,
    ) -> pd.Series:
        """
        Size based on ATR: size = risk_per_trade / (ATR / close).

        ATR captures intraday range (high-low), so it reacts faster
        to volatility spikes than close-to-close std.
        """
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=atr_window,
        ).average_true_range()

        atr_pct = (atr / df["close"]).clip(lower=1e-6)
        raw_size = risk_per_trade / atr_pct
        return raw_size.clip(min_size, max_size).fillna(min_size).rename("position_size")

    @staticmethod
    def kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
    ) -> float:
        """
        Kelly criterion -- theoretically optimal bet size.

        Full Kelly is aggressive and assumes perfect parameter estimates.
        Half-Kelly (fraction=0.5) is standard practice: sacrifices ~25% of
        growth rate but cuts variance roughly in half.

        Formula: K = W - (1-W)/R
          where W = win_rate, R = avg_win / avg_loss

        Returns a scalar fraction in [0.0, 1.0].
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        R = avg_win / avg_loss
        full_kelly = win_rate - (1.0 - win_rate) / R
        return max(0.0, min(1.0, full_kelly * fraction))
