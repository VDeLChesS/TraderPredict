"""
stop_loss.py -- Stop-loss and take-profit configuration.

Three modes:
  fixed    : constant percentage below/above entry price
  atr      : ATR multiple (adapts to current volatility automatically)
  trailing : price-tracking stop that locks in gains

Returns dicts of parameters passed directly to vbt.Portfolio.from_signals().
vectorbt handles the actual stop execution on each bar -- no manual loops.
"""

import pandas as pd
import ta


class StopLossConfig:
    """Static methods -- no state, call directly."""

    @staticmethod
    def fixed(
        sl_pct: float = 0.05,
        tp_pct: float = 0.10,
        trailing: bool = False,
    ) -> dict:
        """
        Fixed percentage stop-loss and take-profit.

        sl_pct : fraction below entry price to trigger stop (0.05 = -5%)
        tp_pct : fraction above entry price to take profit (0.10 = +10%)
        trailing : if True, stop moves up with price (locks in gains)
        """
        params = {}
        if sl_pct is not None and sl_pct > 0:
            params["sl_stop"] = sl_pct
        if tp_pct is not None and tp_pct > 0:
            params["tp_stop"] = tp_pct
        if trailing:
            params["sl_trail"] = True
        return params

    @staticmethod
    def atr_based(
        df: pd.DataFrame,
        atr_window: int = 14,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
        trailing: bool = False,
    ) -> dict:
        """
        Stop-loss and take-profit as multiples of ATR.

        Adapts to regime: widens in volatile markets, tightens in calm.
        Returns per-bar stop levels as arrays (vectorbt broadcasts them).

        sl_atr_mult : e.g. 2.0 = stop at 2x ATR below entry
        tp_atr_mult : e.g. 3.0 = take profit at 3x ATR above entry
        """
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=atr_window,
        ).average_true_range()

        # Express as fraction of price (what vectorbt expects)
        atr_pct = (atr / df["close"]).fillna(0.02)

        params = {}
        if sl_atr_mult is not None and sl_atr_mult > 0:
            params["sl_stop"] = (atr_pct * sl_atr_mult).values
        if tp_atr_mult is not None and tp_atr_mult > 0:
            params["tp_stop"] = (atr_pct * tp_atr_mult).values
        if trailing:
            params["sl_trail"] = True
        return params

    @staticmethod
    def trailing(sl_pct: float = 0.05) -> dict:
        """
        Trailing stop-loss only (no fixed take-profit).
        The stop moves up with price, locking in gains.
        Never moves down -- only tightens.
        """
        return {"sl_stop": sl_pct, "sl_trail": True}
