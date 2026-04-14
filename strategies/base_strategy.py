from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """
    All strategies produce a signals Series aligned to the input DataFrame's index.
    Signal values: +1 (long), 0 (flat/no position).
    Convention: signal on bar N means enter at bar N+1 open (shift applied in engine).
    No lookahead: indicators computed only on data available at each bar.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Parameters
        ----------
        df : OHLCV DataFrame with DatetimeIndex (from DataLoader.load()).
             Must contain at minimum: ['open', 'high', 'low', 'close', 'volume']

        Returns
        -------
        pd.Series of int {0, 1}, same index as df, no NaN values.
        """

    def compute_returns(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        No-fee daily P&L for quick sanity checks.
        return[t] = signal[t-1] * close_pct_change[t]
        """
        shifted    = signals.shift(1).fillna(0)
        daily_ret  = df["close"].pct_change().fillna(0)
        return (shifted * daily_ret).rename("strategy_returns")

    def summary(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """
        Quick performance summary (no fees — use BacktestEngine for proper results).
        Returns dict with: total_return, annualized_return, win_rate, num_trades, max_drawdown.
        """
        strat_ret   = self.compute_returns(df, signals)
        equity      = (1 + strat_ret).cumprod()

        total_ret   = float(equity.iloc[-1] - 1)
        n_years     = len(equity) / 252
        ann_ret     = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        nonzero     = strat_ret[strat_ret != 0]
        win_rate    = float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.0

        # Count trades: number of times signal changes from 0→1 or 1→0
        num_trades  = int((signals.diff().fillna(0) != 0).sum())

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())

        return {
            "total_return":      round(total_ret, 4),
            "annualized_return": round(ann_ret, 4),
            "win_rate":          round(win_rate, 4),
            "num_trades":        num_trades,
            "max_drawdown":      round(max_dd, 4),
        }
