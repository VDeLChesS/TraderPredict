"""
ema_crossover.py -- EMA fast/slow crossover strategy.

Identical logic to MACrossoverStrategy but uses Exponential Moving Averages
instead of Simple Moving Averages. EMA gives more weight to recent prices,
making it more responsive to trend changes (at the cost of more whipsaws).

Typical parameter sets:
  - EMA 12/26 (classic MACD periods)
  - EMA 9/21  (fast day-trading crossover)
  - EMA 20/50 (comparable to SMA 20/50 baseline)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategies.base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA fast/slow crossover.
    Entry:  fast EMA crosses above slow EMA → signal = 1
    Exit:   fast EMA crosses below slow EMA → signal = 0
    No shorting in default mode.
    """

    def __init__(
        self,
        fast_window: int = 12,
        slow_window: int = 26,
        allow_short: bool = False,
    ):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns pd.Series of {0, 1} (or {-1, 0, 1} if allow_short=True).
        EMA is defined from bar 0, but we zero-out the first slow_window-1 bars
        as warmup to let the EMA stabilise (matching SMA convention).
        No lookahead: ewm uses only past data.
        """
        close = df["close"]
        ema_fast = close.ewm(span=self.fast_window, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_window, adjust=False).mean()

        # Raw signal: 1 when fast > slow, else 0
        raw = (ema_fast > ema_slow).astype(int)

        # Zero out warmup bars (let EMA stabilise)
        raw.iloc[: self.slow_window - 1] = 0

        if self.allow_short:
            raw = raw.replace(0, -1)
            raw.iloc[: self.slow_window - 1] = 0

        return raw.rename("ema_crossover_signal")

    def get_indicator_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df copy with added columns: ema_fast, ema_slow, signal."""
        out = df.copy()
        out["ema_fast"] = df["close"].ewm(span=self.fast_window, adjust=False).mean()
        out["ema_slow"] = df["close"].ewm(span=self.slow_window, adjust=False).mean()
        out["signal"]   = self.generate_signals(df)
        return out

    def plot(self, df: pd.DataFrame, symbol: str, results_dir: str = "results") -> str:
        """
        2-panel chart: price + EMA lines (top), signal bar (bottom).
        Saves PNG to results/{symbol}_ema_crossover.png.
        """
        idf = self.get_indicator_df(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(idf.index, idf["close"],    label="Close",   linewidth=1, color="black")
        ax1.plot(idf.index, idf["ema_fast"],
                 label=f"EMA {self.fast_window}", linewidth=1.2, color="blue", alpha=0.8)
        ax1.plot(idf.index, idf["ema_slow"],
                 label=f"EMA {self.slow_window}", linewidth=1.2, color="orange", alpha=0.8)
        ax1.set_title(f"{symbol} — EMA Crossover (EMA{self.fast_window}/EMA{self.slow_window})")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.fill_between(idf.index, idf["signal"], alpha=0.6, color="green", label="Long signal")
        ax2.set_ylabel("Signal")
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{symbol}_ema_crossover.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
