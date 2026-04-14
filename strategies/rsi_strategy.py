import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion strategy.
    State machine:
      - RSI drops below oversold  → enter long (signal = 1)
      - RSI rises above overbought → exit (signal = 0)
      - Otherwise: hold (carry forward previous state)
    No shorting.
    """

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        self.rsi_window = rsi_window
        self.oversold   = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns pd.Series of {0, 1}, same index as df, no NaN.
        Warmup bars (RSI not yet computed) default to flat (0).
        Vectorised: no row-by-row Python loop.
        """
        rsi = ta.momentum.RSIIndicator(
            close=df["close"], window=self.rsi_window
        ).rsi()

        # Mark state-change events; NaN between events is forward-filled.
        # enter takes priority over exit (written last) — impossible overlap in practice.
        events = pd.Series(np.nan, index=df.index, dtype=float)
        events[rsi > self.overbought] = 0.0  # exit first
        events[rsi < self.oversold]   = 1.0  # enter takes priority

        return events.ffill().fillna(0).astype(int).rename("rsi_signal")

    def get_indicator_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df copy with added columns: rsi, signal."""
        out = df.copy()
        out["rsi"]    = ta.momentum.RSIIndicator(
            close=df["close"], window=self.rsi_window
        ).rsi()
        out["signal"] = self.generate_signals(df)
        return out

    def plot(self, df: pd.DataFrame, symbol: str, results_dir: str = "results") -> str:
        """
        2-panel chart: price close (top), RSI with oversold/overbought bands (bottom).
        Saves PNG to results/{symbol}_rsi.png.
        Returns the saved file path.
        """
        idf = self.get_indicator_df(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2]})

        ax1.plot(idf.index, idf["close"], label="Close", linewidth=1, color="black")
        # Shade long periods
        long_mask = idf["signal"] == 1
        ax1.fill_between(idf.index, idf["close"].min(), idf["close"].max(),
                         where=long_mask, alpha=0.15, color="green", label="Long")
        ax1.set_title(f"{symbol} — RSI Strategy (window={self.rsi_window}, "
                      f"oversold={self.oversold}, overbought={self.overbought})")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(idf.index, idf["rsi"], label=f"RSI({self.rsi_window})",
                 linewidth=1, color="purple")
        ax2.axhline(self.oversold,   color="green", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(self.overbought, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(50, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
        ax2.fill_between(idf.index, idf["rsi"], self.oversold,
                         where=(idf["rsi"] < self.oversold), alpha=0.3, color="green")
        ax2.fill_between(idf.index, idf["rsi"], self.overbought,
                         where=(idf["rsi"] > self.overbought), alpha=0.3, color="red")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{symbol}_rsi.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
