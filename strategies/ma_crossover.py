import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategies.base_strategy import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """
    SMA fast/slow crossover.
    Entry:  fast SMA crosses above slow SMA → signal = 1
    Exit:   fast SMA crosses below slow SMA → signal = 0
    No shorting in default mode.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
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
        First slow_window-1 bars are 0 (warmup — no signal until both SMAs are valid).
        No lookahead: SMAs use only past data via rolling().mean().
        """
        close = df["close"]
        sma_fast = close.rolling(self.fast_window).mean()
        sma_slow = close.rolling(self.slow_window).mean()

        # Raw signal: 1 when fast > slow, else 0
        raw = (sma_fast > sma_slow).astype(int)

        # Zero out warmup bars (where slow SMA is not yet defined)
        raw.iloc[: self.slow_window - 1] = 0

        if self.allow_short:
            raw = raw.replace(0, -1)
            raw.iloc[: self.slow_window - 1] = 0  # still no signal in warmup

        return raw.rename("ma_crossover_signal")

    def get_indicator_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df copy with added columns: sma_fast, sma_slow, signal."""
        out = df.copy()
        out["sma_fast"] = df["close"].rolling(self.fast_window).mean()
        out["sma_slow"] = df["close"].rolling(self.slow_window).mean()
        out["signal"]   = self.generate_signals(df)
        return out

    def plot(self, df: pd.DataFrame, symbol: str, results_dir: str = "results") -> str:
        """
        2-panel chart: price + SMA lines (top), signal bar (bottom).
        Saves PNG to results/{symbol}_ma_crossover.png.
        Returns the saved file path.
        """
        idf = self.get_indicator_df(df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(idf.index, idf["close"],   label="Close",   linewidth=1, color="black")
        ax1.plot(idf.index, idf["sma_fast"],
                 label=f"SMA {self.fast_window}", linewidth=1.2, color="blue", alpha=0.8)
        ax1.plot(idf.index, idf["sma_slow"],
                 label=f"SMA {self.slow_window}", linewidth=1.2, color="orange", alpha=0.8)
        ax1.set_title(f"{symbol} — MA Crossover (SMA{self.fast_window}/SMA{self.slow_window})")
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
        out_path = os.path.join(results_dir, f"{symbol}_ma_crossover.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
