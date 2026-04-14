"""
engine.py — Backtesting engine wrapping vectorbt.
Handles: fees, slippage, walk-forward split, strategy comparison, equity curve plots.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt

from backtesting.metrics import compute_all_metrics
from strategies.base_strategy import BaseStrategy

warnings.filterwarnings("ignore", category=FutureWarning)


class BacktestEngine:
    """
    Proper backtesting with realistic costs.
    All positions are long-only (signal=1) or flat (signal=0).
    Signal shift convention: signal on bar N → trade executes at bar N+1 open
    (implemented via shift before computing entries/exits).
    """

    def __init__(
        self,
        init_cash: float = 10_000.0,
        fees: float = 0.001,       # 0.1% per trade one-way
        slippage: float = 0.0005,  # 0.05% price impact per trade
        freq: str = "D",
    ):
        self.init_cash = init_cash
        self.fees      = fees
        self.slippage  = slippage
        self.freq      = freq

    # ------------------------------------------------------------------
    # Core: single full-period backtest
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "ASSET",
        risk_manager=None,
    ) -> dict:
        """
        Run a single backtest.

        Parameters
        ----------
        df           : OHLCV DataFrame (DatetimeIndex, daily).
        signals      : pd.Series of {0, 1} aligned to df.index.
        symbol       : label only.
        risk_manager : optional RiskManager instance. If provided, applies
                       position sizing, stop-loss/take-profit, and circuit breaker.
                       If None, engine behaves exactly as before (full backward compat).

        Returns
        -------
        dict with keys:
          metrics       : dict from compute_all_metrics()
          equity_curve  : pd.Series  (daily portfolio value)
          returns       : pd.Series  (daily portfolio returns)
          trade_log     : pd.DataFrame  (per-trade records)
          vbt_portfolio : vbt.Portfolio object
          risk_params   : dict or None (risk management details, if applied)
        """
        close = df["close"].copy()

        # Base vectorbt params
        vbt_kwargs = {
            "init_cash": self.init_cash,
            "fees":      self.fees,
            "slippage":  self.slippage,
            "freq":      self.freq,
            "log":       False,
        }

        # Apply risk management if provided
        risk_params = None
        if risk_manager is not None:
            risk_params = risk_manager.apply(df, signals)
            signals = risk_params["signals"]

            # Stop-loss / take-profit
            if risk_params.get("sl_stop") is not None:
                vbt_kwargs["sl_stop"] = risk_params["sl_stop"]
            if risk_params.get("tp_stop") is not None:
                vbt_kwargs["tp_stop"] = risk_params["tp_stop"]
            if risk_params.get("sl_trail"):
                vbt_kwargs["sl_trail"] = True

            # Position sizing
            size = risk_params.get("size")
            if size is not None:
                vbt_kwargs["size"] = size
                vbt_kwargs["size_type"] = "percent"

        # Shift signals by 1: signal on bar N executes at bar N+1
        sig_shifted = signals.shift(1).fillna(0).astype(int)

        # Rising edge -> entry; falling edge -> exit
        sig_prev = sig_shifted.shift(1).fillna(0).astype(int)
        entries  = (sig_shifted == 1) & (sig_prev != 1)
        exits    = (sig_shifted == 0) & (sig_prev != 0)

        # Run vectorbt portfolio
        portfolio = vbt.Portfolio.from_signals(
            close, entries, exits, **vbt_kwargs,
        )

        equity_curve = portfolio.value()
        port_returns = portfolio.returns()

        # Trade log
        try:
            trade_log = portfolio.trades.records_readable
        except Exception:
            trade_log = pd.DataFrame()

        metrics = compute_all_metrics(equity_curve, port_returns)
        metrics["num_trades"] = int(entries.sum())

        return {
            "metrics":       metrics,
            "equity_curve":  equity_curve,
            "returns":       port_returns,
            "trade_log":     trade_log,
            "vbt_portfolio": portfolio,
            "risk_params":   risk_params,
        }

    # ------------------------------------------------------------------
    # Walk-forward: train/test split
    # ------------------------------------------------------------------

    def run_walkforward(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        train_pct: float = 0.70,
        symbol: str = "ASSET",
    ) -> dict:
        """
        Simple train/test walk-forward split.

        Steps:
          1. Generate signals on train portion → in-sample backtest
          2. Generate signals on test portion  → out-of-sample backtest
          3. Full-period backtest for reference

        Returns dict with keys: train, test, full, split_date
        Key question answered: does the strategy degrade on unseen data?
        """
        split_idx = int(len(df) * train_pct)
        split_date = str(df.index[split_idx].date())

        # Generate signals once on the full df: avoids triple computation and
        # ensures the test slice retains full warmup history from training bars.
        full_signals  = strategy.generate_signals(df)
        train_signals = full_signals.iloc[:split_idx]
        test_signals  = full_signals.iloc[split_idx:]

        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

        train_result = self.run_backtest(train_df, train_signals, symbol + "_train")
        test_result  = self.run_backtest(test_df,  test_signals,  symbol + "_test")
        full_result  = self.run_backtest(df,        full_signals,  symbol + "_full")

        return {
            "train":      train_result,
            "test":       test_result,
            "full":       full_result,
            "split_date": split_date,
        }

    # ------------------------------------------------------------------
    # Compare multiple strategies side-by-side
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: dict,
        symbol: str = "ASSET",
    ) -> pd.DataFrame:
        """
        Run run_backtest() for each strategy in the dict.
        Returns DataFrame: rows = strategy names, columns = metric keys.
        Also adds a buy-and-hold benchmark row.
        """
        rows = []

        # Buy-and-hold benchmark
        bh_equity  = pd.Series(
            self.init_cash * (df["close"] / df["close"].iloc[0]),
            index=df.index,
        )
        bh_returns = df["close"].pct_change().fillna(0)
        bh_metrics = compute_all_metrics(bh_equity, bh_returns)
        bh_metrics["num_trades"] = 1
        rows.append({"strategy": "Buy & Hold", **bh_metrics})

        for name, strat in strategies.items():
            signals = strat.generate_signals(df)
            result  = self.run_backtest(df, signals, symbol)
            rows.append({"strategy": name, **result["metrics"]})

        df_out = pd.DataFrame(rows).set_index("strategy")
        return df_out

    # ------------------------------------------------------------------
    # Equity curve plot
    # ------------------------------------------------------------------

    def plot_equity_curves(
        self,
        equity_curves: dict,
        symbol: str,
        benchmark_df: pd.DataFrame = None,
        save_path: str = None,
    ) -> str:
        """
        Plot multiple equity curves on one chart.
        Adds buy-and-hold as a dashed benchmark if benchmark_df is provided.
        Saves PNG if save_path is given.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Normalise all curves to start at 1.0 for easy comparison
        for name, curve in equity_curves.items():
            normalised = curve / curve.iloc[0]
            ax.plot(curve.index, normalised, label=name, linewidth=1.5)

        if benchmark_df is not None:
            bh = benchmark_df["close"] / benchmark_df["close"].iloc[0]
            ax.plot(bh.index, bh, label="Buy & Hold", linewidth=1,
                    linestyle="--", color="grey", alpha=0.7)

        ax.set_title(f"{symbol} — Strategy Equity Curves (normalised, fees+slippage included)")
        ax.set_ylabel("Portfolio value (normalised to 1.0)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=100)
        plt.close(fig)
        return save_path or ""
