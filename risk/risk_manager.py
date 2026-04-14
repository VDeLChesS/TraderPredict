"""
risk_manager.py -- Orchestrator combining position sizing, stops, and circuit breaker.

Usage:
    rm = RiskManager(
        position_mode="vol_scaled",
        stop_mode="atr",
        use_circuit_breaker=True,
        max_drawdown_pct=0.20,
    )
    result = engine.run_backtest(df, signals, symbol, risk_manager=rm)

The RiskManager is optional.  BacktestEngine works without it (all-in, no stops,
no breaker) -- exactly the same behaviour as before Phase 7.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossConfig
from risk.circuit_breaker import CircuitBreaker


class RiskManager:
    """
    Combine position sizing + stop-loss + circuit breaker into one object.

    Parameters
    ----------
    position_mode : 'fixed' | 'vol_scaled' | 'atr_scaled'
    position_fraction : base fraction for fixed mode (default 1.0 = all-in)
    vol_lookback : rolling window for vol_scaled mode
    target_risk_pct : daily risk budget for vol/atr sizing
    atr_window : ATR lookback for atr_scaled sizing and atr stops

    stop_mode : 'none' | 'fixed' | 'atr' | 'trailing'
    sl_pct : stop-loss fraction for fixed mode
    tp_pct : take-profit fraction for fixed mode
    sl_atr_mult : ATR multiplier for stop-loss in atr mode
    tp_atr_mult : ATR multiplier for take-profit in atr mode
    trailing : enable trailing stop (works with fixed or atr mode)

    use_circuit_breaker : enable max-drawdown circuit breaker
    max_drawdown_pct : trigger threshold (0.20 = -20%)
    cooldown_bars : bars to stay flat after trigger
    recovery_pct : drawdown must recover above this to resume
    """

    POSITION_MODES = ("fixed", "vol_scaled", "atr_scaled")
    STOP_MODES = ("none", "fixed", "atr", "trailing")

    def __init__(
        self,
        # Position sizing
        position_mode: str = "fixed",
        position_fraction: float = 1.0,
        vol_lookback: int = 20,
        target_risk_pct: float = 0.02,
        atr_window: int = 14,
        # Stop-loss / take-profit
        stop_mode: str = "none",
        sl_pct: float = 0.05,
        tp_pct: float = 0.10,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
        trailing: bool = False,
        # Circuit breaker
        use_circuit_breaker: bool = False,
        max_drawdown_pct: float = 0.20,
        cooldown_bars: int = 10,
        recovery_pct: float = None,
    ):
        if position_mode not in self.POSITION_MODES:
            raise ValueError(f"position_mode must be one of {self.POSITION_MODES}")
        if stop_mode not in self.STOP_MODES:
            raise ValueError(f"stop_mode must be one of {self.STOP_MODES}")

        self.position_mode = position_mode
        self.position_fraction = position_fraction
        self.vol_lookback = vol_lookback
        self.target_risk_pct = target_risk_pct
        self.atr_window = atr_window

        self.stop_mode = stop_mode
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trailing = trailing

        self.use_circuit_breaker = use_circuit_breaker
        self._circuit_breaker = (
            CircuitBreaker(max_drawdown_pct, cooldown_bars, recovery_pct)
            if use_circuit_breaker
            else None
        )

    # ------------------------------------------------------------------
    # Core: apply all risk controls
    # ------------------------------------------------------------------

    def apply(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """
        Apply all risk controls to the raw signals.

        Returns
        -------
        dict with keys:
          signals      : pd.Series -- modified (circuit breaker may zero some bars)
          size         : pd.Series or None -- per-bar position sizes
          sl_stop      : float, np.ndarray, or None
          tp_stop      : float, np.ndarray, or None
          sl_trail     : bool
          breaker_info : dict or None
        """
        result = {
            "signals": signals,
            "size": None,
            "sl_stop": None,
            "tp_stop": None,
            "sl_trail": False,
            "breaker_info": None,
        }

        # 1. Circuit breaker (modifies signals before anything else)
        if self._circuit_breaker is not None:
            modified, breaker_info = self._circuit_breaker.apply(df, signals)
            result["signals"] = modified
            result["breaker_info"] = breaker_info

        # 2. Position sizing
        result["size"] = self._compute_size(df, result["signals"])

        # 3. Stop-loss / take-profit
        stop_params = self._compute_stops(df)
        result["sl_stop"] = stop_params.get("sl_stop")
        result["tp_stop"] = stop_params.get("tp_stop")
        result["sl_trail"] = stop_params.get("sl_trail", False)

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable summary of the risk configuration."""
        parts = []

        # Position sizing
        if self.position_mode == "fixed":
            parts.append(f"Position sizing: fixed {self.position_fraction:.0%}")
        elif self.position_mode == "vol_scaled":
            parts.append(
                f"Position sizing: vol-scaled "
                f"(target_risk={self.target_risk_pct:.1%}, lookback={self.vol_lookback})"
            )
        elif self.position_mode == "atr_scaled":
            parts.append(
                f"Position sizing: ATR-scaled "
                f"(risk={self.target_risk_pct:.1%}, window={self.atr_window})"
            )

        # Stops
        if self.stop_mode == "none":
            parts.append("Stops: none")
        elif self.stop_mode == "fixed":
            trail_str = " (trailing)" if self.trailing else ""
            parts.append(f"Stops: fixed SL={self.sl_pct:.1%}, TP={self.tp_pct:.1%}{trail_str}")
        elif self.stop_mode == "atr":
            trail_str = " (trailing)" if self.trailing else ""
            parts.append(
                f"Stops: ATR-based SL={self.sl_atr_mult}x, "
                f"TP={self.tp_atr_mult}x{trail_str}"
            )
        elif self.stop_mode == "trailing":
            parts.append(f"Stops: trailing SL={self.sl_pct:.1%}")

        # Circuit breaker
        if self._circuit_breaker is not None:
            cb = self._circuit_breaker
            parts.append(
                f"Circuit breaker: max_dd={cb.max_drawdown_pct:.0%}, "
                f"cooldown={cb.cooldown_bars} bars, "
                f"recovery={cb.recovery_pct:.0%}"
            )
        else:
            parts.append("Circuit breaker: off")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Diagnostics plot
    # ------------------------------------------------------------------

    def plot_risk_dashboard(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        results_dir: str = "results",
    ) -> str:
        """
        3-panel chart:
          Top:    Close price with blocked periods shaded red
          Middle: Approximate drawdown with threshold line
          Bottom: Position sizes over time

        Returns path to saved PNG.
        """
        risk_params = self.apply(df, signals)
        modified_signals = risk_params["signals"]
        size = risk_params["size"]

        # Recompute drawdown for plotting
        close_ret = df["close"].pct_change().fillna(0)
        strat_ret = signals.shift(1).fillna(0) * close_ret
        equity = (1 + strat_ret).cumprod()
        drawdown = (equity - equity.cummax()) / equity.cummax()

        blocked = (signals != 0) & (modified_signals == 0)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Price + blocked periods
        ax0 = axes[0]
        ax0.plot(df.index, df["close"], color="black", linewidth=1, label="Close")
        if blocked.any():
            ax0.fill_between(
                df.index, df["close"].min(), df["close"].max(),
                where=blocked, alpha=0.2, color="red", label="Blocked (circuit breaker)",
            )
        ax0.set_title(f"{symbol} -- Risk Management Dashboard")
        ax0.set_ylabel("Price")
        ax0.legend(loc="upper left", fontsize=8)
        ax0.grid(alpha=0.3)

        # Panel 2: Drawdown
        ax1 = axes[1]
        ax1.fill_between(df.index, drawdown, 0, alpha=0.4, color="#F44336")
        ax1.plot(df.index, drawdown, color="#F44336", linewidth=0.8)
        if self._circuit_breaker is not None:
            ax1.axhline(
                -self._circuit_breaker.max_drawdown_pct,
                color="red", linestyle="--", linewidth=1.2,
                label=f"Breaker threshold ({self._circuit_breaker.max_drawdown_pct:.0%})",
            )
            ax1.axhline(
                -self._circuit_breaker.recovery_pct,
                color="green", linestyle="--", linewidth=1.0, alpha=0.7,
                label=f"Recovery level ({self._circuit_breaker.recovery_pct:.0%})",
            )
        ax1.set_ylabel("Drawdown")
        ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(alpha=0.3)

        # Panel 3: Position sizes
        ax2 = axes[2]
        if size is not None:
            ax2.fill_between(df.index, size, 0, alpha=0.5, color="#2196F3")
            ax2.plot(df.index, size, color="#2196F3", linewidth=0.8)
        ax2.set_ylabel("Position Size")
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{symbol}_risk_dashboard.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_size(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Dispatch to the appropriate PositionSizer method."""
        if self.position_mode == "vol_scaled":
            return PositionSizer.volatility_scaled(
                df,
                lookback=self.vol_lookback,
                target_risk_pct=self.target_risk_pct,
            )
        if self.position_mode == "atr_scaled":
            return PositionSizer.atr_scaled(
                df,
                atr_window=self.atr_window,
                risk_per_trade=self.target_risk_pct,
            )
        return PositionSizer.fixed(signals, self.position_fraction)

    def _compute_stops(self, df: pd.DataFrame) -> dict:
        """Dispatch to the appropriate StopLossConfig method."""
        if self.stop_mode == "fixed":
            return StopLossConfig.fixed(self.sl_pct, self.tp_pct, self.trailing)
        if self.stop_mode == "atr":
            return StopLossConfig.atr_based(
                df,
                atr_window=self.atr_window,
                sl_atr_mult=self.sl_atr_mult,
                tp_atr_mult=self.tp_atr_mult,
                trailing=self.trailing,
            )
        if self.stop_mode == "trailing":
            return StopLossConfig.trailing(self.sl_pct)
        return {}


# ------------------------------------------------------------------
# Text report
# ------------------------------------------------------------------

def print_risk_report(
    risk_params: dict,
    risk_manager: RiskManager,
    symbol: str,
) -> None:
    """Print a summary of risk management results to the console."""
    sep = "=" * 60
    dash = "-" * 60
    print("")
    print(sep)
    print(f"  {symbol}  |  Risk Management Report")
    print(sep)
    print(f"  Config: {risk_manager.describe()}")
    print(dash)

    # Position sizing stats
    size = risk_params.get("size")
    if size is not None and hasattr(size, "mean"):
        print(f"  Position size  -- mean: {size.mean():.3f}  "
              f"min: {size.min():.3f}  max: {size.max():.3f}")

    # Stop-loss info
    sl = risk_params.get("sl_stop")
    tp = risk_params.get("tp_stop")
    trail = risk_params.get("sl_trail", False)
    if sl is not None:
        if hasattr(sl, "__len__"):
            print(f"  Stop-loss      -- mean: {float(pd.Series(sl).mean()):.3f}  "
                  f"(per-bar, ATR-based)")
        else:
            print(f"  Stop-loss      -- {sl:.1%} fixed")
    if tp is not None:
        if hasattr(tp, "__len__"):
            print(f"  Take-profit    -- mean: {float(pd.Series(tp).mean()):.3f}  "
                  f"(per-bar, ATR-based)")
        else:
            print(f"  Take-profit    -- {tp:.1%} fixed")
    if trail:
        print(f"  Trailing stop  -- enabled")

    # Circuit breaker
    breaker_info = risk_params.get("breaker_info")
    if breaker_info is not None:
        print(dash)
        print(f"  Circuit Breaker Results:")
        print(f"    Trigger events:     {breaker_info['num_trigger_events']}")
        print(f"    Bars in drawdown:   {breaker_info['breaker_triggered_bars']}")
        print(f"    Bars blocked:       {breaker_info['total_blocked_bars']}")
        print(f"    Approx max DD:      {breaker_info['approx_max_drawdown']:.2%}")
    print(sep)
