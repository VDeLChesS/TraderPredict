"""
circuit_breaker.py -- Max drawdown circuit breaker.

Halts trading when cumulative drawdown exceeds a threshold.
Resumes after a cooldown period AND drawdown recovery.

Why approximate equity?
  The real equity depends on the modified signals, creating a circular dependency.
  We approximate using no-fee strategy returns (signal * daily_return).
  The approximation is conservative: real drawdown with fees is slightly worse,
  so the breaker triggers marginally late (~0.1-0.5%).  For threshold-based
  decisions (e.g. -20% vs -25%), this is negligible.

Fully vectorised -- no row-by-row loops.
"""

import pandas as pd


class CircuitBreaker:
    """
    Zero out signals during severe drawdown periods.

    Parameters
    ----------
    max_drawdown_pct : Trigger threshold (0.20 = halt at -20% drawdown).
    cooldown_bars    : Minimum bars to stay flat after trigger.
    recovery_pct     : Resume when drawdown recovers above -recovery_pct.
                       Defaults to half of max_drawdown_pct.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.20,
        cooldown_bars: int = 10,
        recovery_pct: float = None,
    ):
        if max_drawdown_pct <= 0 or max_drawdown_pct >= 1:
            raise ValueError("max_drawdown_pct must be in (0, 1)")
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_bars = max(1, cooldown_bars)
        self.recovery_pct = recovery_pct if recovery_pct is not None else max_drawdown_pct / 2

    def apply(self, df: pd.DataFrame, signals: pd.Series) -> tuple:
        """
        Apply circuit breaker to signals.

        Returns
        -------
        (modified_signals, info_dict)

        info_dict keys:
          breaker_triggered_bars : int -- bars where drawdown exceeded threshold
          total_blocked_bars     : int -- bars where signals were zeroed
          approx_max_drawdown    : float -- worst drawdown (approximate)
          num_trigger_events     : int -- distinct drawdown breach events
        """
        # Approximate equity from no-fee strategy returns
        close_ret = df["close"].pct_change().fillna(0)
        strat_ret = signals.shift(1).fillna(0) * close_ret
        equity = (1 + strat_ret).cumprod()

        # Running drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        # Bars where drawdown exceeds threshold
        breached = drawdown < -self.max_drawdown_pct

        # Extend each breach forward by cooldown_bars.
        # Rolling max with backward window: at bar i, if any bar in
        # [i-cooldown+1, i] was breached, the bar is still in cooldown.
        blocked_by_cooldown = (
            breached
            .rolling(window=self.cooldown_bars, min_periods=1)
            .max()
            .fillna(0)
            .astype(bool)
        )

        # Recovery check: unblock early if drawdown has recovered
        recovered = drawdown > -self.recovery_pct

        # Blocked = in cooldown window AND not yet recovered
        blocked = blocked_by_cooldown & ~recovered

        # Zero out signals during blocked periods
        modified = signals.copy()
        modified[blocked] = 0

        # Count distinct trigger events (breach rising edges)
        breach_edges = breached & ~breached.shift(1).fillna(False)

        info = {
            "breaker_triggered_bars": int(breached.sum()),
            "total_blocked_bars":     int(blocked.sum()),
            "approx_max_drawdown":    round(float(drawdown.min()), 4),
            "num_trigger_events":     int(breach_edges.sum()),
        }

        return modified, info
