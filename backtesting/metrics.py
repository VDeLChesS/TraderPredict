"""
metrics.py — Pure numpy/pandas metric computation.
No vectorbt dependency: swap the backtesting library freely without touching this file.
All functions accept pandas Series with a DatetimeIndex (daily bars by default).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def total_return(equity_curve: pd.Series) -> float:
    """(final value / initial value) - 1"""
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    n_years = number_of_bars / periods_per_year
    """
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return 0.0
    n_years = len(equity_curve) / periods_per_year
    if n_years <= 0:
        return 0.0
    tr = total_return(equity_curve)
    return float((1 + tr) ** (1 / n_years) - 1)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.
    excess = returns - risk_free_rate / periods_per_year
    Sharpe = mean(excess) / std(excess) * sqrt(periods_per_year)
    Returns np.nan if std == 0 or fewer than 2 observations.
    """
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sortino ratio.
    Downside deviation = std of negative returns only.
    """
    if len(returns) < 2:
        return float("nan")
    excess      = returns - risk_free_rate / periods_per_year
    downside    = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = downside.std(ddof=1)
    if down_std == 0 or np.isnan(down_std):
        return float("nan")
    return float(excess.mean() / down_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Worst peak-to-trough decline as a negative float.
    e.g. -0.35 means -35% drawdown.
    """
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def win_rate(returns: pd.Series) -> float:
    """Fraction of active (non-zero) days that are positive."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """
    Gross profit / Gross loss.
    Returns np.inf if there are no losing days.
    Returns 0.0 if there are no winning days.
    """
    gains  = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    if gains == 0:
        return 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Convenience wrapper — returns all metrics in a single dict.
    Keys: total_return, cagr, sharpe, sortino, max_drawdown,
          win_rate, profit_factor
    """
    return {
        "total_return":  round(total_return(equity_curve), 4),
        "cagr":          round(cagr(equity_curve, periods_per_year), 4),
        "sharpe":        round(sharpe_ratio(returns, risk_free_rate, periods_per_year), 4),
        "sortino":       round(sortino_ratio(returns, risk_free_rate, periods_per_year), 4),
        "max_drawdown":  round(max_drawdown(equity_curve), 4),
        "win_rate":      round(win_rate(returns), 4),
        "profit_factor": round(profit_factor(returns), 4),
    }
