"""
portfolio_allocator.py -- Multi-symbol portfolio aggregation.

Takes per-symbol backtest results from MultiStrategyPipeline and combines
them into a single portfolio equity curve under various allocation methods:

  - "equal_weight"  : 1/N capital per symbol, rebalanced annually
  - "vol_weighted"  : weight inversely proportional to realised volatility
                      (lower vol -> bigger weight)
  - "risk_parity"   : equal risk contribution (also vol-inverse, normalised)

Why portfolio aggregation?
  Single-asset Sharpe ratios overstate live portfolio performance because
  they ignore diversification.  A portfolio of low-correlation assets has
  better risk-adjusted returns than the average of its components.

Inputs:
  results_dict from MultiStrategyPipeline.run() — the nested
  {symbol: {strategies: {...}, ...}} structure.

Outputs:
  PortfolioResult with equity_curve, returns, metrics, weights, contributions.
"""

import numpy as np
import pandas as pd

from backtesting.metrics import compute_all_metrics


class PortfolioAllocator:
    """
    Combine per-symbol backtest results into a single portfolio.

    Parameters
    ----------
    results : dict
        Output from MultiStrategyPipeline.run().
    strategy_name : str
        Which strategy to allocate across (must exist in every symbol).
    risk_config : str
        Which risk configuration to use ("no_risk", "conservative", "moderate").
    method : "equal_weight" | "vol_weighted" | "risk_parity"
    rebalance_freq : str
        Pandas resample alias for rebalancing ("YE"=annual, "QE"=quarterly,
        "ME"=monthly).
    vol_lookback : int
        Bars of return history used to estimate vol for weighting.
    init_cash : float
        Total starting capital for the portfolio.
    """

    VALID_METHODS = ("equal_weight", "vol_weighted", "risk_parity")

    def __init__(
        self,
        results: dict,
        strategy_name: str = "EMA 12/26",
        risk_config: str = "no_risk",
        method: str = "equal_weight",
        rebalance_freq: str = "YE",
        vol_lookback: int = 60,
        init_cash: float = 10_000.0,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        self.results        = results
        self.strategy_name  = strategy_name
        self.risk_config    = risk_config
        self.method         = method
        self.rebalance_freq = rebalance_freq
        self.vol_lookback   = vol_lookback
        self.init_cash      = init_cash

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Build the portfolio. Returns a dict with:
          equity_curve : pd.Series
          returns      : pd.Series (daily portfolio returns)
          metrics      : dict of standard metrics
          weights      : pd.DataFrame (symbol weights at each rebalance)
          per_symbol   : dict of per-symbol contribution stats
        """
        symbol_returns = self._extract_symbol_returns()
        if symbol_returns.empty:
            return self._empty_result()

        # Outer-join all symbols on a common DatetimeIndex
        symbol_returns = symbol_returns.fillna(0.0)

        # Compute weights at each rebalance point
        weights = self._compute_rebalance_weights(symbol_returns)

        # Daily portfolio return = sum(weight * symbol_return)
        # weights are forward-filled between rebalance dates
        port_returns = (symbol_returns * weights).sum(axis=1)
        equity_curve = self.init_cash * (1 + port_returns).cumprod()

        metrics = compute_all_metrics(equity_curve, port_returns)
        metrics["num_trades"] = self._approximate_num_trades()

        per_symbol = self._build_per_symbol_stats(symbol_returns, weights)

        return {
            "equity_curve": equity_curve,
            "returns":      port_returns,
            "metrics":      metrics,
            "weights":      weights,
            "per_symbol":   per_symbol,
            "method":       self.method,
            "strategy":     self.strategy_name,
            "risk_config":  self.risk_config,
        }

    # ------------------------------------------------------------------
    # Private: data extraction
    # ------------------------------------------------------------------

    def _extract_symbol_returns(self) -> pd.DataFrame:
        """Build a (date x symbol) DataFrame of strategy returns."""
        per_symbol = {}
        for sym, sym_data in self.results.items():
            strat_results = sym_data.get("strategies", {})
            if self.strategy_name not in strat_results:
                continue
            risk_results = strat_results[self.strategy_name]
            if self.risk_config not in risk_results:
                continue
            per_symbol[sym] = risk_results[self.risk_config]["returns"]

        if not per_symbol:
            return pd.DataFrame()

        return pd.DataFrame(per_symbol).sort_index()

    # ------------------------------------------------------------------
    # Private: weight computation
    # ------------------------------------------------------------------

    def _compute_rebalance_weights(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-bar symbol weights, recomputed at each rebalance date.

        Returns a DataFrame indexed by the same dates as returns_df, with
        one column per symbol. Weights are forward-filled between rebalances.
        """
        n_symbols = returns_df.shape[1]

        # Identify rebalance dates: the first bar of each period
        rebalance_dates = returns_df.resample(self.rebalance_freq).first().index

        weights = pd.DataFrame(
            index=returns_df.index,
            columns=returns_df.columns,
            dtype=float,
        )

        for rdate in rebalance_dates:
            # Find first index >= rdate (the actual rebalance bar)
            idx = returns_df.index.searchsorted(rdate)
            if idx >= len(returns_df):
                continue

            # Look back to estimate vol if needed
            past = returns_df.iloc[max(0, idx - self.vol_lookback):idx]

            if self.method == "equal_weight" or len(past) < 5:
                w = np.ones(n_symbols) / n_symbols
            elif self.method == "vol_weighted":
                vol = past.std(ddof=1).replace(0, np.nan)
                inv_vol = 1.0 / vol
                inv_vol = inv_vol.fillna(0.0)
                w = (inv_vol / inv_vol.sum()).values if inv_vol.sum() > 0 \
                    else np.ones(n_symbols) / n_symbols
            else:  # risk_parity (same as inverse vol for uncorrelated assets)
                vol = past.std(ddof=1).replace(0, np.nan)
                inv_vol = 1.0 / vol
                inv_vol = inv_vol.fillna(0.0)
                w = (inv_vol / inv_vol.sum()).values if inv_vol.sum() > 0 \
                    else np.ones(n_symbols) / n_symbols

            weights.iloc[idx] = w

        # Forward-fill weights between rebalance dates
        weights = weights.ffill().fillna(1.0 / n_symbols)
        return weights

    # ------------------------------------------------------------------
    # Private: stats
    # ------------------------------------------------------------------

    def _build_per_symbol_stats(self, returns_df, weights) -> dict:
        """Compute total contribution of each symbol."""
        contributions = (returns_df * weights).sum(axis=0)
        avg_weights   = weights.mean(axis=0)
        return {
            sym: {
                "avg_weight":  float(avg_weights[sym]),
                "contribution": float(contributions[sym]),
            }
            for sym in returns_df.columns
        }

    def _approximate_num_trades(self) -> int:
        """Sum num_trades across symbols for the chosen strategy/risk."""
        total = 0
        for sym, sym_data in self.results.items():
            strat_results = sym_data.get("strategies", {})
            if self.strategy_name not in strat_results:
                continue
            risk_results = strat_results[self.strategy_name]
            if self.risk_config not in risk_results:
                continue
            total += int(risk_results[self.risk_config]["metrics"].get("num_trades", 0))
        return total

    def _empty_result(self) -> dict:
        return {
            "equity_curve": pd.Series(dtype=float),
            "returns":      pd.Series(dtype=float),
            "metrics":      {},
            "weights":      pd.DataFrame(),
            "per_symbol":   {},
            "method":       self.method,
            "strategy":     self.strategy_name,
            "risk_config":  self.risk_config,
        }
