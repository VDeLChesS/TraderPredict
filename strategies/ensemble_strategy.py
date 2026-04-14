"""
ensemble_strategy.py -- Combine multiple strategies into one signal.

Why ensemble?
  No single strategy dominates across all market regimes.
  Combining uncorrelated signals reduces variance and drawdown
  while preserving average edge.

Four combination modes:
  majority_vote  -- signal = 1 if more than half of strategies agree
  weighted_vote  -- weighted average of signals, threshold applied
  unanimous      -- signal = 1 only if ALL strategies agree (conservative)
  ml_gated       -- rule-based signal, but only when ML also agrees
                   (ML acts as a confidence filter, not the primary alpha)

Typical use:
  - majority_vote of [MA crossover, RSI, ML] gives a balanced blend
  - ml_gated with MA crossover = trend + AI confirmation
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategies.base_strategy import BaseStrategy


class EnsembleStrategy(BaseStrategy):
    """
    Combine an arbitrary list of strategies into a single {0,1} signal.

    Parameters
    ----------
    strategies : dict of {name: BaseStrategy}
    weights    : dict of {name: float} -- used only for 'weighted_vote'.
                 Defaults to equal weight if not provided.
    mode       : 'majority_vote' | 'weighted_vote' | 'unanimous' | 'ml_gated'
    threshold  : vote fraction required to trigger entry (for weighted_vote).
                 Default 0.5 (= more than half).
    ml_name    : name of the ML strategy in the dict (for 'ml_gated' mode).
    rule_name  : name of the rule-based strategy (for 'ml_gated' mode).
    """

    VALID_MODES = ("majority_vote", "weighted_vote", "unanimous", "ml_gated")

    def __init__(
        self,
        strategies: dict,
        weights: dict = None,
        mode: str = "majority_vote",
        threshold: float = 0.5,
        ml_name: str = None,
        rule_name: str = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if mode == "ml_gated" and (ml_name is None or rule_name is None):
            raise ValueError("ml_gated mode requires ml_name and rule_name")

        self.strategies = strategies
        self.weights    = weights or {n: 1.0 for n in strategies}
        self.mode       = mode
        self.threshold  = threshold
        self.ml_name    = ml_name
        self.rule_name  = rule_name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Compute individual signals then combine according to mode."""
        return self._apply_combination(self._collect_signals(df)).rename(
            f"ensemble_{self.mode}_signal"
        )

    def get_individual_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame of all individual strategy signals (for diagnostics)."""
        return self._collect_signals(df)

    # ------------------------------------------------------------------
    # Private: shared signal collection (avoids double computation)
    # ------------------------------------------------------------------

    def _collect_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every sub-strategy signal once. Used by all public methods."""
        signals = {
            name: strat.generate_signals(df).reindex(df.index).fillna(0).astype(int)
            for name, strat in self.strategies.items()
        }
        return pd.DataFrame(signals, index=df.index)

    def _apply_combination(self, sig_df: pd.DataFrame) -> pd.Series:
        """Apply the configured mode to a pre-built signal DataFrame."""
        if self.mode == "majority_vote":
            return self._majority_vote(sig_df)
        if self.mode == "weighted_vote":
            return self._weighted_vote(sig_df)
        if self.mode == "unanimous":
            return self._unanimous(sig_df)
        if self.mode == "ml_gated":
            return self._ml_gated(sig_df)
        raise ValueError(f"Unknown mode: {self.mode}")

    def signal_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pearson correlation matrix of individual strategy signals.
        Low correlations (~0) = diverse, independent signals = good ensemble.
        High correlations (~1) = redundant strategies = no diversification benefit.
        """
        return self.get_individual_signals(df).corr()

    def signal_agreement_rate(self, df: pd.DataFrame) -> dict:
        """
        Fraction of bars where all strategies agree vs. disagree.
        High disagreement = strategies are diverse (healthy for ensembles).
        """
        sig_df    = self.get_individual_signals(df)
        all_agree = (sig_df.nunique(axis=1) == 1).mean()
        all_long  = (sig_df == 1).all(axis=1).mean()
        all_flat  = (sig_df == 0).all(axis=1).mean()
        return {
            "pct_all_agree": round(float(all_agree), 4),
            "pct_all_long":  round(float(all_long), 4),
            "pct_all_flat":  round(float(all_flat), 4),
            "pct_disagree":  round(1.0 - float(all_agree), 4),
        }

    def plot_signal_stack(
        self,
        df: pd.DataFrame,
        symbol: str,
        results_dir: str = "results",
    ) -> str:
        """
        Multi-panel chart: close price + individual signals + combined signal.
        Saves PNG and returns the path.
        """
        sig_df   = self._collect_signals(df)
        combined = self._apply_combination(sig_df).rename(f"ensemble_{self.mode}_signal")
        n_strats = len(self.strategies)

        fig, axes = plt.subplots(
            n_strats + 2, 1,
            figsize=(14, 3 * (n_strats + 2)),
            sharex=True,
        )

        # Panel 0: close price with combined signal shading
        ax0 = axes[0]
        ax0.plot(df.index, df["close"], color="black", linewidth=1, label="Close")
        ax0.fill_between(
            df.index, df["close"].min(), df["close"].max(),
            where=(combined == 1),
            alpha=0.15, color="green", label="Ensemble long",
        )
        ax0.set_title(f"{symbol} -- Ensemble Strategy ({self.mode})")
        ax0.set_ylabel("Price")
        ax0.legend(loc="upper left", fontsize=8)
        ax0.grid(alpha=0.3)

        # Individual signal panels
        colors = ["#2196F3", "#FF9800", "#9C27B0", "#F44336", "#009688"]
        for i, (name, _) in enumerate(self.strategies.items()):
            ax = axes[i + 1]
            ax.fill_between(
                sig_df.index, sig_df[name],
                alpha=0.6, color=colors[i % len(colors)], label=name,
            )
            ax.set_ylabel(name[:12], fontsize=8)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(alpha=0.3)

        # Combined signal panel
        ax_last = axes[-1]
        ax_last.fill_between(
            combined.index, combined,
            alpha=0.7, color="green", label=f"Combined ({self.mode})",
        )
        ax_last.set_ylabel("Combined", fontsize=8)
        ax_last.set_ylim(-0.1, 1.1)
        ax_last.legend(loc="upper left", fontsize=7)
        ax_last.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"{symbol}_ensemble_{self.mode}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Private combination methods
    # ------------------------------------------------------------------

    def _majority_vote(self, sig_df: pd.DataFrame) -> pd.Series:
        """Signal = 1 if strictly more than half of strategies are long."""
        n = len(sig_df.columns)
        vote_sum = sig_df.sum(axis=1)
        return (vote_sum > n / 2).astype(int)

    def _weighted_vote(self, sig_df: pd.DataFrame) -> pd.Series:
        """Weighted average of signals; enter if weighted sum >= threshold. Vectorised."""
        w = pd.Series([self.weights.get(n, 1.0) for n in sig_df.columns], index=sig_df.columns)
        weighted_sum = sig_df.mul(w, axis=1).sum(axis=1)
        return (weighted_sum / w.sum() >= self.threshold).astype(int)

    def _unanimous(self, sig_df: pd.DataFrame) -> pd.Series:
        """Signal = 1 only if ALL strategies are long."""
        return (sig_df == 1).all(axis=1).astype(int)

    def _ml_gated(self, sig_df: pd.DataFrame) -> pd.Series:
        """
        ML acts as a gate on the rule-based signal:
        Combined = 1 only when rule_name=1 AND ml_name=1.
        If ML says flat, we stay flat regardless of the rule signal.
        This is the most useful mode: rule-based finds the opportunity,
        ML confirms it has a positive expected value.
        """
        rule_sig = sig_df[self.rule_name]
        ml_sig   = sig_df[self.ml_name]
        return (rule_sig & ml_sig).astype(int)
