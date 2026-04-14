"""
multi_strategy.py -- Full multi-strategy pipeline.

For each symbol:
  1. Load data
  2. Build features
  3. Run every strategy (rule-based, ML, ensemble)
  4. Run backtests with and without risk management
  5. Run walk-forward evaluation (in-sample vs out-of-sample)
  6. Collect all results into a single comparison structure

Strategies evaluated:
  - MA Crossover 20/50 (baseline)
  - MA 10/30 (fast)
  - MA 50/200 (golden cross)
  - EMA 12/26 (exponential crossover)
  - RSI (14, 30/70)
  - ML XGBoost (multiple thresholds if enabled)
  - ML LightGBM (multiple thresholds if enabled)
  - Ensemble: majority vote (MA + RSI + ML)
  - Ensemble: ML-gated (MA confirmed by ML)

Each strategy is also tested in three risk configurations:
  - No risk management (raw signals)
  - Conservative (vol-scaled sizing + ATR stops + circuit breaker)
  - Moderate (fixed sizing + trailing stop, no breaker)
"""

import os
import warnings
from pathlib import Path

import pandas as pd

from data_loader.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.direction_classifier import DirectionClassifier
from strategies.ma_crossover import MACrossoverStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.ml_strategy import MLStrategy
from strategies.ensemble_strategy import EnsembleStrategy
from backtesting.engine import BacktestEngine
from risk.risk_manager import RiskManager

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Risk configs (reusable across symbols)
# ------------------------------------------------------------------

RISK_CONFIGS = {
    "no_risk": None,
    "conservative": RiskManager(
        position_mode="vol_scaled",
        stop_mode="atr",
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        use_circuit_breaker=True,
        max_drawdown_pct=0.20,
        cooldown_bars=10,
    ),
    "moderate": RiskManager(
        position_mode="fixed",
        position_fraction=1.0,
        stop_mode="trailing",
        sl_pct=0.05,
        use_circuit_breaker=False,
    ),
}

# ------------------------------------------------------------------
# ML threshold tuning grid (Item 2)
# ------------------------------------------------------------------

ML_THRESHOLD_GRID = [
    {"long_threshold": 0.55, "exit_threshold": 0.50, "label": ""},
    {"long_threshold": 0.60, "exit_threshold": 0.55, "label": " T60"},
    {"long_threshold": 0.65, "exit_threshold": 0.55, "label": " T65"},
]


class MultiStrategyPipeline:
    """
    Orchestrate multi-strategy backtesting across symbols.

    Usage:
        pipeline = MultiStrategyPipeline(symbols, data_dir, model_dir)
        results  = pipeline.run()
        # results is a dict: {symbol: {strategy_name: {risk_config: backtest_result}}}
    """

    def __init__(
        self,
        symbols: list,
        raw_dir: Path,
        processed_dir: Path,
        model_dir: Path = Path("models/saved"),
        results_dir: Path = Path("results"),
        init_cash: float = 10_000.0,
        fees: float = 0.001,
        slippage: float = 0.0005,
        walkforward_train_pct: float = 0.70,
    ):
        self.symbols       = symbols
        self.model_dir     = Path(model_dir)
        self.results_dir   = Path(results_dir)
        self.loader        = DataLoader(raw_dir, processed_dir)
        self.engine        = BacktestEngine(init_cash=init_cash, fees=fees, slippage=slippage)
        self.fe            = FeatureEngineer(drop_warmup=False)
        self.walkforward_train_pct = walkforward_train_pct

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run all strategies x all risk configs x all symbols.

        Returns
        -------
        dict structure:
          {
            symbol: {
              "data": pd.DataFrame,  # OHLCV
              "strategies": {
                strategy_name: {
                  risk_config_name: backtest_result_dict
                }
              },
              "comparison": pd.DataFrame,  # summary table
              "walkforward": {
                strategy_name: walkforward_result_dict
              },
            }
          }
        """
        all_results = {}

        for sym in self.symbols:
            print(f"\n{'='*70}")
            print(f"  Processing: {sym}")
            print(f"{'='*70}")

            df = self._load_data(sym)
            if df is None:
                continue

            strategies = self._build_strategies(df, sym)
            sym_results = self._run_all_backtests(df, strategies, sym)

            # Walk-forward evaluation (Item 4)
            print(f"\n  Walk-forward evaluation ({self.walkforward_train_pct:.0%} train)...")
            walkforward = self._run_walkforward_all(df, strategies, sym)

            comparison = self._build_comparison_table(sym_results, sym)

            all_results[sym] = {
                "data":        df,
                "strategies":  sym_results,
                "comparison":  comparison,
                "walkforward": walkforward,
            }

        return all_results

    # ------------------------------------------------------------------
    # Private: data loading
    # ------------------------------------------------------------------

    def _load_data(self, symbol: str):
        """Load processed data; return None on failure."""
        try:
            df = self.loader.load(symbol)
            print(f"  Loaded {symbol}: {len(df)} bars "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
            return df
        except FileNotFoundError:
            print(f"  [SKIP] No data for {symbol} -- run data fetch first")
            return None

    # ------------------------------------------------------------------
    # Private: strategy construction
    # ------------------------------------------------------------------

    def _build_strategies(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Build all applicable strategies for a symbol.
        Returns dict of {name: strategy_instance}.
        ML strategies only included if trained model exists.
        """
        strategies = {}

        # --- Rule-based: MA variants ---
        ma = MACrossoverStrategy(fast_window=20, slow_window=50)
        strategies["MA Crossover"] = ma

        # Additional MA variants (Item 3)
        strategies["MA 10/30"]  = MACrossoverStrategy(fast_window=10, slow_window=30)
        strategies["MA 50/200"] = MACrossoverStrategy(fast_window=50, slow_window=200)

        # EMA crossover (Item 3)
        strategies["EMA 12/26"] = EMACrossoverStrategy(fast_window=12, slow_window=26)

        # --- Rule-based: RSI ---
        rsi = RSIStrategy(rsi_window=14, oversold=30, overbought=70)
        strategies["RSI"] = rsi

        # --- ML strategies (load if model file exists) ---
        sym_file = symbol.replace("-", "_")
        ml_strats = {}

        for model_type in ["xgboost", "lightgbm"]:
            model_path = self.model_dir / f"{sym_file}_{model_type}.pkl"
            if model_path.exists():
                try:
                    clf = DirectionClassifier.load(model_path)

                    # Default threshold strategy (for ensembles)
                    default_cfg = ML_THRESHOLD_GRID[0]
                    ml_strat = MLStrategy(
                        clf,
                        long_threshold=default_cfg["long_threshold"],
                        exit_threshold=default_cfg["exit_threshold"],
                    )
                    name = f"ML {model_type.upper()}"
                    strategies[name] = ml_strat
                    ml_strats[model_type] = ml_strat
                    print(f"  Loaded ML model: {model_path.name}")

                    # Threshold variants (Item 2) -- skip the default (already added)
                    for cfg in ML_THRESHOLD_GRID[1:]:
                        variant_strat = MLStrategy(
                            clf,
                            long_threshold=cfg["long_threshold"],
                            exit_threshold=cfg["exit_threshold"],
                        )
                        variant_name = f"ML {model_type.upper()}{cfg['label']}"
                        strategies[variant_name] = variant_strat

                except Exception as e:
                    print(f"  [WARN] Failed to load {model_path.name}: {e}")

        # --- Ensembles (require at least one ML model) ---
        best_ml_name = None
        best_ml_strat = None
        if "xgboost" in ml_strats:
            best_ml_name  = "ML XGBOOST"
            best_ml_strat = ml_strats["xgboost"]
        elif "lightgbm" in ml_strats:
            best_ml_name  = "ML LIGHTGBM"
            best_ml_strat = ml_strats["lightgbm"]

        if best_ml_strat is not None:
            # Majority vote: MA + RSI + ML
            ens_vote = EnsembleStrategy(
                strategies={"MA": ma, "RSI": rsi, "ML": best_ml_strat},
                mode="majority_vote",
            )
            strategies["Ensemble (vote)"] = ens_vote

            # ML-gated: MA signal confirmed by ML
            ens_gated = EnsembleStrategy(
                strategies={"MA": ma, "ML": best_ml_strat},
                mode="ml_gated",
                ml_name="ML",
                rule_name="MA",
            )
            strategies["Ensemble (ML-gated)"] = ens_gated

        print(f"  Strategies: {list(strategies.keys())}")
        return strategies

    # ------------------------------------------------------------------
    # Private: run backtests
    # ------------------------------------------------------------------

    def _run_all_backtests(self, df, strategies, symbol) -> dict:
        """
        Run every (strategy, risk_config) combination.
        Returns nested dict: {strategy_name: {risk_config: result}}
        """
        results = {}

        for strat_name, strat in strategies.items():
            results[strat_name] = {}
            signals = strat.generate_signals(df)

            for risk_name, risk_mgr in RISK_CONFIGS.items():
                result = self.engine.run_backtest(
                    df, signals, symbol, risk_manager=risk_mgr,
                )
                results[strat_name][risk_name] = result

                m = result["metrics"]
                tag = f"{strat_name} / {risk_name}"
                print(f"    {tag:<40} "
                      f"ret={m['total_return']:>7.2%}  "
                      f"sharpe={m['sharpe']:>6.3f}  "
                      f"maxDD={m['max_drawdown']:>7.2%}  "
                      f"trades={m['num_trades']}")

        return results

    # ------------------------------------------------------------------
    # Private: walk-forward evaluation (Item 4)
    # ------------------------------------------------------------------

    def _run_walkforward_all(self, df, strategies, symbol) -> dict:
        """
        Run walk-forward (train/test split) for every strategy.
        Returns {strategy_name: walkforward_result}.

        walkforward_result has keys: train, test, full, split_date
        Each sub-result contains metrics, equity_curve, etc.
        """
        walkforward = {}

        for strat_name, strat in strategies.items():
            try:
                wf_result = self.engine.run_walkforward(
                    df, strat,
                    train_pct=self.walkforward_train_pct,
                    symbol=symbol,
                )
                walkforward[strat_name] = wf_result

                train_m = wf_result["train"]["metrics"]
                test_m  = wf_result["test"]["metrics"]

                # Degradation: how much does Sharpe drop from in-sample to OOS?
                sharpe_in  = train_m["sharpe"]
                sharpe_oos = test_m["sharpe"]
                degradation = (sharpe_in - sharpe_oos) / abs(sharpe_in) if sharpe_in != 0 else 0.0

                print(f"    WF {strat_name:<32} "
                      f"IS={sharpe_in:>6.3f}  "
                      f"OOS={sharpe_oos:>6.3f}  "
                      f"deg={degradation:>+6.1%}")

            except Exception as e:
                print(f"    WF {strat_name:<32} [ERROR] {e}")

        return walkforward

    # ------------------------------------------------------------------
    # Private: comparison table
    # ------------------------------------------------------------------

    def _build_comparison_table(self, sym_results: dict, symbol: str) -> pd.DataFrame:
        """
        Flatten nested results into a single DataFrame.
        Rows = (strategy, risk_config), columns = metrics.
        """
        rows = []
        for strat_name, risk_results in sym_results.items():
            for risk_name, result in risk_results.items():
                row = {
                    "symbol":      symbol,
                    "strategy":    strat_name,
                    "risk_config": risk_name,
                    **result["metrics"],
                }
                rows.append(row)

        return pd.DataFrame(rows)
