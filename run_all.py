"""
run_all.py -- End-to-end orchestration script.

Usage:
    python run_all.py              # Full pipeline: all symbols, all strategies, all risk configs
    python run_all.py --quick      # Quick mode: BTC-USD and SPY only, no_risk only

Phases executed:
  1. Load data (from cached Parquet, or fetch from yfinance)
  2. Build all strategies (MA, MA variants, EMA, RSI, ML, Ensembles)
  3. Run backtests under multiple risk configurations
  4. Run walk-forward evaluation (in-sample vs out-of-sample)
  5. Generate comparison tables, leaderboard, equity curves
  6. Save results/full_comparison.csv + walkforward_comparison.csv + PNG charts
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR,
    SYMBOLS, START_DATE, END_DATE,
)
from data_loader.data_loader import DataLoader
from pipeline.multi_strategy import MultiStrategyPipeline
from pipeline.report_generator import ReportGenerator

os.makedirs(str(RESULTS_DIR), exist_ok=True)


def main():
    quick = "--quick" in sys.argv
    symbols = ["BTC-USD", "SPY"] if quick else SYMBOLS

    print("=" * 70)
    print("  TraderPredict -- Multi-Strategy Pipeline")
    print(f"  Symbols: {symbols}")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print("=" * 70)

    # Ensure data exists
    loader = DataLoader(DATA_RAW_DIR, DATA_PROCESSED_DIR)
    for sym in symbols:
        try:
            loader.load(sym)
        except FileNotFoundError:
            print(f"  Fetching {sym} from yfinance...")
            raw = loader.fetch(sym, START_DATE, END_DATE)
            loader.clean(raw, sym)

    # Run pipeline
    pipeline = MultiStrategyPipeline(
        symbols=symbols,
        raw_dir=DATA_RAW_DIR,
        processed_dir=DATA_PROCESSED_DIR,
        model_dir=DATA_RAW_DIR.parent.parent / "models" / "saved",
        results_dir=RESULTS_DIR,
    )
    results = pipeline.run()

    # Generate reports
    gen = ReportGenerator(results, results_dir=str(RESULTS_DIR))

    print("\n")
    gen.print_summary()
    gen.print_leaderboard()
    gen.print_walkforward_summary()
    gen.save_csv()
    gen.save_walkforward_csv()

    # Portfolio aggregation across symbols (Tier 2 Item 5)
    if len(symbols) > 1:
        portfolios = gen.build_portfolios(
            strategy_names=["MA Crossover", "EMA 12/26", "MA 10/30"],
            risk_config="no_risk",
            methods=["equal_weight", "vol_weighted", "risk_parity"],
        )
        gen.print_portfolio_summary(portfolios)
        gen.save_portfolio_csv(portfolios)

    print("\n  Generating plots...")
    gen.plot_equity_curves()
    gen.plot_risk_comparison()
    gen.plot_walkforward_degradation()
    if len(symbols) > 1:
        gen.plot_portfolio_equity(portfolios)

    print("\n" + "=" * 70)
    print("  Pipeline complete. Results in: results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
