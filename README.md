# TraderPredict

A modular, end-to-end algorithmic trading research framework in Python. TraderPredict combines rule-based strategies, gradient-boosted ML models, ensemble methods, and configurable risk management — all evaluated under realistic transaction costs with proper walk-forward validation.

Built for **research and education**, not live trading.

---

## What it does

TraderPredict runs a complete backtesting pipeline for any list of stocks/crypto:

1. **Fetches historical OHLCV data** (via yfinance) and caches it as Parquet
2. **Engineers 55+ features** — returns, volatility, trend, momentum, Bollinger bands, volume, regime
3. **Trains ML direction classifiers** (XGBoost + LightGBM) to predict next-day price direction
4. **Runs 13 strategies** per symbol: rule-based, ML, and ensembles
5. **Tests each strategy under 3 risk configurations** — no risk, conservative, moderate
6. **Performs walk-forward evaluation** to measure in-sample vs out-of-sample degradation
7. **Generates reports** — comparison tables, leaderboards, equity curves, walk-forward charts

A single `python run_all.py` call produces ~312 backtest results across 8 symbols, plus 24 PNG charts and 2 CSV exports.

---

## Features

### Strategies (13 per symbol)
- **MA Crossover** (SMA 20/50) — classic trend-following baseline
- **MA 10/30** — fast trend variant
- **MA 50/200** — golden cross / long-term trend
- **EMA 12/26** — exponential moving average crossover
- **RSI** (14, 30/70) — mean-reversion overbought/oversold
- **ML XGBoost** — gradient boosting direction classifier (3 threshold variants: 0.55, 0.60, 0.65)
- **ML LightGBM** — same, with LightGBM
- **Ensemble (majority vote)** — MA + RSI + ML combined by majority vote
- **Ensemble (ML-gated)** — MA signal confirmed by ML

### Risk management (3 configs per strategy)
- **`no_risk`** — raw signals, full capital deployed (baseline for fair strategy comparison)
- **`conservative`** — volatility-scaled position sizing + ATR stops + circuit breaker on max drawdown
- **`moderate`** — fixed sizing with trailing stop-loss

Available primitives (in `risk/`):
- `PositionSizer` — fixed, volatility-scaled, ATR-scaled, Kelly criterion
- `StopLossConfig` — fixed %, ATR-based adaptive, trailing
- `CircuitBreaker` — halts trading when drawdown exceeds threshold (vectorized)
- `RiskManager` — orchestrator passed to the backtest engine

### Realistic backtesting
- Built on **vectorbt** for performant vectorized backtests
- **Fees** (0.1% per trade) and **slippage** (0.05%) applied by default
- **Signal shift** by 1 bar — no lookahead, signal on bar N executes at bar N+1 open
- **No-lookahead feature engineering** — every indicator uses only past data
- **Time-series-safe ML** — TimeSeriesSplit cross-validation, no shuffling

### Walk-forward evaluation
- 70/30 train/test split per strategy
- Reports in-sample Sharpe, out-of-sample Sharpe, and degradation %
- Flags strategies as **robust** (positive OOS, < 50% degradation) or **overfit**

### Metrics computed
- Total return, CAGR
- Sharpe ratio (annualized)
- Sortino ratio
- Max drawdown
- Win rate
- Profit factor
- Number of trades

---

## How it works

### Architecture

```
TraderPredict/
├── data_loader/        # yfinance fetch + cleaning + caching
├── features/           # FeatureEngineer (55+ technical indicators)
├── models/             # DirectionClassifier (XGBoost / LightGBM)
│   └── saved/          # Trained .pkl model files (gitignored)
├── strategies/         # MA, EMA, RSI, ML, Ensemble strategies
│   ├── base_strategy.py
│   ├── ma_crossover.py
│   ├── ema_crossover.py
│   ├── rsi_strategy.py
│   ├── ml_strategy.py
│   └── ensemble_strategy.py
├── risk/               # Position sizing, stops, circuit breaker
│   ├── position_sizer.py
│   ├── stop_loss.py
│   ├── circuit_breaker.py
│   └── risk_manager.py
├── backtesting/        # vectorbt engine + metric computation
│   ├── engine.py
│   └── metrics.py
├── pipeline/           # Multi-strategy orchestration + reporting
│   ├── multi_strategy.py
│   └── report_generator.py
├── scripts/            # Standalone CLI utilities
│   └── train_models.py
├── config.py           # Symbols, date range, paths, risk defaults
├── run_all.py          # End-to-end orchestration entry point
└── requirements.txt
```

### Data flow

```
yfinance ──► raw Parquet ──► cleaned OHLCV ──► FeatureEngineer
                                                     │
                                                     ▼
                                           55+ features + targets
                                                     │
                                ┌────────────────────┼────────────────────┐
                                ▼                    ▼                    ▼
                          MA / RSI / EMA       ML (XGBoost,         Ensemble
                          (rule-based)          LightGBM)
                                │                    │                    │
                                └────────────────────┼────────────────────┘
                                                     ▼
                                              signals {0, 1}
                                                     │
                                                     ▼
                                           RiskManager.apply()
                                                     │
                                                     ▼
                                          BacktestEngine (vectorbt)
                                                     │
                                                     ▼
                                       metrics + equity curve + trades
```

### No-lookahead guarantees
- All rolling indicators use only past data (`rolling()`, `ewm()`, `shift()`)
- Signal is shifted by 1 bar inside `BacktestEngine` — execution at next open
- ML targets (`target_ret_1d`, `target_direction_1d`) are stripped before inference
- Walk-forward CV uses `TimeSeriesSplit`, never shuffles

---

## Installation

```bash
git clone https://github.com/<your-username>/TraderPredict.git
cd TraderPredict

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+. Tested on Windows 11 with Python 3.14.

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `vectorbt`, `yfinance`, `ta`, `matplotlib`.

---

## Usage

### Quick start

```bash
# Train ML models for all configured symbols
python scripts/train_models.py

# Run the full pipeline (fetches data automatically if missing)
python run_all.py

# Quick mode (BTC-USD + SPY only)
python run_all.py --quick
```

### Configure symbols and date range

Edit `config.py`:

```python
SYMBOLS    = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "TSLA", "GLD", "TLT"]
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
INTERVAL   = "1d"
```

### Train models for specific symbols

```bash
# Train both XGBoost and LightGBM for two symbols
python scripts/train_models.py --symbols BTC-USD AAPL

# Use 5-day direction target instead of 1-day
python scripts/train_models.py --target 5d

# Train only XGBoost
python scripts/train_models.py --model-types xgboost
```

### Programmatic usage

```python
from pipeline.multi_strategy import MultiStrategyPipeline
from pipeline.report_generator import ReportGenerator
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR

pipeline = MultiStrategyPipeline(
    symbols=["BTC-USD", "SPY"],
    raw_dir=DATA_RAW_DIR,
    processed_dir=DATA_PROCESSED_DIR,
    results_dir=RESULTS_DIR,
)
results = pipeline.run()

gen = ReportGenerator(results, results_dir=str(RESULTS_DIR))
gen.print_summary()
gen.print_leaderboard()
gen.print_walkforward_summary()
gen.save_csv()
gen.save_walkforward_csv()
gen.plot_equity_curves()
```

---

## Outputs

After running `python run_all.py`, check the `results/` directory:

### CSV files
- **`full_comparison.csv`** — Every strategy x symbol x risk config combination (~312 rows for 8 symbols). Columns: symbol, strategy, risk_config, total_return, cagr, sharpe, sortino, max_drawdown, win_rate, profit_factor, num_trades.
- **`walkforward_comparison.csv`** — In-sample vs out-of-sample metrics per strategy (~104 rows). Columns: is_sharpe, oos_sharpe, degradation_pct, etc.

### PNG charts (3 per symbol)
- **`{symbol}_multi_equity.png`** — All strategy equity curves overlaid on one chart, normalised to 1.0
- **`{symbol}_risk_comparison.png`** — MA Crossover under each risk config (no_risk vs conservative vs moderate)
- **`{symbol}_walkforward.png`** — Bar chart: IS Sharpe vs OOS Sharpe per strategy

### Console reports
- Per-symbol comparison tables grouped by risk config
- Cross-symbol leaderboard (best strategy per symbol by Sharpe)
- Risk management impact table
- Walk-forward summary with robust/overfit flags

---

## Limitations

> **TraderPredict is a research and education tool. It is not financial advice and not designed for live trading.**

### Methodological limits
- **Daily bars only.** No intraday data, no tick-level execution, no bid/ask spread modeling.
- **Long-only.** No short selling, no options, no leverage modeling beyond capital fraction.
- **Single-asset backtests.** Each symbol is tested in isolation — no portfolio allocation, no correlation between trades, no margin sharing.
- **No regime detection in execution.** Strategies trade the same way in all market conditions.
- **Walk-forward is a single split** (70/30), not rolling/expanding window CV at the backtest level.
- **No Monte Carlo or sensitivity analysis** — all results are point estimates.

### Data limits
- **Survivorship bias.** Symbols are hand-picked; delisted assets are not included.
- **yfinance quirks.** Adjusted close history can be revised retroactively; corporate actions may affect older data.
- **Fixed lookback.** `START_DATE = 2019-01-01` includes COVID volatility but not the 2008 crisis or the 2015–18 sideways period.

### Cost model limits
- Flat **0.1% fees** and **0.05% slippage** per trade — does not model variable spreads, market impact at size, exchange fee schedules, or financing costs.
- **No taxes.**

### ML limits
- Direction classifiers are trained on **~1,300 rows per asset** — far below what neural networks need. Tabular gradient boosting is the right tool, but signal-to-noise is genuinely low on daily bars.
- **No hyperparameter tuning** in the pipeline (Optuna integration is on the roadmap).
- **No feature selection or pruning** — all 55 features fed to every model.
- **No probability calibration** beyond what XGBoost/LightGBM produce out of the box.

### Not implemented (yet)
- Paper trading / live broker integration
- Multi-timeframe analysis (combining daily + weekly signals)
- Sector rotation or cross-asset portfolio allocation
- Alternative data (sentiment, on-chain, macro)
- Bayesian / probabilistic backtesting

---

## Reading the results

A few rules of thumb when interpreting outputs:

- **Sharpe > 1.0** is genuinely good for daily strategies after fees.
- **Sharpe < 0.5** is barely better than random — most "winners" in backtests fall in this range.
- **OOS Sharpe < 50% of IS Sharpe** is a strong overfitting signal. Watch the `walkforward_comparison.csv` `degradation_pct` column.
- **Trade counts > 200** for ML strategies are typical (signal flips often). High trade counts amplify the impact of fees.
- **Conservative risk configs trade returns for drawdown protection** — expect lower Sharpe but much smaller max drawdown. The `RISK MANAGEMENT IMPACT` table in the console output makes this trade-off explicit.
- **Buy & Hold often wins on equity assets in bull markets.** A strategy that loses to B&H but with much lower drawdown can still be valuable depending on your objective.

---

## Project status

This project was built progressively in 8 phases:

1. Data loader (yfinance + Parquet caching)
2. Feature engineering (55+ indicators)
3. Rule-based strategies (MA, RSI)
4. Backtesting engine (vectorbt + metrics)
5. ML direction classifiers
6. ML strategy + ensembles
7. Risk management (sizing, stops, circuit breaker)
8. Multi-strategy pipeline + reports

Recent additions (Tier 1 quick wins):
- Expanded universe: 8 symbols (BTC, ETH, SPY, QQQ, AAPL, TSLA, GLD, TLT)
- ML threshold tuning (3 confidence levels)
- MA variants + EMA crossover
- Walk-forward evaluation

---

## Roadmap

**Tier 2 — Medium effort**
- 5-day direction target option
- Volatility regime filtering
- Optuna hyperparameter tuning
- Multi-timeframe features (daily + weekly)
- Portfolio allocation across symbols

**Tier 3 — Substantial features**
- Paper trading via broker API
- Alternative data integration
- Sensitivity / Monte Carlo analysis
- Sector rotation strategy
- Intraday support

---

## License

Released under the MIT License. See `LICENSE` for details.

## Disclaimer

Past performance is not indicative of future results. This software is provided "as is" for research and educational purposes. The author is not a registered investment advisor. Do not use this code to make real trading decisions without independent verification and proper risk controls.
