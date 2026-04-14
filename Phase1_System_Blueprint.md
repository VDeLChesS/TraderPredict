# TraderPredict — Phase 1: Institutional-Grade AI Trading System Blueprint
> Status: **PLANNING ONLY — Awaiting Approval Before Any Implementation**

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Strategy & Feature Engineering](#2-data-strategy--feature-engineering)
3. [Modeling Approach](#3-modeling-approach)
4. [Decision Engine](#4-decision-engine)
5. [Infrastructure & Tech Stack](#5-infrastructure--tech-stack)
6. [Backtesting & Evaluation Framework](#6-backtesting--evaluation-framework)
7. [Risks, Limitations & Realistic Expectations](#7-risks-limitations--realistic-expectations)

---

## 1. System Architecture Overview

### 1.1 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│  Market Feeds │ Macro APIs │ News/NLP │ Order Book │ Alt Data       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Kafka Topics (real-time stream)
┌───────────────────────────▼─────────────────────────────────────────┐
│                     STREAM PROCESSING LAYER                          │
│  Bytewax / Apache Flink — normalization, dedup, alignment           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
          ┌─────────────────┴───────────────────┐
          │                                     │
┌─────────▼──────────┐              ┌───────────▼───────────┐
│   REAL-TIME PATH   │              │     BATCH PATH        │
│  (sub-second to    │              │  (hourly / daily /    │
│   5-minute)        │              │   weekly)             │
│                    │              │                       │
│ • Tick features    │              │ • Model retraining    │
│ • Order book       │              │ • Macro features      │
│ • Sentiment        │              │ • Fundamental data    │
│ • Live model infer │              │ • Regime detection    │
└─────────┬──────────┘              └───────────┬───────────┘
          └─────────────────┬───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                      FEATURE STORE (Feast)                          │
│  Online Store (Redis) ←→ Offline Store (ClickHouse / Parquet S3)   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                         MODEL LAYER                                  │
│  Statistical │ GBM │ Transformer │ NLP │ GNN │ RL Agent │ Ensemble  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                      SIGNAL AGGREGATION                              │
│  Meta-Learner (stacking) → confidence-weighted composite signal     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                       DECISION ENGINE                                │
│  Trade Scoring → Risk Engine → Portfolio Constructor → OMS          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                      EXECUTION LAYER                                 │
│  Smart Order Router → Broker API → Position Tracker → PnL Monitor  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                   MONITORING & FEEDBACK LOOP                         │
│  MLflow │ Grafana │ Drift Detection │ Continuous Retraining          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Real-Time vs. Batch Decomposition

| Component | Latency Target | Technology | Trigger |
|---|---|---|---|
| Tick ingestion | < 10ms | Kafka + Bytewax | Continuous |
| Order book snapshot | 100ms | Redis FIFO buffer | Per tick |
| Technical indicators | 1–5 min | Bytewax windowed ops | Per bar close |
| News sentiment | 30–60 sec | FinBERT inference server | On publish |
| Live model inference | < 500ms | BentoML endpoint | Per signal cycle |
| Daily feature batch | < 5 min | Airflow DAG | Market close |
| Model retraining | 4–24 hrs | Airflow + GPU cluster | Weekly or on drift |
| Regime detection | 1 hr | HMM batch job | Hourly |
| Portfolio optimization | < 1 sec | Clarabel solver | Pre-open + intraday |
| Macro data refresh | Daily | FRED/BLS API polling | 8:30 AM ET |

### 1.3 Modular Service Design

Each of the following runs as an independently deployable microservice (containerized):

- **DataService**: Unified connector layer to all external APIs; normalizes to internal schema
- **FeatureService**: Computes and serves features; interfaces with Feast
- **ModelService**: Hosts trained model endpoints; versioned via MLflow model registry
- **SignalService**: Runs meta-learner ensemble; outputs composite signal + confidence
- **RiskService**: Real-time position-level risk calculations (VaR, drawdown, correlation)
- **PortfolioService**: Black-Litterman optimization + constraint enforcement
- **ExecutionService**: OMS, smart order routing, broker API abstraction
- **MonitorService**: PnL attribution, drift detection, alerting

---

## 2. Data Strategy & Feature Engineering

### 2.1 Data Sources by Asset Class

#### Market Data (OHLCV + Tick)
| Source | Asset Classes | Latency | Notes |
|---|---|---|---|
| Polygon.io | US Equities, Options, Crypto | Real-time | Primary for US equities |
| Interactive Brokers API | All global | Real-time | Primary execution + data |
| Alpaca Markets | US Equities | Real-time | Backup + paper trading |
| CME Group Datamine | Futures (ES, NQ, CL, GC) | End-of-day / snapshot | Historical depth |
| OANDA / FXCM | Forex (40+ pairs) | Real-time | FX primary |
| Binance/Coinbase APIs | Crypto (top 50 pairs) | Real-time | L2 order book available |
| CBOE LiveVol | Options chain + IV surface | Real-time | Options microstructure |
| Tiingo | Equities (extended history) | EOD | Clean adjusted OHLCV |

#### Macroeconomic Data
| Source | Data Series | Cadence |
|---|---|---|
| FRED API | 800k+ series: CPI, PCE, yields, M2, PMI | On release |
| BLS API | NFP, unemployment, CPI components | Monthly |
| US Treasury Direct | Yield curve (1m to 30y) | Daily |
| IMF Data API | Global macro (GDP, trade balances) | Quarterly |
| OECD API | Leading indicators, CLI | Monthly |

#### Alternative Data
| Source | Signal Type | Cadence |
|---|---|---|
| Glassnode API | On-chain: SOPR, NVT, exchange flows, MVRV | Hourly |
| CryptoQuant | Exchange reserves, miner flows, funding rates | Real-time |
| Dune Analytics | DeFi TVL, protocol usage metrics | Hourly |
| Reddit (Pushshift) | WallStreetBets sentiment, mentions | Near real-time |
| StockTwits API | Ticker-level retail sentiment | Near real-time |
| SEC EDGAR (full-text) | Insider trading (Form 4), 13F filings | On filing |
| Quandl / Sharadar | Fundamentals: P/E, P/B, EV/EBITDA, FCF | Quarterly |
| Ortex / S3 Partners | Short interest, borrow rates | Daily |
| Open Insider | Insider transaction aggregates | Daily |

#### News & Events
| Source | Coverage | Processing |
|---|---|---|
| Bloomberg (if licensed) | Comprehensive financial news | FinBERT NLP |
| Alpha Vantage News | US equities news feed | FinBERT NLP |
| Benzinga API | Earnings, upgrades/downgrades | Event extraction |
| GDELT Project | Geopolitical events (free) | Tone scoring |
| Fed Calendar | FOMC dates, minutes, speeches | Event flag |
| Earnings Whispers | Earnings estimate consensus | Feature |

#### Order Book / Microstructure
| Data | Source | Use |
|---|---|---|
| Level 2 snapshots (top 10) | IBKR / Polygon | Order imbalance signal |
| Trade-level prints | Consolidated tape | Flow classification |
| Dark pool prints | Quandl Dark Pool Index | Institutional flow |
| Options flow | Unusual Whales / CBOE | Smart money signal |
| Funding rates | Binance / FTX archive | Crypto leverage signal |

### 2.2 Feature Engineering Strategy

All features are computed in a **point-in-time correct** manner — no future leakage. Every feature is tagged with its observation timestamp and lookahead window.

#### Technical / Price-Based Features (per asset, multi-timeframe)

```
Timeframes: 1m, 5m, 15m, 1h, 4h, 1D, 5D

Price Features:
  - Log returns at each timeframe
  - OHLC ratios: (Close-Open)/ATR, (High-Low)/ATR
  - VWAP deviation: (Price - VWAP) / σ_daily
  - Gap features: overnight gap, gap fill probability

Momentum:
  - RSI(7), RSI(14), RSI(21) — avoid overfitting to one period
  - MACD signal line, MACD histogram momentum
  - Rate of Change (ROC): 5, 10, 20, 60 bars
  - Williams %R, Stochastic Oscillator (5,3,3)
  - Cross-asset momentum: asset return vs. sector ETF return

Volatility:
  - ATR(14) normalized by price
  - Realized volatility: rolling 5D, 20D, 60D
  - GARCH(1,1) conditional volatility forecast
  - IV percentile rank (options-based where available)
  - Volatility of volatility (VoV)
  - Parkinson estimator: ln(High/Low)² / (4ln2)

Volume:
  - Volume ratio: current / 20D average volume
  - On-Balance Volume (OBV) slope
  - Volume-price correlation (20-bar rolling)
  - VWAP anchored to session open
  - Volume Profile: POC, VAH, VAL distances

Trend:
  - EMA stack: 8, 21, 50, 200 periods — slope and alignment score
  - ADX(14) — trend strength
  - Ichimoku cloud position (above/below span A/B)
  - Supertrend direction and distance
```

#### Microstructure Features

```
Order Book (Level 2):
  - Order imbalance ratio: (BidVol - AskVol) / (BidVol + AskVol)
  - Depth ratio: depth at best 5 levels bid vs. ask
  - Quote stuffing indicator: LOB update rate anomaly
  - Effective spread: 2 × |Trade Price - Midpoint|
  - Kyle's Lambda (price impact coefficient): ΔPrice / ΔOrderFlow

Trade Flow:
  - Buy/sell classification: Lee-Ready algorithm on tick data
  - VPIN (Volume-Synchronized Probability of Informed Trading)
  - Trade size distribution: avg, skew, kurtosis of trade sizes
  - Aggressive order ratio: market orders vs. limit orders
```

#### Statistical / Factor Features

```
Cross-Asset:
  - Rolling correlation: asset vs. SPX, TLT, DXY, VIX (20D, 60D)
  - Beta to market (60D rolling OLS)
  - Sector relative strength: asset vs. sector ETF (XLK, XLE, etc.)
  - Fama-French factor exposures: SMB, HML, RMW, CMA, WML loadings

Statistical Arbitrage:
  - Cointegration z-score (Engle-Granger): for pairs
  - Spread half-life (Ornstein-Uhlenbeck fit): θ = speed of reversion
  - Hurst exponent: < 0.5 = mean-reverting, > 0.5 = trending
  - Variance ratio (Lo-MacKinlay): tests for random walk

Regime Features:
  - HMM state probability vector (4 states)
  - VIX level and term structure slope (VIX9D / VIX3M)
  - Credit spread (HY-IG spread, IG-Treasury spread)
  - Yield curve: 2s10s slope, 3m10y slope (recession indicator)
  - Advance-Decline line, Put/Call ratio
```

#### NLP / Sentiment Features

```
News-based:
  - FinBERT sentiment score: positive / negative / neutral probability
  - Sentiment momentum: 5D EMA of daily sentiment vs. 20D EMA
  - News volume spike: current volume vs. 30D average
  - Earnings surprise magnitude (actual vs. consensus EPS)
  - Analyst revision momentum: upgrade/downgrade ratio (30D)

Social:
  - Reddit mention volume (normalized by post volume)
  - StockTwits bull/bear ratio (1D, 5D)
  - Unusual social spike detector: z-score of mention volume

On-Chain (Crypto):
  - MVRV Z-score: (MarketCap - RealizedCap) / σ(MarketCap)
  - SOPR: Spent Output Profit Ratio (> 1 = sellers in profit)
  - NVT Signal: 90D MA of NetworkValue/TransactionVolume
  - Exchange netflow: net BTC/ETH flowing to/from exchanges
  - Funding rate: 8hr perpetual swap funding (positive = longs paying)
  - Stablecoin supply ratio: USDT supply / BTC MarketCap
```

### 2.3 Data Quality & Storage Architecture

```
Raw Ingestion → Data Lake (S3/MinIO, Parquet format, partitioned by date+symbol)
    ↓
Validation Layer:
  • Schema enforcement (Great Expectations)
  • Staleness detection (alert if feed age > threshold)
  • Outlier detection: price jumps > 5σ flagged for review
  • Adjusted vs. unadjusted price tracking (corporate actions)
  • Point-in-time snapshot preservation (no overwrites)
    ↓
Feature Store:
  • Offline: ClickHouse (fast column queries for training)
  • Online: Redis (< 10ms feature retrieval for inference)
  • Feast manages consistency between both
    ↓
Model Training Store:
  • MLflow artifacts (model weights, configs, metrics)
  • Versioned datasets with timestamp ranges
```

---

## 3. Modeling Approach

### 3.1 Model Zoo — 7 Families

#### Family 1: Statistical Baselines (Speed: instant, Interpretability: high)

```python
# ARIMA for detrended return series
# Use ADF test first to confirm stationarity
# SARIMA for assets with seasonal components (commodities)

# Kalman Filter — dynamic estimate of "fair value"
# State: [price, velocity, acceleration]
# Observation: raw price
# Use: smooth noisy prices, detect deviations

# Vector Autoregression (VAR) — cross-asset dynamics
# Assets: [SPX, TLT, DXY, Gold, Oil] — macro basket
# Use: capture lead-lag relationships

# Ornstein-Uhlenbeck for mean-reversion pairs
# dX = θ(μ - X)dt + σdW
# θ: speed of reversion (half-life = ln(2)/θ)
# Signal: z-score of spread vs. OU equilibrium
```

**Purpose**: Provides interpretable baseline, sanity check on ML signals.  
**Prediction horizon**: Intraday to 1-week.

#### Family 2: Gradient Boosting (Core Alpha Engine)

```
Model: LightGBM (primary), XGBoost (ensemble member)
Input: 500+ engineered features across all categories above
Output: Predicted return percentile (0–1) at each horizon
Horizons: 1h, 4h, 1D, 5D (separate models per horizon)

Training:
  - Rolling window: 2 years training, 3 months validation
  - Walk-forward: retrain monthly
  - Purged K-Fold cross-validation (embargo = 2 × horizon)

Regularization:
  - L1/L2 regularization (lambda_l1, lambda_l2)
  - min_data_in_leaf ≥ 200 (prevents memorizing specific dates)
  - Feature subsampling (colsample_bytree = 0.7)
  - Monotonic constraints on features with known directionality

Interpretability:
  - SHAP values per prediction (feature attribution)
  - Feature importance tracked over time (detect signal decay)
  - TreeExplainer for fast SHAP computation at inference time
```

#### Family 3: Deep Learning — Temporal Models

```
A) Temporal Fusion Transformer (TFT)
   Architecture: Variable selection networks + LSTM encoder + 
                 multi-head attention + quantile output heads
   Input: 120 timesteps of technical features + static metadata
   Output: P10, P50, P90 quantile forecasts at 5 horizons
   Why TFT: Handles mixed-frequency inputs, provides uncertainty bounds,
             interprets temporal patterns via attention weights

B) Bidirectional LSTM with Attention
   Architecture: 3-layer BiLSTM → Multi-head self-attention → FC head
   Input: 60-bar normalized OHLCV + top 20 technical features
   Dropout: 0.3 (variational dropout for uncertainty estimation)
   Output: Directional probability + expected magnitude

C) WaveNet-style Dilated CNN
   Architecture: Dilated causal convolutions (rates: 1,2,4,8,16,32,64)
   Receptive field: 128 bars without exponential parameter growth
   Use case: Detects cyclical / seasonal patterns at multiple frequencies

Training:
  - PyTorch Lightning (training loop standardization)
  - Mixed precision (fp16) training on GPU
  - Early stopping on validation Sharpe (NOT validation loss)
  - Batch size: 256, learning rate warmup + cosine annealing
```

#### Family 4: NLP Models (Event-Driven Alpha)

```
A) FinBERT (ProsusAI/finbert from HuggingFace)
   Fine-tuned on: Financial PhraseBank + proprietary labeled corpus
   Input: News headline + first 512 tokens of article body
   Output: Sentiment probability (positive/negative/neutral)
   Serving: TorchServe, GPU inference, batched for throughput

B) Named Entity Recognition (NER)
   Model: spaCy large model + custom financial entity types
   Entities: COMPANY, TICKER, PERSON, EVENT, INDICATOR
   Use: Link news to specific tickers, filter irrelevant articles

C) Event Extraction (Earnings / M&A / Macro)
   Model: Fine-tuned BERT classifier on event type
   Event types: Earnings beat/miss, acquisition, dividend change,
                FDA approval, Fed pivot, geopolitical escalation
   Output: Event type + sentiment + impact estimate
```

#### Family 5: Graph Neural Network (Cross-Asset Intelligence)

```
Graph Construction:
  - Nodes: 500 assets (S&P 500 universe)
  - Edges: Dynamic correlation threshold (|ρ| > 0.4 over 60D)
  - Edge weights: |correlation| value
  - Additional edges: supply chain relationships (from SEC 10-K parsing)

Architecture: GraphSAGE (inductive learning — handles new nodes)
  - 3 aggregation layers
  - Mean aggregation (robust to missing neighbors)
  - Node features: per-asset technical + fundamental feature vector
  - Output: Refined node embeddings used as additional features in meta-learner

Use case:
  - Detect contagion: if sector leader shows stress, propagate signal
  - Cross-asset spillover: commodity shock → energy equity impact
  - Update: Graph reconstructed daily with fresh correlations
```

#### Family 6: Reinforcement Learning Agent

```
Environment: RealisticTradingEnv (custom Gym environment)
  - State space: All features from above (normalized, ~200-dim vector)
  - Action space: Continuous [-1.0, +1.0] per asset (position fraction)
  - Reward function: Sharpe ratio over trailing 20 steps
                     + Penalty: -0.01 × |ΔPosition| (transaction costs)
                     + Penalty: -0.05 if drawdown > 5% from peak
  - Episode: 252 trading days (1 year)
  - Realistic simulation: variable slippage, bid-ask spread sampling

Algorithm: Soft Actor-Critic (SAC)
  - Off-policy: sample efficiency from replay buffer
  - Maximum entropy RL: encourages exploration, prevents overfit
  - Twin Q-networks: reduces overestimation bias
  - Automatic entropy tuning: α adapts to target entropy

Training:
  - 10,000 training episodes on historical data
  - Curriculum learning: start with low volatility periods
  - Domain randomization: vary transaction costs, liquidity
  - Evaluation: separate 2-year out-of-sample test period

Role in system: RL agent is ONE vote in the ensemble, not the sole decision maker.
               Useful for learning position sizing and regime transitions.
```

### 3.2 Ensemble Meta-Learner Architecture

```
Layer 0 (Base Models):
  - ARIMA/OU signals        → scalar(s)
  - LightGBM (1h, 4h, 1D)  → return percentile per horizon
  - TFT                     → quantile forecasts (P10, P50, P90)
  - BiLSTM                  → directional probability
  - FinBERT sentiment       → sentiment score
  - GNN embedding           → asset embedding vector
  - RL agent action         → position recommendation

Layer 1 (Stacking Meta-Learner):
  - Model: LightGBM (fast, handles non-linear interactions between base signals)
  - Input: All Layer 0 outputs + current regime state + market conditions
  - Training: 60/20/20 temporal split (NOT random — time respects causality)
  - Output: Final signal strength (-1 to +1) + confidence score (0 to 1)

Regime-Conditional Weighting (overrides meta-learner for robustness):
  Regime 1 (Low-Vol Trend):   Weight GBM 35%, TFT 30%, RL 20%, Stat 15%
  Regime 2 (High-Vol Trend):  Weight GBM 25%, RL 35%, Stat 25%, TFT 15%
  Regime 3 (Mean-Reverting):  Weight Stat 40%, GBM 30%, TFT 20%, RL 10%
  Regime 4 (Crisis):          Weight Stat 50%, VIX-hedge signals 30%, GBM 20%
                              → Force net delta close to 0, increase cash

Disagreement Signal:
  σ(model predictions) → high σ = low confidence = reduced position sizing
  Threshold: if σ > 0.4, cap position at 50% of calculated size
```

---

## 4. Decision Engine

### 4.1 Trade Scoring

Every candidate trade receives a composite score before execution:

```
TRADE_SCORE = (
    0.35 × AlphaScore        # Model signal strength × confidence
  + 0.25 × RegimeScore       # Signal validity in detected regime
  + 0.20 × LiquidityScore    # Market impact relative to ADV
  + 0.20 × RiskScore         # Marginal VaR contribution (inverted)
)

AlphaScore     = MetaLearner_signal × MetaLearner_confidence
RegimeScore    = Probability(regime matches signal type)
LiquidityScore = 1 - MarketImpact / ExpectedAlpha
                 MarketImpact = η × σ × √(Size / ADV)  [Almgren-Chriss]
RiskScore      = 1 - (ΔPortfolioVaR / VaR_budget_per_trade)

Execution threshold: TRADE_SCORE ≥ 0.62 (tunable)
Position sizing begins only above this threshold.
```

### 4.2 Position Sizing Model

```
Primary: Fractional Kelly Criterion
  f* = (μ_predicted - r_f) / σ²_predicted
  f_actual = 0.25 × f*   (quarter-Kelly for safety; prevents ruin from estimation error)

Adjustment factors (multiplicative):
  × Confidence factor:   MetaLearner_confidence (0 to 1)
  × Regime factor:       0.5 in Crisis, 0.75 in High-Vol, 1.0 in Low-Vol
  × Drawdown factor:     1.0 - (CurrentDrawdown / MaxAllowedDrawdown)
  × Liquidity factor:    min(1.0, ADV × 0.01 / TargetNotional)
                         (cap at 1% of average daily volume)

Hard caps:
  - Single position: max 10% of NAV
  - Sector: max 30% gross exposure
  - Geography: max 40% in any single country
  - Net beta: -0.3 to +0.3 (market-neutral target)
  - Gross leverage: max 2.5× NAV (futures-adjusted)
```

### 4.3 Risk Engine (Real-Time)

```
VaR Calculation (3 methods, take conservative estimate):
  1. Historical Simulation: 500-day lookback, 99th percentile 1-day VaR
  2. Parametric: VaR = Position × σ × z_{99%} (assumes normality — for sanity check only)
  3. Monte Carlo: 10,000 scenarios using DCC-GARCH covariance matrix

CVaR (Conditional VaR = Expected Shortfall):
  CVaR_{99%} = E[Loss | Loss > VaR_{99%}]
  Used as PRIMARY constraint in portfolio optimization (more tail-sensitive)

Dynamic Correlations:
  DCC-GARCH(1,1) — Dynamic Conditional Correlation model
  Q_t = (1 - a - b)Q̄ + a(ε_{t-1}ε'_{t-1}) + bQ_{t-1}
  Updated daily; covariance matrix fed to portfolio optimizer

Drawdown Control Logic:
  DD_current = (NAV_peak - NAV_current) / NAV_peak

  IF DD_current > 5%:  Reduce all positions by 20%
  IF DD_current > 10%: Reduce all positions by 50%, halt new entries
  IF DD_current > 15%: Full liquidation to cash, manual review required
  IF DD_current > 20%: System halted, human override required

Concentration Risk:
  - No two positions with |ρ| > 0.7 each exceeding 5% weight
  - Factor risk: max 30% of portfolio variance from any single factor
  - Sector VaR contribution: alert if any sector > 40% of total portfolio VaR
```

### 4.4 Portfolio Construction (Black-Litterman)

```
Standard Markowitz fails in practice due to estimation error amplification.
We use Black-Litterman (BL) to blend market equilibrium with model views.

Step 1: Market Equilibrium Returns
  Π = δ × Σ × w_market
  δ = risk aversion coefficient (calibrated to market Sharpe)
  Σ = DCC-GARCH covariance matrix
  w_market = market cap weights (from universe)

Step 2: Model Views (P, Q, Ω)
  P = pick matrix (which assets each view references)
  Q = vector of expected returns per view (from meta-learner)
  Ω = uncertainty of each view (diagonal: σ² of model prediction errors)

Step 3: BL Posterior
  E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} × [(τΣ)^{-1}Π + P'Ω^{-1}Q]
  Posterior covariance adjusts for view uncertainty

Step 4: Mean-Variance Optimization with constraints
  Solver: Clarabel (open-source SOCP solver, fast, no license fees)
  Objective: Maximize Sharpe (return / σ) of portfolio
  Constraints:
    - CVaR_{99%} ≤ daily_risk_budget (e.g., 0.5% of NAV)
    - w_i ∈ [-0.10, +0.10] for each asset
    - Σ|w_i| ≤ 2.5 (gross leverage cap)
    - Turnover constraint: Σ|Δw_i| ≤ 0.20 per day (transaction cost control)

Step 5: Transaction Cost Awareness
  Net expected return adjusted: μ_net = μ_BL - λ × Σ|Δw_i| × TradingCost_i
  Optimizer sees net-of-cost returns, naturally reducing unnecessary turnover
```

### 4.5 Execution Layer

```
Order Management System (OMS):
  - Tracks all open orders, positions, fills, PnL in real-time (Redis)
  - Order states: PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED / CANCELLED
  - Position reconciliation against broker every 60 seconds

Smart Order Routing:
  - Small orders (< 1% ADV): Market order or aggressive limit (1-tick through best)
  - Medium orders (1-5% ADV): VWAP algorithm over 30 min to 2 hours
  - Large orders (> 5% ADV): TWAP over full session + dark pool routing
  - Crypto: Route through 3 exchanges simultaneously for best fill

Broker APIs:
  - Primary: Interactive Brokers (TWS API) — all asset classes
  - US Equity backup: Alpaca Markets (REST + WebSocket)
  - Crypto: CCXT unified library (Binance, Coinbase, Kraken)
  - Options: IBKR native options chain access

Latency targets:
  - Signal → order generation: < 100ms
  - Order → broker submission: < 50ms
  - Not competing with HFT; targeting alpha half-lives of hours to days
```

---

## 5. Infrastructure & Tech Stack

### 5.1 Full Stack Summary

| Layer | Technology | Justification |
|---|---|---|
| Language (ML) | Python 3.11 | Ecosystem depth, library support |
| Language (execution) | Python (Rust optional phase 2) | Speed sufficient for target latency |
| Message bus | Apache Kafka | Industry standard, durable, replay |
| Stream processing | Bytewax (Python-native Flink) | Python-native, lower ops overhead than Java Flink |
| Time-series DB | TimescaleDB (PostgreSQL extension) | SQL-compatible, mature, compression |
| OLAP / feature store offline | ClickHouse | 10–100× faster than PostgreSQL for analytical queries |
| Feature store | Feast | Open-source, production-grade, Redis + offline sync |
| Data Lake | MinIO + Parquet | S3-compatible, on-prem capable, columnar efficiency |
| Cache | Redis 7 | Sub-millisecond feature retrieval |
| ML framework | PyTorch 2.x + Lightning | De facto standard for DL research + production |
| Traditional ML | LightGBM, XGBoost, scikit-learn | Fast, battle-tested tree models |
| RL | Stable-Baselines3, RLlib | SAC implementation + distributed training |
| NLP | HuggingFace Transformers | Access to FinBERT, custom fine-tuning |
| GNN | PyTorch Geometric | GraphSAGE, DGL alternative |
| Experiment tracking | MLflow | Model registry, artifact store, metrics |
| Hyperparameter opt. | Optuna | TPE sampler, pruning, distributed |
| Model serving | BentoML | Python-native, fast deployment |
| Workflow orchestration | Apache Airflow 2.x | DAG management for batch pipelines |
| Portfolio solver | Clarabel (via CVXPY) | SOCP solver, no commercial license |
| Containerization | Docker + Docker Compose | Dev; Kubernetes for production scale |
| Monitoring | Prometheus + Grafana | System metrics, custom PnL dashboards |
| Data validation | Great Expectations | Schema enforcement, statistical checks |
| Drift detection | Evidently AI | Feature drift, model performance drift |
| Alerting | PagerDuty / Telegram bot | Critical system and risk alerts |
| Version control | Git + DVC | Code + data versioning |

### 5.2 Hardware Requirements (Minimum Viable)

```
Development / Research Machine:
  CPU: 16+ cores (AMD EPYC or Intel Xeon)
  RAM: 64 GB (128 GB preferred for large feature matrices)
  GPU: NVIDIA RTX 4090 or A100 (for DL training)
  Storage: 4 TB NVMe SSD (tick data storage)
  Network: 1 Gbps (sufficient for non-HFT)

Production Deployment (Cloud: AWS / GCP recommended):
  Data ingestion: 2× c6i.2xlarge (8 vCPU, 16 GB RAM)
  Feature computation: 2× r6i.4xlarge (16 vCPU, 128 GB RAM)
  ML inference: 1× g4dn.xlarge (T4 GPU, 16 GB VRAM)
  Model training: 1× p3.2xlarge (V100, on-demand for weekly retraining)
  Database: db.r6g.2xlarge for TimescaleDB RDS
  Total estimated cost: ~$800–1,500/month on AWS
```

### 5.3 Network & Security

- All API keys stored in HashiCorp Vault (not in code or env files)
- TLS 1.3 on all external connections
- VPN / private subnet for broker API communication
- Rate limiter middleware on all data ingestion services (avoid API bans)
- Circuit breaker pattern: automatic pause if data feed anomaly detected

---

## 6. Backtesting & Evaluation Framework

### 6.1 Core Design Principles

**Rule 1 — Point-in-Time Correctness**  
Every signal uses only data that was observable at the exact moment it would have been generated. This means:
- Adjusted price data tagged with adjustment date (no look-ahead adjustment)
- Earnings data used only after public release timestamp
- Macro data used only after official release (e.g., NFP at 8:30 AM, not 9:00 AM)
- Model predictions trained only on data available before the prediction timestamp

**Rule 2 — No Survivorship Bias**  
Universe construction includes all tickers that were in the index at the time, including companies that later went bankrupt or were acquired.

**Rule 3 — Realistic Transaction Costs**

| Asset | Cost Model |
|---|---|
| US Equities | 0.5 bps commission + Almgren-Chriss market impact |
| Futures | $2.25/contract + 0.25 tick slippage |
| Forex | 0.5 pip spread (liquid pairs), 2 pip (exotic) |
| Crypto | 8 bps taker + 0.1% slippage on 0.5% ADV orders |
| Options | $0.65/contract + bid-ask mid slippage |

**Rule 4 — Event-Driven Architecture**  
Custom event-driven backtester (not vectorized) processes bar-by-bar:
1. Receive new bar event
2. Update feature store
3. Run model inference
4. Check risk constraints
5. Generate orders if signal crosses threshold
6. Simulate fill with realistic slippage model
7. Update portfolio state

### 6.2 Walk-Forward Methodology

```
Expanding Window Walk-Forward:

Train: [2015-01-01 → 2019-12-31]  Test: [2020-01-01 → 2020-03-31]
Train: [2015-01-01 → 2020-03-31]  Test: [2020-04-01 → 2020-06-30]
Train: [2015-01-01 → 2020-06-30]  Test: [2020-07-01 → 2020-09-30]
... (advance quarterly, retrain monthly in production)

Purged K-Fold (for cross-validation during training):
  Embargo period = 2 × prediction_horizon  (prevents label leakage)
  Example for 1D horizon: embargo = 2 trading days on either side of split

  Fold 1: Train [Jan-Sep], PURGE [Sep 28 - Oct 2], Test [Oct-Dec]
  Fold 2: Train [Jan-Jun, Nov-Dec], PURGE, Test [Jul-Sep]
  ... k=5 folds minimum, k=10 preferred
```

### 6.3 Overfitting Detection — Critical Methods

```
1. Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2016)
   Adjusts for: number of trials, non-normality of returns, autocorrelation
   DSR = SR × [1 - γ × log(T) / (T-1)]^{0.5}
   Threshold: DSR ≥ 1.0 for strategy to be considered non-random

2. Combinatorial Purged Cross-Validation (CPCV)
   Tests all N!/(k!(N-k)!) combinations of train/test splits
   Produces distribution of Sharpe ratios (not just single estimate)
   Red flag: if median CPCV Sharpe << in-sample Sharpe by > 50%

3. Backtest Overfitting Probability (BOP) — CSCV Method
   Run 1,000+ simulations shuffling train/test assignments
   BOP = fraction where random split outperforms original split
   Acceptable: BOP < 0.10 (< 10% chance overfitting explains results)

4. Minimum Backtest Length (MinBTL)
   MinBTL = (8.4 × σ_SR) / SR_min  (months of data required)
   Ensures statistical significance given signal-to-noise ratio

5. Feature Stability Index (FSI)
   Monitor whether feature importance rankings are stable across folds
   High FSI variability = model is fitting noise, not signal
```

### 6.4 Performance Evaluation Metrics

```
Return Metrics:
  Annualized Return (CAGR)
  Sharpe Ratio = (R_p - R_f) / σ_p × √252
  Sortino Ratio = (R_p - R_f) / σ_downside × √252  [penalizes only downside vol]
  Calmar Ratio = CAGR / |MaxDrawdown|
  Omega Ratio = E[max(R-L,0)] / E[max(L-R,0)]  [L = target return threshold]

Risk Metrics:
  Maximum Drawdown (MDD): peak-to-trough
  Average Drawdown Duration (recovery time)
  VaR 95%, VaR 99% (1-day, 10-day)
  CVaR 99% (Expected Shortfall)
  Tail Ratio: 95th percentile return / |5th percentile return|

Statistical Validity:
  t-statistic on annualized alpha (must be > 3.0 for credibility)
  p-value < 0.01
  Minimum 3 years of OOS data preferred

Signal Quality:
  Hit Rate (% winning trades)
  Profit Factor = Gross Profits / Gross Losses (target > 1.5)
  Information Coefficient (IC) = Spearman rank correlation(signal, forward return)
    Acceptable IC range: 0.03 – 0.08 (institutional strategies)
  ICIR = Mean(IC) / Std(IC) (IC Information Ratio, target > 0.5)
  Signal turnover: daily position change as % of portfolio

Benchmark Comparisons:
  Alpha vs. SPX (beta-adjusted)
  Benchmark: AQR Managed Futures style benchmark
  Tracking error (for market-neutral strategies: should be high vs. SPX)
```

### 6.5 Paper Trading Validation Gate

Before any live capital deployment:
1. 90 days of paper trading (no capital at risk)
2. Slippage attribution: compare simulated fills to actual market prices
3. Signal distribution comparison: live signal histogram vs. backtest histogram
4. Performance attribution: how much of alpha is explained by beta? factor exposures?
5. Drawdown stress test: inject historical crisis periods (2008, 2020 COVID, 2022 rates)
6. Latency profiling: measure actual signal → order → fill latency
7. Manual review of 20 random trade decisions for logic validity

---

## 7. Risks, Limitations & Realistic Expectations

### 7.1 Where the System Will Fail

**Data Quality Failures**
- Vendor outages (Polygon.io downtime, FRED API delays) → stale features → false signals
- Tick data gaps on illiquid instruments generate phantom volatility spikes
- Corporate actions (splits, mergers, spin-offs) corrupt adjusted price history if not handled at the vendor level — require manual audit pipeline
- NLP models misclassify domain-specific jargon ("hot" = positive for tech, negative for macro)

**Model Degradation**
- Concept drift: regime shifts (e.g., zero-rate → high-rate environment in 2022) invalidate models trained on prior regimes
- Crowded trades: when multiple quant funds run similar models, the alpha decays because everyone front-runs each other's entries and exits
- Non-stationarity of financial time series fundamentally limits forecast accuracy — the ground truth generating process itself changes
- Feature importance shifts: a feature that was predictive in 2019 may be noise in 2025

**Execution Gaps**
- Backtest assumes fills at model-close prices; live fills are worse due to information asymmetry and adverse selection
- Large orders on illiquid assets move the market against us (market impact not fully capturable in backtest)
- Exchange outages (crypto especially), broker API failures, rate limiting are routine operational risks
- Latency spikes during high-volatility events cause missed fills on time-sensitive signals

**Black Swan / Tail Risk**
- COVID (March 2020), Flash Crash (May 2010): all VaR models failed in the tails
- Correlation breakdown: in crisis, normally uncorrelated assets become correlated (liquidity crisis drives simultaneous selling of everything)
- Liquidity vacuum events: bid-ask spreads widen 10–100× making exit impossible at model price
- Regulatory actions: sudden restrictions on short selling, position limits, circuit breakers

**Structural Risks**
- HFT front-running: large orders in illiquid markets can be detected and traded against
- Data vendor lock-in: proprietary data sources can reprice or restrict access
- API rate limits: news and alternative data APIs frequently throttle high-volume requests
- Model monoculture: if retraining is automated and a bad data batch occurs, model degrades silently

### 7.2 Realistic Performance Expectations

```
IMPORTANT: These are honest ranges, not marketing projections.

Research Backtest (optimistic, in-sample):
  Sharpe: 2.0 – 4.0    ← mostly noise from overfitting
  CAGR:   25 – 60%
  MDD:    5 – 15%
  ⚠️  Do not trust these numbers. They are in-sample artifacts.

Walk-Forward OOS Backtest (credible estimate):
  Sharpe: 0.8 – 1.8
  CAGR:   12 – 30%
  MDD:    10 – 25%
  Hit Rate: 52 – 57%

Live Paper Trading (realistic):
  Sharpe: 0.6 – 1.4    ← further degradation from execution friction
  CAGR:   8 – 20%
  MDD:    15 – 30%

Live Capital (realistic, year 1–2):
  Sharpe: 0.4 – 1.2
  CAGR:   6 – 18%
  MDD:    15 – 35%
  ⚠️  Expect worse than paper trading due to psychological effects,
      capital constraints, and unmodeled market impact.

Industry Context:
  - Top quant hedge funds (Renaissance, Two Sigma): Sharpe 2.0–5.0 after fees
  - Mid-tier systematic funds: Sharpe 0.8–1.5 after fees
  - This system (well-executed): Sharpe 0.6–1.2 live, year 1
  - Signal half-life for most ML signals: 6–18 months before alpha decays
```

### 7.3 Key Risk Priorities (by severity)

| Rank | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Overfitting mistaken for skill | Very High | Catastrophic | CPCV, DSR, min 3yr OOS |
| 2 | Transaction costs exceed alpha | High | Severe | Cost-aware optimization, low turnover |
| 3 | Data vendor outage in live | Medium | High | Redundant feeds, staleness alerts |
| 4 | Black swan drawdown | Low | Catastrophic | Hard 15% DD stop, tail hedges |
| 5 | Model drift (concept drift) | High | Moderate | Monthly retraining, drift detection |
| 6 | Crowded trade decay | Medium | High | Signal diversification, novelty monitoring |
| 7 | Execution slippage > model | Medium | Moderate | Paper trading gate, slippage attribution |
| 8 | Regulatory change | Low | High | Monitor CFTC/SEC, position limit buffers |

---

## Implementation Readiness Checklist (Phase 2 Prerequisites)

Before proceeding to Phase 2 implementation, the following must be decided:

- [ ] Asset universe: Which markets to trade first? (Recommended: US Equities → Crypto → Futures)
- [ ] Capital allocation: Starting capital size (affects position sizing, data cost justification)
- [ ] Data budget: Which data sources can be afforded? (Free tier vs. paid APIs)
- [ ] Execution mode: Paper trading only (Phase 2) or live from the start?
- [ ] Hosting: Cloud (AWS/GCP) or local machine?
- [ ] Time horizon focus: Intraday (harder) vs. Swing (1–5 days, recommended to start)
- [ ] Regulatory jurisdiction: US (SEC/FINRA rules), EU (MiFID II), or crypto-first?

---

*Document Version: 1.0 | Status: Awaiting Approval | Phase 1 Complete*
