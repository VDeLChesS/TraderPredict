# TraderPredict — Phase 2: Implementation Task Breakdown

> **Status:** Approved for Implementation  
> **Methodology:** Each task is atomic, independently testable, and logically ordered.  
> **Naming Convention:** `[MODULE]-[SPRINT]-[SEQ]` — e.g., `INFRA-01-001`

---

## MASTER EXECUTION ORDER

```
SPRINT 01 — Foundation & Infrastructure
SPRINT 02 — Data Ingestion Layer
SPRINT 03 — Feature Engineering Pipeline
SPRINT 04 — Model Development (Statistical + ML)
SPRINT 05 — Deep Learning & NLP Models
SPRINT 06 — Signal Aggregation & Ensemble
SPRINT 07 — Strategy & Decision Engine
SPRINT 08 — Risk Management & Portfolio Construction
SPRINT 09 — Backtesting Framework
SPRINT 10 — Execution Layer & OMS
SPRINT 11 — Monitoring & Feedback Loop
SPRINT 12 — Live Simulation & Hardening
```

---

## SPRINT 01 — Foundation & Infrastructure

**Goal:** Establish the project skeleton, environment, and core infrastructure so every subsequent module has a stable base to build on.

---

### INFRA-01-001 — Project Repository Structure
**What:** Create the monorepo directory layout and base configuration files.  
**Deliverable:** Folder tree committed to Git with `pyproject.toml`, `.env.example`, `docker-compose.yml` skeleton, and `Makefile`.  
**Test:** `make lint` and `make test` run without errors on an empty project.  
**Dependencies:** None.

```
traderpredivt/
├── services/
│   ├── data_service/
│   ├── feature_service/
│   ├── model_service/
│   ├── signal_service/
│   ├── risk_service/
│   ├── decision_engine/
│   └── execution_service/
├── shared/
│   ├── schemas/          # Pydantic models shared across services
│   ├── utils/
│   └── config/
├── infra/
│   ├── kafka/
│   ├── redis/
│   └── postgres/
├── notebooks/            # Research & experimentation
├── backtesting/
├── monitoring/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── system/
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

### INFRA-01-002 — Python Environment & Dependency Management
**What:** Set up `pyproject.toml` with all core dependencies using `uv` for fast installs.  
**Deliverable:** Locked `requirements.txt` / `uv.lock`. CI installs succeed in under 60 seconds.  
**Test:** `python -c "import pandas, numpy, torch, kafka, redis, fastapi"` succeeds in a fresh virtualenv.  
**Dependencies:** INFRA-01-001.

**Core packages to pin:**
```toml
[project]
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "polars>=0.20",           # fast columnar ops
    "torch>=2.2",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "xgboost>=2.0",
    "transformers>=4.40",     # HuggingFace
    "kafka-python>=2.0",
    "redis>=5.0",
    "fastapi>=0.110",
    "pydantic>=2.6",
    "feast>=0.38",
    "mlflow>=2.12",
    "bytewax>=0.19",
    "cvxpy>=1.4",
    "clarabel>=0.7",
    "vectorbt>=0.26",
    "backtrader>=1.9",
    "httpx>=0.27",
    "loguru>=0.7",
    "prometheus-client>=0.20",
]
```

---

### INFRA-01-003 — Docker Compose Local Stack
**What:** Define `docker-compose.yml` for local development with Kafka, Zookeeper, Redis, PostgreSQL, ClickHouse, and MLflow.  
**Deliverable:** Single `docker compose up -d` brings up all services within 2 minutes.  
**Test:** Health-check endpoints respond on all expected ports. Run `make stack-up && make stack-health`.  
**Dependencies:** INFRA-01-001.

**Services to define:**
```yaml
# Ports to expose:
# Kafka:       9092
# Zookeeper:   2181
# Redis:       6379
# PostgreSQL:  5432  (operational metadata)
# ClickHouse:  8123  (offline feature store)
# MLflow:      5000
# Grafana:     3000
# Prometheus:  9090
```

---

### INFRA-01-004 — Shared Schema Library (Pydantic)
**What:** Define all internal data transfer objects (DTOs) as Pydantic v2 models in `shared/schemas/`.  
**Deliverable:** Schema classes with validation, serialization, and JSON Schema export.  
**Test:** Unit tests cover field validation edge cases (nulls, wrong types, boundary values) for all schemas.  
**Dependencies:** INFRA-01-002.

**Schemas to implement:**
```python
# shared/schemas/market.py
class OHLCVBar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource          # enum: POLYGON, ALPACA, YFINANCE, etc.

class TickData(BaseModel):
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: Literal["bid", "ask", "trade"]

class OrderBookSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]   # (price, size)
    asks: list[tuple[float, float]]

# shared/schemas/signal.py
class TradeSignal(BaseModel):
    symbol: str
    timestamp: datetime
    direction: Literal["long", "short", "flat"]
    confidence: float               # 0.0–1.0
    model_id: str
    raw_scores: dict[str, float]    # per-model contributions
    regime: str                     # detected market regime
    horizon: str                    # "1d", "1w", etc.

# shared/schemas/risk.py
class PositionRisk(BaseModel):
    symbol: str
    notional: float
    var_95: float
    cvar_95: float
    beta: float
    current_drawdown: float

# shared/schemas/portfolio.py
class PortfolioOrder(BaseModel):
    symbol: str
    direction: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit", "stop"]
    limit_price: Optional[float]
    reasoning: str
```

---

### INFRA-01-005 — Logging, Config, and Observability Skeleton
**What:** Implement centralized logging with `loguru`, config management with `pydantic-settings`, and Prometheus metrics stub.  
**Deliverable:** `shared/config/settings.py` loads from environment; every service uses the same logger pattern.  
**Test:** Change an env var and confirm the config reloads correctly. Prometheus `/metrics` endpoint returns 200.  
**Dependencies:** INFRA-01-004.

```python
# shared/config/settings.py
class Settings(BaseSettings):
    # Data Sources
    polygon_api_key: SecretStr
    alpaca_api_key: SecretStr
    fred_api_key: SecretStr
    news_api_key: SecretStr
    
    # Infrastructure
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379"
    clickhouse_url: str = "http://localhost:8123"
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Trading
    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.05
    max_drawdown_limit: float = 0.15
    
    model_config = SettingsConfigDict(env_file=".env", secrets_dir="/run/secrets")
```

---

### INFRA-01-006 — Kafka Topic Initialization
**What:** Define and create all Kafka topics with correct partition counts and retention policies.  
**Deliverable:** `infra/kafka/topics.py` script that idempotently creates all topics.  
**Test:** Topics exist post-creation; producing and consuming a test message succeeds on each topic.  
**Dependencies:** INFRA-01-003.

**Topics to create:**
```python
TOPICS = {
    "market.ticks":           {"partitions": 12, "retention_ms": 86_400_000},     # 1 day
    "market.ohlcv.1m":        {"partitions": 6,  "retention_ms": 604_800_000},    # 7 days
    "market.ohlcv.5m":        {"partitions": 6,  "retention_ms": 604_800_000},
    "orderbook.snapshots":    {"partitions": 12, "retention_ms": 3_600_000},      # 1 hour
    "macro.updates":          {"partitions": 2,  "retention_ms": 2_592_000_000},  # 30 days
    "news.raw":               {"partitions": 4,  "retention_ms": 604_800_000},
    "news.sentiment":         {"partitions": 4,  "retention_ms": 604_800_000},
    "features.realtime":      {"partitions": 6,  "retention_ms": 86_400_000},
    "signals.raw":            {"partitions": 4,  "retention_ms": 604_800_000},
    "signals.composite":      {"partitions": 2,  "retention_ms": 604_800_000},
    "orders.proposed":        {"partitions": 2,  "retention_ms": 604_800_000},
    "orders.executed":        {"partitions": 2,  "retention_ms": 2_592_000_000},
    "risk.alerts":            {"partitions": 2,  "retention_ms": 2_592_000_000},
}
```

---

## SPRINT 02 — Data Ingestion Layer

**Goal:** Build reliable, multi-source data connectors that feed normalized data into Kafka and the offline store.

---

### DATA-02-001 — Market Data Connector: Polygon.io (Historical)
**What:** Implement a Python client for Polygon.io REST API to fetch historical OHLCV bars for equities.  
**Deliverable:** `services/data_service/connectors/polygon_historical.py` with rate-limit handling and retry logic.  
**Test:** Fetch 1 year of daily OHLCV for `["AAPL", "MSFT", "SPY"]`. Validate schema, no NaNs in OHLCV fields, timestamps monotonically increasing.  
**Dependencies:** INFRA-01-004, INFRA-01-005.

```python
class PolygonHistoricalConnector:
    def __init__(self, api_key: str, rate_limit_per_min: int = 5):
        ...
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        timespan: Literal["minute", "hour", "day"] = "day",
        multiplier: int = 1,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars with automatic pagination and retry."""
        ...
    
    async def fetch_batch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timespan: str = "day",
    ) -> dict[str, list[OHLCVBar]]:
        """Concurrent fetch across symbols with semaphore throttling."""
        ...
```

---

### DATA-02-002 — Market Data Connector: Alpaca WebSocket (Real-Time)
**What:** Implement a WebSocket client for Alpaca's real-time market data stream. Publishes ticks to `market.ticks` Kafka topic.  
**Deliverable:** `services/data_service/connectors/alpaca_stream.py` — async listener that auto-reconnects on drop.  
**Test:** Subscribe to 5 symbols for 60 seconds. Assert ≥ 1 tick received per symbol and all messages pass schema validation.  
**Dependencies:** DATA-02-001, INFRA-01-006.

```python
class AlpacaStreamConnector:
    def __init__(self, api_key: str, secret_key: str, kafka_producer: KafkaProducer):
        ...
    
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to tick stream and forward to Kafka."""
        ...
    
    async def _handle_tick(self, tick: dict) -> None:
        validated = TickData(**tick)
        await self.kafka_producer.send("market.ticks", validated.model_dump_json())
    
    async def run_forever(self) -> None:
        """Main loop with exponential backoff reconnect."""
        ...
```

---

### DATA-02-003 — Market Data Connector: Yahoo Finance (Fallback/Free Tier)
**What:** Implement `yfinance`-based connector for historical data as a free fallback. Used for crypto, commodities, and indices.  
**Deliverable:** `services/data_service/connectors/yfinance_connector.py`.  
**Test:** Fetch SPY, BTC-USD, GC=F (Gold), CL=F (Crude Oil) for 2 years. Schema parity with Polygon connector output.  
**Dependencies:** INFRA-01-004.

---

### DATA-02-004 — Macro Data Connector: FRED API
**What:** Fetch macroeconomic series from the Federal Reserve Economic Data (FRED) API.  
**Deliverable:** `services/data_service/connectors/fred_connector.py` with a series registry and daily refresh scheduler.  
**Test:** Fetch all series in the registry. Assert no series older than 7 days. Publish to `macro.updates` Kafka topic.  
**Dependencies:** INFRA-01-004, INFRA-01-006.

**Series registry to implement:**
```python
FRED_SERIES = {
    # Interest Rates
    "DFF":    "Fed Funds Rate",
    "DGS10":  "10Y Treasury Yield",
    "DGS2":   "2Y Treasury Yield",
    "T10Y2Y": "10Y-2Y Yield Spread",
    # Inflation
    "CPIAUCSL": "CPI Urban Consumers",
    "PCEPI":    "PCE Price Index",
    # Employment
    "UNRATE":  "Unemployment Rate",
    "PAYEMS":  "Nonfarm Payrolls",
    # Growth
    "GDP":     "Gross Domestic Product",
    "INDPRO":  "Industrial Production",
    # Credit & Liquidity
    "DEXUSEU": "EUR/USD Exchange Rate",
    "VIXCLS":  "CBOE VIX",
    "BAMLH0A0HYM2": "HY Spread (OAS)",
}
```

---

### DATA-02-005 — News Data Connector: NewsAPI + RSS Feeds
**What:** Aggregate financial news from NewsAPI and Bloomberg/Reuters RSS feeds. Publish raw articles to `news.raw` Kafka topic.  
**Deliverable:** `services/data_service/connectors/news_connector.py` with deduplication via Redis SET.  
**Test:** Run for 5 minutes. Assert unique articles (no duplicate IDs), correct source tagging, and publishing rate ≥ 1/min during market hours.  
**Dependencies:** INFRA-01-003, INFRA-01-006.

```python
class NewsConnector:
    NEWS_SOURCES = [
        "bloomberg.com", "reuters.com", "ft.com", 
        "wsj.com", "cnbc.com", "marketwatch.com"
    ]
    RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bloomberg.com/markets/news.rss",
    ]
    
    async def poll_newsapi(self, query: str = "market finance economy") -> list[NewsArticle]:
        ...
    
    async def poll_rss(self) -> list[NewsArticle]:
        ...
    
    async def publish_with_dedup(self, articles: list[NewsArticle]) -> int:
        """Returns count of net-new articles published."""
        ...
```

---

### DATA-02-006 — Order Book Connector: Binance WebSocket (Crypto)
**What:** Subscribe to Binance Level-2 order book stream for top crypto pairs. Publish `OrderBookSnapshot` to `orderbook.snapshots`.  
**Deliverable:** `services/data_service/connectors/binance_orderbook.py`.  
**Test:** Stream BTC/USDT and ETH/USDT for 60 seconds. Validate bid/ask arrays are sorted correctly (bids descending, asks ascending). Assert latency from Binance event to Kafka publish < 200ms.  
**Dependencies:** INFRA-01-004, INFRA-01-006.

---

### DATA-02-007 — Data Service: Unified Orchestrator
**What:** Wrap all connectors in a single `DataService` FastAPI application with start/stop/status endpoints.  
**Deliverable:** `services/data_service/main.py` — containerized service with health endpoint.  
**Test:** `GET /health` returns `{"status": "ok", "connectors": {...}}`. Each connector's status reflects its actual state (running/stopped/error).  
**Dependencies:** DATA-02-001 through DATA-02-006.

---

### DATA-02-008 — Historical Data Ingestion to ClickHouse
**What:** Batch-ingest 5 years of daily OHLCV for all target symbols into ClickHouse offline store.  
**Deliverable:** `services/data_service/historical_loader.py` — idempotent loader with checkpointing.  
**Test:** Query ClickHouse for symbol count, date range coverage, and zero-gap assertions. Re-run loader: zero duplicate rows.  
**Dependencies:** DATA-02-001, DATA-02-003, INFRA-01-003.

**Target Universe:**
```python
SYMBOL_UNIVERSE = {
    "equities":    ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "futures":     ["ES=F", "NQ=F", "RTY=F", "YM=F", "CL=F", "GC=F", "SI=F", "ZB=F"],
    "forex":       ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
    "crypto":      ["BTC-USD", "ETH-USD", "SOL-USD"],
    "volatility":  ["^VIX", "^VVIX"],
}
```

---

## SPRINT 03 — Feature Engineering Pipeline

**Goal:** Transform raw market and macro data into model-ready features, available in both real-time (Redis) and offline (ClickHouse) stores.

---

### FEAT-03-001 — Technical Indicator Library
**What:** Implement a pure-Python/NumPy library of 40+ technical indicators optimized for vectorized computation.  
**Deliverable:** `shared/features/technical.py` — all functions take `np.ndarray` or `pd.Series` as input.  
**Test:** Numerical equivalence tests against TA-Lib reference implementation for each indicator. Max relative error < 1e-6.  
**Dependencies:** INFRA-01-002.

**Indicators to implement:**
```python
# Trend
def sma(close: np.ndarray, period: int) -> np.ndarray: ...
def ema(close: np.ndarray, period: int) -> np.ndarray: ...
def wma(close: np.ndarray, period: int) -> np.ndarray: ...
def macd(close: np.ndarray, fast=12, slow=26, signal=9) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def adx(high, low, close, period=14) -> np.ndarray: ...
def ichimoku(high, low, close) -> dict[str, np.ndarray]: ...

# Momentum
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray: ...
def stochastic(high, low, close, k=14, d=3) -> tuple[np.ndarray, np.ndarray]: ...
def cci(high, low, close, period=20) -> np.ndarray: ...
def williams_r(high, low, close, period=14) -> np.ndarray: ...
def roc(close: np.ndarray, period: int) -> np.ndarray: ...

# Volatility
def atr(high, low, close, period=14) -> np.ndarray: ...
def bollinger_bands(close, period=20, std_dev=2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def keltner_channel(high, low, close, period=20) -> tuple: ...
def historical_volatility(close, period=20) -> np.ndarray: ...

# Volume
def obv(close, volume) -> np.ndarray: ...
def vwap(high, low, close, volume) -> np.ndarray: ...
def cmf(high, low, close, volume, period=20) -> np.ndarray: ...
def volume_profile(close, volume, bins=50) -> dict: ...

# Cross-Asset / Derived
def beta(asset_returns, benchmark_returns, period=60) -> np.ndarray: ...
def correlation_matrix(returns_df: pd.DataFrame, period=60) -> pd.DataFrame: ...
def z_score(series: np.ndarray, period: int) -> np.ndarray: ...
def hurst_exponent(series: np.ndarray) -> float: ...
```

---

### FEAT-03-002 — Microstructure Features (Order Book)
**What:** Compute order book derived features from `OrderBookSnapshot` messages.  
**Deliverable:** `shared/features/microstructure.py`.  
**Test:** Feed synthetic order book data; assert bid-ask spread, order imbalance, and depth ratios are computed correctly.  
**Dependencies:** INFRA-01-004, FEAT-03-001.

```python
def bid_ask_spread(snapshot: OrderBookSnapshot) -> float:
    """Best ask - best bid."""
    ...

def order_imbalance(snapshot: OrderBookSnapshot, depth: int = 5) -> float:
    """(bid_vol - ask_vol) / (bid_vol + ask_vol) for top N levels."""
    ...

def weighted_mid_price(snapshot: OrderBookSnapshot, depth: int = 5) -> float:
    """Volume-weighted mid price across top N levels."""
    ...

def book_depth_ratio(snapshot: OrderBookSnapshot, depth: int = 10) -> float:
    """Ratio of cumulative bid depth to ask depth."""
    ...

def trade_flow_imbalance(ticks: list[TickData], window_sec: int = 60) -> float:
    """Net signed volume over window: buy_vol - sell_vol."""
    ...
```

---

### FEAT-03-003 — Macro Feature Engineering
**What:** Transform raw FRED series into model-ready macro features with cross-series derivations.  
**Deliverable:** `shared/features/macro.py`.  
**Test:** Validate yield curve slope, inflation momentum, and credit spread trend calculations against known historical periods (e.g., 2007 inversion, 2022 hike cycle).  
**Dependencies:** DATA-02-004.

```python
def yield_curve_slope(df: pd.DataFrame) -> pd.Series:
    """10Y minus 2Y yield."""
    return df["DGS10"] - df["DGS2"]

def inflation_momentum(df: pd.DataFrame, periods=[1, 3, 6]) -> pd.DataFrame:
    """Month-over-month CPI change across multiple periods."""
    ...

def real_rate(df: pd.DataFrame) -> pd.Series:
    """10Y nominal minus 1Y trailing CPI."""
    ...

def credit_stress_index(df: pd.DataFrame) -> pd.Series:
    """Composite of VIX, HY spread, and yield curve."""
    ...

def macro_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """All macro features as a single feature matrix."""
    ...
```

---

### FEAT-03-004 — Real-Time Feature Computation with Bytewax
**What:** Implement streaming feature computation pipeline using Bytewax that consumes `market.ohlcv.1m` and `orderbook.snapshots` and publishes to `features.realtime`.  
**Deliverable:** `services/feature_service/realtime_pipeline.py` — Bytewax dataflow.  
**Test:** Feed 1000 simulated OHLCV bars through the pipeline. Assert RSI, MACD, ATR values match batch computation. End-to-end latency < 100ms per bar.  
**Dependencies:** FEAT-03-001, FEAT-03-002, INFRA-01-006.

```python
# Bytewax dataflow skeleton
def build_feature_dataflow() -> Dataflow:
    flow = Dataflow("feature_pipeline")
    
    # Input: consume from Kafka
    flow.input("kafka_in", KafkaSource(topics=["market.ohlcv.1m", "orderbook.snapshots"]))
    
    # Parse and route by type
    flow.map("parse", parse_kafka_message)
    flow.filter_map("ohlcv_only", extract_ohlcv)
    
    # Windowed stateful feature computation
    flow.stateful_map("technical_features", TechnicalFeatureState(window=200))
    flow.stateful_map("microstructure_features", MicrostructureFeatureState(window=60))
    
    # Merge and publish
    flow.output("kafka_out", KafkaSink(topic="features.realtime"))
    
    return flow
```

---

### FEAT-03-005 — Offline Feature Store Population (Feast)
**What:** Configure Feast feature store and write historical features to ClickHouse offline store for model training.  
**Deliverable:** `services/feature_service/feast_registry.py` — feature views, entities, and feature service definitions.  
**Test:** `feast materialize` completes without error. Point-in-time join retrieves correct historical features for 3 arbitrary training examples.  
**Dependencies:** FEAT-03-001, FEAT-03-003, DATA-02-008, INFRA-01-003.

---

### FEAT-03-006 — Feature Validation & Quality Checks
**What:** Implement automated feature quality checks that run on every batch update.  
**Deliverable:** `services/feature_service/quality_checks.py` using Great Expectations.  
**Test:** Intentionally inject NaN, infinite values, and out-of-range data. Assert all violations are caught and logged.  
**Dependencies:** FEAT-03-005.

```python
class FeatureQualityChecker:
    EXPECTATIONS = {
        "rsi_14":          {"min": 0, "max": 100, "null_fraction_max": 0.01},
        "log_return_1d":   {"min": -0.5, "max": 0.5, "null_fraction_max": 0.001},
        "volume":          {"min": 0, "null_fraction_max": 0.0},
        "bid_ask_spread":  {"min": 0, "null_fraction_max": 0.05},
    }
    
    def run_checks(self, feature_df: pd.DataFrame) -> ValidationResult:
        ...
```

---

## SPRINT 04 — Statistical & Classical ML Models

**Goal:** Build the first tier of models — interpretable, fast, and production-proven approaches.

---

### MODEL-04-001 — Return Prediction: Linear Factor Model
**What:** Implement a rolling OLS factor model (Fama-French style) for expected return estimation.  
**Deliverable:** `services/model_service/models/linear_factor.py`.  
**Test:** Backtest on 3 years of out-of-sample data. Information Coefficient (IC) > 0.03. Coefficient signs match economic priors (momentum positive, value negative for growth stocks).  
**Dependencies:** FEAT-03-005.

```python
class LinearFactorModel:
    """Rolling OLS with configurable factor universe."""
    
    FACTORS = [
        "momentum_12_1",    # 12-month return minus last month
        "value_ep",         # Earnings/Price ratio
        "size_log_mcap",    # Log market cap
        "quality_roe",      # Return on equity
        "vol_realized_20",  # Realized volatility
        "macro_real_rate",  # Real interest rate
    ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, window: int = 252) -> None:
        """Rolling window OLS fit."""
        ...
    
    def predict(self, X: pd.DataFrame) -> SignalOutput:
        """Returns predicted returns with confidence intervals."""
        ...
```

---

### MODEL-04-002 — Regime Detection: Hidden Markov Model
**What:** Train a Gaussian HMM on returns, volatility, and macro features to identify 4 market regimes.  
**Deliverable:** `services/model_service/models/hmm_regime.py`.  
**Test:** Confirm the 4 regimes map to economically interpretable states (risk-on bull, low-vol grind, high-vol stress, bear/crisis) by overlaying regime labels on historical S&P 500 chart and manually validating.  
**Dependencies:** FEAT-03-001, FEAT-03-003.

```python
class HMMRegimeDetector:
    REGIMES = {
        0: "risk_on_trending",
        1: "low_vol_range",
        2: "high_vol_stress",
        3: "crisis_bear",
    }
    
    def fit(self, features: pd.DataFrame, n_states: int = 4) -> None:
        """Train HMM using hmmlearn."""
        ...
    
    def predict_regime(self, features: pd.DataFrame) -> tuple[int, np.ndarray]:
        """Returns (current_regime, probability_vector)."""
        ...
    
    def plot_regimes(self, features: pd.DataFrame, price: pd.Series) -> None:
        """Visualize regime overlays on price chart."""
        ...
```

---

### MODEL-04-003 — Return Prediction: LightGBM Gradient Boosted Trees
**What:** Train LightGBM models on the full feature set for 1-day and 5-day forward return prediction.  
**Deliverable:** `services/model_service/models/lgbm_predictor.py` with hyperparameter tuning via Optuna.  
**Test:** Out-of-sample Sharpe > 0.8 on 2022–2024 walk-forward. Feature importance plot must not be dominated by lagged returns (data leakage check). SHAP values computed and stored.  
**Dependencies:** FEAT-03-005, MODEL-04-002.

```python
class LGBMPredictor:
    def __init__(self, horizon: Literal["1d", "5d"] = "1d"):
        self.horizon = horizon
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        regime_weights: np.ndarray | None = None,  # up-weight recent regimes
    ) -> TrainingReport:
        ...
    
    def predict(self, X: pd.DataFrame) -> SignalOutput:
        """Returns point prediction + quantile uncertainty bounds."""
        ...
    
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values per feature per sample."""
        ...
    
    def tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> dict:
        """Optuna hyperparameter search."""
        ...
```

---

### MODEL-04-004 — Volatility Forecasting: GARCH(1,1) and EGARCH
**What:** Fit GARCH and EGARCH models per symbol for next-period volatility forecasting. Used downstream for position sizing.  
**Deliverable:** `services/model_service/models/garch_vol.py`.  
**Test:** Compare out-of-sample RMSE of GARCH forecast vs. realized volatility against a naive historical vol baseline. GARCH must outperform baseline.  
**Dependencies:** DATA-02-008, FEAT-03-001.

```python
class GARCHVolatilityModel:
    def fit(self, returns: pd.Series, model_type: Literal["GARCH", "EGARCH"] = "EGARCH") -> None:
        """Fit using arch library."""
        ...
    
    def forecast(self, horizon: int = 5) -> VolatilityForecast:
        """Returns annualized vol forecast with confidence interval."""
        ...
    
    def fit_universe(self, returns_df: pd.DataFrame) -> dict[str, "GARCHVolatilityModel"]:
        """Parallel fitting across symbol universe."""
        ...
```

---

### MODEL-04-005 — Mean Reversion: Cointegration Pairs Detection
**What:** Detect statistically cointegrated pairs from the symbol universe for stat-arb strategy.  
**Deliverable:** `services/model_service/models/pairs_cointegration.py`.  
**Test:** Johansen test p-value < 0.05 for all confirmed pairs. Half-life of mean reversion between 5 and 60 days. Validate at least 3 robust pairs exist in the universe.  
**Dependencies:** DATA-02-008, FEAT-03-001.

```python
class CointegrationPairsFinder:
    def find_pairs(
        self,
        prices: pd.DataFrame,
        max_pairs: int = 20,
        p_value_threshold: float = 0.05,
    ) -> list[PairResult]:
        """Run Engle-Granger and Johansen tests on all symbol pairs."""
        ...
    
    def compute_spread(self, pair: PairResult, prices: pd.DataFrame) -> pd.Series:
        """Hedge-ratio-adjusted spread between the pair."""
        ...
    
    def half_life(self, spread: pd.Series) -> float:
        """Ornstein-Uhlenbeck mean reversion half-life in days."""
        ...
```

---

### MODEL-04-006 — Model Registry Integration (MLflow)
**What:** Wrap all model classes with MLflow tracking — log params, metrics, artifacts, and register to model registry.  
**Deliverable:** `services/model_service/mlflow_wrapper.py` — decorator/context manager pattern.  
**Test:** Train LightGBM model with wrapper. Assert experiment appears in MLflow UI with all metrics, params, and SHAP artifacts. Model can be loaded back using `mlflow.pyfunc.load_model()`.  
**Dependencies:** MODEL-04-003, INFRA-01-003.

---

## SPRINT 05 — Deep Learning & NLP Models

**Goal:** Build the high-capacity models for pattern recognition and sentiment processing.

---

### MODEL-05-001 — Time Series Transformer: Temporal Fusion Transformer (TFT)
**What:** Implement and train a Temporal Fusion Transformer for multi-horizon price direction forecasting.  
**Deliverable:** `services/model_service/models/tft_predictor.py` using PyTorch Forecasting library.  
**Test:** Out-of-sample directional accuracy > 53% on held-out 2024 data. Attention weights must vary meaningfully across inputs (not collapsed). Training must not overfit: val_loss / train_loss < 1.3.  
**Dependencies:** FEAT-03-005, INFRA-01-002.

```python
class TFTPredictor:
    """Temporal Fusion Transformer for multi-horizon forecasting."""
    
    def __init__(
        self,
        max_encoder_length: int = 60,    # 60 bars of history
        max_prediction_length: int = 5,   # predict 5 bars ahead
        hidden_size: int = 128,
        attention_head_size: int = 4,
        dropout: float = 0.1,
    ):
        ...
    
    def build_dataset(self, features: pd.DataFrame) -> TimeSeriesDataSet:
        """Configure PyTorch Forecasting dataset with feature groups."""
        ...
    
    def train(
        self,
        train_data: TimeSeriesDataSet,
        val_data: TimeSeriesDataSet,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        gradient_clip_val: float = 0.1,
    ) -> TrainingReport:
        ...
    
    def predict(self, X: pd.DataFrame) -> SignalOutput:
        ...
```

---

### MODEL-05-002 — NLP Sentiment Model: FinBERT Fine-Tuning
**What:** Fine-tune FinBERT on financial news headlines for 3-class sentiment (positive/neutral/negative). Deploy as inference endpoint.  
**Deliverable:** `services/model_service/models/finbert_sentiment.py` + BentoML service definition.  
**Test:** F1 > 0.80 on Financial PhraseBank test set. Inference latency < 100ms per headline on CPU. Batch inference (100 headlines) < 2 seconds on GPU.  
**Dependencies:** DATA-02-005, INFRA-01-002.

```python
class FinBERTSentimentModel:
    BASE_MODEL = "ProsusAI/finbert"
    
    def __init__(self, model_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path or self.BASE_MODEL
        )
    
    def fine_tune(
        self,
        train_df: pd.DataFrame,   # columns: text, label (0/1/2)
        val_df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
    ) -> TrainingReport:
        ...
    
    def predict_sentiment(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[SentimentResult]:
        """Returns label + probability for each text."""
        ...
    
    def aggregate_article_sentiment(
        self,
        article: NewsArticle,
    ) -> SentimentResult:
        """Sentiment from headline + first 3 sentences."""
        ...
```

---

### MODEL-05-003 — News Sentiment Pipeline (Kafka Consumer)
**What:** Build the Kafka consumer that reads `news.raw`, runs FinBERT inference, and publishes `SentimentResult` to `news.sentiment`.  
**Deliverable:** `services/model_service/sentiment_pipeline.py`.  
**Test:** End-to-end: inject 50 articles into `news.raw`; assert all 50 appear in `news.sentiment` within 30 seconds with valid sentiment scores.  
**Dependencies:** MODEL-05-002, DATA-02-005.

---

### MODEL-05-004 — LSTM Sequence Model for Intraday Patterns
**What:** Train a bidirectional LSTM on 1-minute bars to capture intraday mean-reversion and momentum patterns.  
**Deliverable:** `services/model_service/models/lstm_intraday.py`.  
**Test:** Directional accuracy on 30-minute-ahead prediction > 52%. No look-ahead bias verified by unit test that checks data alignment. Gradient norms stable during training (no explosion).  
**Dependencies:** FEAT-03-004, FEAT-03-005.

```python
class IntraLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8)
        self.fc = nn.Linear(hidden_size * 2, 3)   # long / flat / short
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

---

### MODEL-05-005 — Graph Neural Network: Cross-Asset Correlation Model
**What:** Build a GNN that models relationships between assets using correlation and supply-chain graphs to improve multi-asset signal quality.  
**Deliverable:** `services/model_service/models/gnn_correlation.py` using PyTorch Geometric.  
**Test:** GNN ensemble improves Sharpe by ≥ 5% over non-graph baseline on held-out data. Graph connectivity must update weekly based on rolling correlation.  
**Dependencies:** FEAT-03-005, MODEL-04-001.

---

## SPRINT 06 — Signal Aggregation & Ensemble

**Goal:** Combine all individual model outputs into a single, calibrated, confidence-weighted composite signal.

---

### SIGNAL-06-001 — Individual Model Signal Normalizer
**What:** Standardize all model outputs to a common `[-1, +1]` scale with associated confidence.  
**Deliverable:** `services/signal_service/normalizer.py`.  
**Test:** Feed outputs from each of the 6 models. Assert all outputs are within `[-1, +1]`. Assert confidence scores are in `[0, 1]` and sum-normalized.  
**Dependencies:** All MODEL-04 and MODEL-05 tasks.

```python
class SignalNormalizer:
    def normalize(self, raw_output: ModelOutput) -> NormalizedSignal:
        """Maps raw model prediction to [-1, +1] with confidence."""
        ...
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        model_id: str,
        regime: str,
    ) -> float:
        """Regime-adjusted Platt scaling calibration."""
        ...
```

---

### SIGNAL-06-002 — Meta-Learner Ensemble (Stacking)
**What:** Train a stacking meta-learner (Ridge regression + LightGBM) that takes individual model signals as inputs and outputs the composite signal.  
**Deliverable:** `services/signal_service/ensemble.py`.  
**Test:** Ensemble Sharpe > best single model Sharpe on 2-year walk-forward test. Verify meta-learner weights are stable (not dominated by a single model > 80%).  
**Dependencies:** SIGNAL-06-001.

```python
class EnsembleMetaLearner:
    """
    Stacking ensemble: L1 = 6 base models, L2 = Ridge + LGBM blend.
    Weights adapt dynamically based on regime and recent model performance.
    """
    
    def __init__(self):
        self.ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        self.lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.05)
        self.regime_weights: dict[str, np.ndarray] = {}
    
    def fit(
        self,
        model_signals: pd.DataFrame,  # columns = model_ids, rows = timestamps
        realized_returns: pd.Series,
        regimes: pd.Series,
    ) -> None:
        """Cross-validated stacking fit, stratified by regime."""
        ...
    
    def predict(
        self,
        model_signals: dict[str, float],
        current_regime: int,
        regime_probs: np.ndarray,
    ) -> CompositeSignal:
        """Returns direction, magnitude, confidence, and per-model attribution."""
        ...
```

---

### SIGNAL-06-003 — Signal Service API
**What:** Deploy the ensemble as a FastAPI service that accepts feature vectors and returns composite signals.  
**Deliverable:** `services/signal_service/main.py` with `/signal`, `/health`, and `/explain` endpoints.  
**Test:** Load test at 100 req/sec for 60 seconds. P99 latency < 200ms. `/explain` returns SHAP-style attribution per model.  
**Dependencies:** SIGNAL-06-002, INFRA-01-005.

---

### SIGNAL-06-004 — Signal Kafka Consumer & Publisher
**What:** Consume `features.realtime`, run ensemble inference, publish `CompositeSignal` to `signals.composite`.  
**Deliverable:** `services/signal_service/kafka_consumer.py`.  
**Test:** End-to-end from feature publication to composite signal < 500ms. Confirm signal frequency matches input bar frequency (1-minute).  
**Dependencies:** SIGNAL-06-003, INFRA-01-006.

---

## SPRINT 07 — Strategy & Decision Engine

**Goal:** Translate composite signals into executable trade decisions via regime-aware strategy selection.

---

### STRAT-07-001 — Strategy Framework: Base Class & Interface
**What:** Define the abstract strategy interface that all strategy implementations must follow.  
**Deliverable:** `services/decision_engine/strategies/base.py`.  
**Test:** Attempt to instantiate base class directly — must raise `NotImplementedError`. Mock subclass passes all interface checks.  
**Dependencies:** SIGNAL-06-003.

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """All strategies must implement this interface."""
    
    @abstractmethod
    def generate_orders(
        self,
        signal: CompositeSignal,
        portfolio_state: PortfolioState,
        risk_budget: RiskBudget,
    ) -> list[ProposedOrder]:
        """Core strategy logic: signal → orders."""
        ...
    
    @abstractmethod
    def get_regime_affinity(self) -> dict[str, float]:
        """Returns how well this strategy performs in each regime. 0–1 scale."""
        ...
    
    @property
    @abstractmethod
    def strategy_id(self) -> str:
        ...
```

---

### STRAT-07-002 — Trend Following Strategy
**What:** Implement a momentum/trend strategy that goes long (short) when composite signal exceeds a threshold and ADX confirms trend strength.  
**Deliverable:** `services/decision_engine/strategies/trend_following.py`.  
**Test:** Backtest on 2020–2024 SPY data. Regime affinity highest in `risk_on_trending`. Win rate > 45% with profit factor > 1.5.  
**Dependencies:** STRAT-07-001, FEAT-03-001.

---

### STRAT-07-003 — Mean Reversion Strategy
**What:** Implement a stat-arb strategy that enters positions when pair spread Z-score exceeds ±2.0 and exits at ±0.5.  
**Deliverable:** `services/decision_engine/strategies/mean_reversion.py`.  
**Test:** Run on discovered cointegrated pairs from MODEL-04-005. Regime affinity highest in `low_vol_range`. Validate spread Z-score boundaries trigger correct long/short leg sizing.  
**Dependencies:** STRAT-07-001, MODEL-04-005.

---

### STRAT-07-004 — Event-Driven Strategy (Earnings + Macro)
**What:** Build a strategy that takes positions around scheduled catalyst events (earnings, FOMC, CPI) using pre-event and post-event signal patterns.  
**Deliverable:** `services/decision_engine/strategies/event_driven.py` with an economic calendar integration.  
**Test:** Historical event catalog covers all 2023–2024 events. Position sizing auto-reduces 24 hours before high-impact events. Post-event drift signals validated against 2023 data.  
**Dependencies:** STRAT-07-001, DATA-02-004.

---

### STRAT-07-005 — Regime-Aware Strategy Allocator
**What:** Implement the meta-logic that dynamically weights active strategies based on current market regime probabilities.  
**Deliverable:** `services/decision_engine/strategy_allocator.py`.  
**Test:** In crisis regime, trend strategy weight < 20% and mean reversion disabled. Weight vector must always sum to 1.0. Transition smoothing prevents flip-flopping.  
**Dependencies:** STRAT-07-002, STRAT-07-003, STRAT-07-004, MODEL-04-002.

```python
class StrategyAllocator:
    REGIME_STRATEGY_WEIGHTS = {
        "risk_on_trending":  {"trend": 0.60, "mean_rev": 0.20, "event": 0.20},
        "low_vol_range":     {"trend": 0.20, "mean_rev": 0.60, "event": 0.20},
        "high_vol_stress":   {"trend": 0.30, "mean_rev": 0.10, "event": 0.60},
        "crisis_bear":       {"trend": 0.10, "mean_rev": 0.00, "event": 0.90},
    }
    
    def allocate(
        self,
        regime_probs: np.ndarray,
        recent_strategy_sharpes: dict[str, float],
    ) -> dict[str, float]:
        """Regime-probability-weighted strategy allocation with performance adjustment."""
        ...
```

---

## SPRINT 08 — Risk Management & Portfolio Construction

**Goal:** Ensure no single trade or regime can cause catastrophic drawdown.

---

### RISK-08-001 — Position Sizing: Kelly Criterion & Variants
**What:** Implement full Kelly, half-Kelly, and fractional Kelly position sizing models.  
**Deliverable:** `services/risk_service/position_sizing.py`.  
**Test:** Verify Kelly formula at known win rate/payout combinations. Assert fractional Kelly (0.25f) never exceeds `max_position_pct` from settings. Monte Carlo simulation shows bounded drawdown.  
**Dependencies:** INFRA-01-005.

```python
class KellyPositionSizer:
    def full_kelly(self, win_prob: float, win_loss_ratio: float) -> float:
        """f* = (bp - q) / b where b=win/loss, p=win_prob, q=1-p."""
        b = win_loss_ratio
        p, q = win_prob, 1 - win_prob
        return (b * p - q) / b
    
    def fractional_kelly(self, win_prob: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
        return fraction * self.full_kelly(win_prob, win_loss_ratio)
    
    def volatility_adjusted_size(
        self,
        signal_confidence: float,
        forecast_vol: float,      # from GARCH model
        target_vol: float = 0.01, # 1% daily portfolio vol per position
    ) -> float:
        """Size = (target_vol / forecast_vol) * confidence_scalar."""
        ...
    
    def regime_adjusted_size(
        self,
        base_size: float,
        regime: str,
        regime_multipliers: dict[str, float] = {
            "risk_on_trending": 1.0,
            "low_vol_range":    0.75,
            "high_vol_stress":  0.5,
            "crisis_bear":      0.25,
        },
    ) -> float:
        ...
```

---

### RISK-08-002 — Risk Metrics: VaR, CVaR, and Drawdown
**What:** Implement real-time risk metric computation for the current portfolio.  
**Deliverable:** `services/risk_service/risk_metrics.py`.  
**Test:** Known portfolio: validate Historical VaR(95%) matches manual calculation. Monte Carlo VaR within 5% of parametric VaR. Drawdown tracker updates correctly on each mark-to-market.  
**Dependencies:** INFRA-01-004.

```python
class RiskMetricsEngine:
    def historical_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Historical simulation VaR (negative = loss)."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def parametric_var(
        self,
        portfolio_value: float,
        portfolio_vol: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Variance-covariance VaR assuming normality."""
        z = scipy.stats.norm.ppf(1 - confidence)
        return portfolio_value * portfolio_vol * z * np.sqrt(horizon_days)
    
    def conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Expected Shortfall = mean of returns below VaR threshold."""
        var = self.historical_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def max_drawdown(self, equity_curve: np.ndarray) -> DrawdownResult:
        """Peak-to-trough max drawdown with duration."""
        ...
    
    def compute_portfolio_metrics(
        self,
        positions: list[PositionRisk],
        correlation_matrix: pd.DataFrame,
    ) -> PortfolioRiskReport:
        ...
```

---

### RISK-08-003 — Portfolio Optimizer: Mean-Variance with Constraints
**What:** Implement Markowitz mean-variance optimization using Clarabel solver with institutional-grade constraints.  
**Deliverable:** `services/risk_service/portfolio_optimizer.py`.  
**Test:** Known expected returns and covariance: optimal weights match analytical solution within 1e-4. Constraints are respected: long-only, max position size, max sector exposure.  
**Dependencies:** RISK-08-002.

```python
class MeanVarianceOptimizer:
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        objective: Literal["max_sharpe", "min_variance", "risk_parity"] = "max_sharpe",
    ) -> OptimalWeights:
        """
        Solve: max w'μ - λ * w'Σw
        Subject to: Σw_i = 1, 0 ≤ w_i ≤ max_weight, sector limits
        Using CVXPY with Clarabel backend.
        """
        ...
```

---

### RISK-08-004 — Real-Time Risk Monitor & Circuit Breakers
**What:** Implement risk monitor that tracks live positions and triggers circuit breakers when limits are breached.  
**Deliverable:** `services/risk_service/risk_monitor.py`.  
**Test:** Simulate a position that breaches drawdown limit. Assert circuit breaker fires within 1 second and publishes to `risk.alerts` topic. Verify monitor restores normal operation after breach resolves.  
**Dependencies:** RISK-08-002, INFRA-01-006.

```python
class RiskCircuitBreaker:
    CIRCUIT_BREAKERS = {
        "max_portfolio_drawdown":   0.10,  # 10% from peak
        "max_daily_loss":           0.03,  # 3% in one day
        "max_position_loss":        0.05,  # 5% per position
        "var_breach_multiplier":    2.0,   # 2x VaR limit
        "correlation_spike":        0.90,  # all positions correlate > 90%
    }
    
    async def monitor_loop(self, interval_sec: float = 1.0) -> None:
        """Continuous monitoring loop."""
        ...
    
    async def trigger_breach(self, breach: RiskBreach) -> None:
        """Publish alert, halt new orders, optionally flatten positions."""
        ...
```

---

## SPRINT 09 — Backtesting Framework

**Goal:** Build a rigorous, leakage-free backtesting system that produces reliable performance statistics.

---

### BACK-09-001 — Backtesting Engine Core
**What:** Build a vectorized backtesting engine with realistic transaction cost modeling.  
**Deliverable:** `backtesting/engine.py`.  
**Test:** Run on known strategy (simple SMA crossover on SPY). Results match VectorBT reference implementation within 0.1% on Sharpe and final PnL.  
**Dependencies:** DATA-02-008, FEAT-03-001.

```python
class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_pct: float = 0.0005,     # 5bps per side
        slippage_pct: float = 0.0002,       # 2bps slippage model
        margin_requirement: float = 0.50,   # 50% margin for leverage
    ):
        ...
    
    def run(
        self,
        signals: pd.DataFrame,       # timestamp × symbol signals
        prices: pd.DataFrame,        # timestamp × symbol OHLCV
        position_sizes: pd.DataFrame # timestamp × symbol sizes (0-1)
    ) -> BacktestResult:
        """Vectorized loop: O(T) not O(T×N)."""
        ...
    
    def generate_report(self, result: BacktestResult) -> PerformanceReport:
        """Quantstats-style full report with all metrics."""
        ...
```

---

### BACK-09-002 — Walk-Forward Validation Framework
**What:** Implement anchored and rolling walk-forward cross-validation to prevent overfitting.  
**Deliverable:** `backtesting/walk_forward.py`.  
**Test:** Confirm no future data leaks: for each fold, verify train end < test start with a minimum 1-day gap. Assert model refit occurs at each fold boundary.  
**Dependencies:** BACK-09-001.

```python
class WalkForwardValidator:
    def __init__(
        self,
        train_window: int = 504,   # 2 years in trading days
        test_window: int = 63,     # 3 months
        step_size: int = 21,       # re-fit monthly
        anchored: bool = False,    # True = expanding window
    ):
        ...
    
    def generate_folds(self, data: pd.DataFrame) -> list[TrainTestFold]:
        """Yields (train_idx, test_idx) pairs with gap enforcement."""
        ...
    
    def run(
        self,
        data: pd.DataFrame,
        model_factory: Callable,
        strategy_factory: Callable,
    ) -> WalkForwardResult:
        ...
```

---

### BACK-09-003 — Performance Analytics Module
**What:** Compute institutional-grade performance statistics from backtest equity curves.  
**Deliverable:** `backtesting/analytics.py`.  
**Test:** Feed known equity curve (hand-computed). Assert all metrics within 0.01 of manual calculations.  
**Dependencies:** BACK-09-001.

```python
class PerformanceAnalytics:
    def compute_all(self, equity_curve: pd.Series, benchmark: pd.Series | None = None) -> dict:
        return {
            # Return metrics
            "total_return":         self.total_return(equity_curve),
            "cagr":                 self.cagr(equity_curve),
            "sharpe_ratio":         self.sharpe(equity_curve),
            "sortino_ratio":        self.sortino(equity_curve),
            "calmar_ratio":         self.calmar(equity_curve),
            "omega_ratio":          self.omega(equity_curve),
            
            # Risk metrics
            "max_drawdown":         self.max_drawdown(equity_curve),
            "avg_drawdown":         self.avg_drawdown(equity_curve),
            "max_drawdown_duration":self.max_dd_duration(equity_curve),
            "var_95":               self.var(equity_curve, 0.95),
            "cvar_95":              self.cvar(equity_curve, 0.95),
            "volatility_annualized":self.annualized_vol(equity_curve),
            
            # Signal quality
            "information_ratio":    self.ir(equity_curve, benchmark),
            "hit_rate":             self.hit_rate(equity_curve),
            "profit_factor":        self.profit_factor(equity_curve),
            "avg_win_loss_ratio":   self.avg_wl_ratio(equity_curve),
            "recovery_factor":      self.recovery_factor(equity_curve),
        }
```

---

### BACK-09-004 — Monte Carlo Simulation
**What:** Run 10,000 equity curve simulations via bootstrapped returns to estimate strategy robustness and confidence intervals.  
**Deliverable:** `backtesting/monte_carlo.py`.  
**Test:** Confirm that simulated Sharpe distribution includes the realized Sharpe. Assert p-value of strategy outperforming 0% return > 95%.  
**Dependencies:** BACK-09-003.

---

### BACK-09-005 — Overfitting Detection: Deflated Sharpe Ratio
**What:** Implement the Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) tests from López de Prado.  
**Deliverable:** `backtesting/overfitting_tests.py`.  
**Test:** Synthetic overfit strategy (tuned on in-sample, random out-of-sample) produces DSR < 0 and PBO > 0.5. Well-designed strategy produces DSR > 0.  
**Dependencies:** BACK-09-004.

---

## SPRINT 10 — Execution Layer

**Goal:** Build the order management and execution system.

---

### EXEC-10-001 — Order Management System (OMS) Core
**What:** Implement the OMS that validates, routes, and tracks all proposed orders.  
**Deliverable:** `services/execution_service/oms.py`.  
**Test:** Reject orders that violate position limits, risk budget, or circuit breaker state. Accept valid orders and assign unique IDs.  
**Dependencies:** RISK-08-004, INFRA-01-004.

---

### EXEC-10-002 — Broker Adapter: Alpaca Paper Trading
**What:** Implement Alpaca paper trading adapter with full order lifecycle (submit, modify, cancel, fill tracking).  
**Deliverable:** `services/execution_service/adapters/alpaca_adapter.py`.  
**Test:** Place a paper market order. Confirm fill price within 0.1% of last trade price. Cancel a limit order. Verify fill events publish to `orders.executed`.  
**Dependencies:** EXEC-10-001.

---

### EXEC-10-003 — Position Tracker & PnL Monitor
**What:** Real-time position book that tracks quantity, average cost, unrealized/realized PnL per symbol.  
**Deliverable:** `services/execution_service/position_tracker.py`.  
**Test:** Simulate 10 sequential trades including longs, shorts, partial fills, and flips. Assert final book matches hand-calculated expected positions and PnL.  
**Dependencies:** EXEC-10-002.

---

## SPRINT 11 — Monitoring & Feedback Loop

**Goal:** Build observability and continuous learning infrastructure.

---

### MON-11-001 — Grafana Dashboards
**What:** Create Grafana dashboards for system health, model performance, and trading activity.  
**Deliverable:** `monitoring/dashboards/` — JSON dashboard definitions importable into Grafana.  
**Test:** Dashboards load without errors. All panels return data from a test backtest run.  
**Dependencies:** INFRA-01-003.

**Dashboards to create:**
- System Health: Kafka lag, Redis memory, service latency, error rates
- Model Performance: IC per model, signal accuracy, regime transition frequency
- Trading Activity: PnL, drawdown, position heatmap, trade log
- Risk Dashboard: VaR gauge, CVaR trend, position risk breakdown

---

### MON-11-002 — Model Drift Detection
**What:** Implement Population Stability Index (PSI) and Kolmogorov-Smirnov tests to detect when model performance is degrading.  
**Deliverable:** `services/model_service/drift_detector.py`.  
**Test:** Inject synthetically shifted feature distribution. Assert drift alert triggers within one evaluation cycle (1 hour).  
**Dependencies:** MODEL-04-006, INFRA-01-006.

---

### MON-11-003 — Automated Model Retraining Pipeline (Airflow)
**What:** Define Airflow DAG that automatically retrains models weekly and promotes to production if performance improves.  
**Deliverable:** `infra/airflow/dags/model_retrain_dag.py`.  
**Test:** Trigger DAG manually. Assert it completes: feature extraction → training → evaluation → MLflow registration → conditional promotion.  
**Dependencies:** MON-11-002, MODEL-04-006.

---

## SPRINT 12 — Live Simulation & Hardening

**Goal:** Run the complete system in paper trading mode and harden for production.

---

### LIVE-12-001 — Live Paper Trading Simulation (2-Week Run)
**What:** Run the full system end-to-end on paper trading for 2 consecutive trading weeks without manual intervention.  
**Deliverable:** Daily PnL report, trade log, signal accuracy tracker.  
**Test (Pass Criteria):** Zero unhandled exceptions. System auto-recovers from any service restart. Sharpe over period > 0. Max drawdown < 5%.  
**Dependencies:** All previous sprints complete.

---

### LIVE-12-002 — Latency & Throughput Profiling
**What:** Profile the entire signal pipeline and identify bottlenecks.  
**Deliverable:** Flame graph report + optimization changelog.  
**Test:** Signal latency (tick → composite signal) < 2 seconds in steady state. System handles 100 symbols simultaneously.  
**Dependencies:** LIVE-12-001.

---

### LIVE-12-003 — Failure Mode Testing
**What:** Simulate failure modes: Kafka broker down, Redis unavailable, model service crash, data feed interruption.  
**Deliverable:** Runbook documenting each failure mode, impact, and auto-recovery behavior.  
**Test:** Each failure mode: system detects, degrades gracefully, and recovers without data loss within defined SLAs.  
**Dependencies:** LIVE-12-001.

---

## IMPLEMENTATION PRIORITY MATRIX

| Task ID | Sprint | Priority | Effort | Unblocked By |
|---|---|---|---|---|
| INFRA-01-001 | 01 | 🔴 Critical | 0.5d | — |
| INFRA-01-002 | 01 | 🔴 Critical | 0.5d | 001 |
| INFRA-01-003 | 01 | 🔴 Critical | 1d | 001 |
| INFRA-01-004 | 01 | 🔴 Critical | 1d | 002 |
| INFRA-01-005 | 01 | 🔴 Critical | 0.5d | 004 |
| INFRA-01-006 | 01 | 🔴 Critical | 0.5d | 003 |
| DATA-02-001 | 02 | 🔴 Critical | 1d | INFRA done |
| DATA-02-002 | 02 | 🔴 Critical | 1d | 02-001 |
| DATA-02-004 | 02 | 🔴 Critical | 1d | INFRA done |
| DATA-02-005 | 02 | 🟠 High | 1d | INFRA done |
| DATA-02-008 | 02 | 🔴 Critical | 1d | 02-001,003 |
| FEAT-03-001 | 03 | 🔴 Critical | 2d | DATA-02 done |
| FEAT-03-004 | 03 | 🔴 Critical | 2d | 03-001,002 |
| FEAT-03-005 | 03 | 🔴 Critical | 1d | 03-001,003 |
| MODEL-04-002 | 04 | 🟠 High | 1d | FEAT done |
| MODEL-04-003 | 04 | 🔴 Critical | 2d | FEAT done |
| MODEL-05-001 | 05 | 🟠 High | 3d | FEAT done |
| MODEL-05-002 | 05 | 🟠 High | 2d | DATA-02-005 |
| SIGNAL-06-002 | 06 | 🔴 Critical | 2d | All models |
| RISK-08-001 | 08 | 🔴 Critical | 1d | INFRA done |
| RISK-08-004 | 08 | 🔴 Critical | 1d | 08-002 |
| BACK-09-001 | 09 | 🔴 Critical | 2d | DATA + FEAT |
| BACK-09-002 | 09 | 🔴 Critical | 1d | 09-001 |

---

## ESTIMATED TIMELINE

| Sprint | Estimated Duration | Key Milestone |
|---|---|---|
| 01 — Foundation | 3 days | Stack running locally |
| 02 — Data Ingestion | 5 days | Live data flowing into Kafka |
| 03 — Features | 5 days | Feature store populated |
| 04 — Classical Models | 5 days | LGBM & HMM trained & logged |
| 05 — Deep Learning | 7 days | TFT + FinBERT deployed |
| 06 — Ensemble | 4 days | Composite signal live |
| 07 — Strategy | 4 days | Orders being generated |
| 08 — Risk | 4 days | Risk engine active |
| 09 — Backtesting | 4 days | Walk-forward results validated |
| 10 — Execution | 3 days | Paper trades executing |
| 11 — Monitoring | 3 days | Dashboards live |
| 12 — Hardening | 5 days | System passes live sim |
| **Total** | **~52 trading days** | **Production-ready paper system** |

---

> **Next Step:** Confirm which Sprint to begin with. Recommended start: **INFRA-01-001** — Project Repository Structure.
