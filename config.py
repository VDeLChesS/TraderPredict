from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
RESULTS_DIR        = PROJECT_ROOT / "results"

SYMBOLS    = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "AAPL", "TSLA", "GLD", "TLT"]
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
INTERVAL   = "1d"

PARQUET_ENGINE = "pyarrow"

# ------------------------------------------------------------------
# Risk management defaults (Phase 7)
# ------------------------------------------------------------------
RISK_DEFAULTS = {
    "position_mode":      "fixed",
    "position_fraction":  1.0,
    "vol_lookback":       20,
    "target_risk_pct":    0.02,
    "atr_window":         14,
    "stop_mode":          "none",
    "sl_pct":             0.05,
    "tp_pct":             0.10,
    "sl_atr_mult":        2.0,
    "tp_atr_mult":        3.0,
    "trailing":           False,
    "use_circuit_breaker": False,
    "max_drawdown_pct":   0.20,
    "cooldown_bars":      10,
    "recovery_pct":       None,
}
