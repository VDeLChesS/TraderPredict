import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """Fetch, clean, store, and validate OHLCV data as Parquet files."""

    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        engine: str = "pyarrow",
    ):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.engine        = engine
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        source: str = "yfinance",
        save: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV and optionally save raw Parquet."""
        logger.info(f"Fetching {symbol} from {source} [{start} → {end}]")

        if source == "yfinance":
            df = self._fetch_yfinance(symbol, start, end, interval)
        elif source == "binance":
            df = self._fetch_binance(symbol, start, end, interval)
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'yfinance' or 'binance'.")

        if save:
            path = self.raw_dir / f"{symbol}_{interval}.parquet"
            self._save_parquet(df, path)
            logger.info(f"Saved raw → {path}")

        return df

    def fetch_all(
        self,
        symbols: list,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict:
        """Fetch and clean all symbols. Returns {symbol: cleaned_df}."""
        results = {}
        for sym in symbols:
            try:
                raw = self.fetch(sym, start, end, interval)
                cleaned = self.clean(raw, sym, interval)
                report = self.validate(cleaned, sym)
                logger.info(f"{sym} validation: {report}")
                results[sym] = cleaned
            except Exception as exc:
                logger.error(f"Failed to fetch/clean {sym}: {exc}")
        return results

    def load(
        self,
        symbol: str,
        interval: str = "1d",
        processed: bool = True,
    ) -> pd.DataFrame:
        """Load a Parquet file from processed/ (or raw/)."""
        directory = self.processed_dir if processed else self.raw_dir
        path = directory / f"{symbol}_{interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No Parquet file found at {path}. "
                f"Run fetch_all() first to download and clean the data."
            )
        return self._load_parquet(path)

    def clean(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Cleaning pipeline:
          1. Drop rows where close <= 0
          2. Forward-fill up to 2 consecutive NaNs (handles weekend/holiday gaps)
          3. Drop any remaining NaN rows
          4. Cast OHLCV to float64
          5. Sort index ascending, drop duplicate dates
          6. Add 'returns' = close.pct_change()
          7. Add 'log_returns' = log(close / close.shift(1))
        Saves result to processed/.
        """
        df = df.copy()

        # 1. Drop non-positive closes
        before = len(df)
        df = df[df["close"] > 0]
        dropped = before - len(df)
        if dropped:
            logger.warning(f"{symbol}: dropped {dropped} rows with close <= 0")

        # 2. Forward-fill up to 2 consecutive NaNs
        df = df.ffill(limit=2)

        # 3. Drop remaining NaN rows
        df = df.dropna(subset=self.REQUIRED_COLS)

        # 4. Cast to float64
        for col in self.REQUIRED_COLS:
            df[col] = df[col].astype("float64")

        # 5. Sort and deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # 6-7. Derived columns
        df["returns"]     = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        path = self.processed_dir / f"{symbol}_{interval}.parquet"
        self._save_parquet(df, path)
        logger.info(f"Saved processed → {path} ({len(df)} rows)")

        return df

    def validate(self, df: pd.DataFrame, symbol: str) -> dict:
        """Return a quality report dict for a DataFrame."""
        if df.empty:
            return {"error": "empty dataframe"}

        missing_pct = float(df[self.REQUIRED_COLS].isna().any(axis=1).mean())
        price_anomalies = int((df["close"] <= 0).sum())

        # Detect gaps > 3 calendar days (for daily bars)
        idx = df.index
        gaps = []
        if len(idx) > 1:
            deltas   = idx.to_series().diff().iloc[1:]
            gap_mask = deltas > pd.Timedelta(days=3)
            gaps = [str(d.date()) for d in deltas.index[gap_mask]]

        dtype_ok = all(str(df[c].dtype).startswith("float") for c in self.REQUIRED_COLS)

        report = {
            "row_count":        len(df),
            "date_range":       (str(df.index[0].date()), str(df.index[-1].date())),
            "missing_pct":      round(missing_pct, 6),
            "price_anomalies":  price_anomalies,
            "gaps":             gaps[:5],  # show first 5 only
            "dtype_ok":         dtype_ok,
        }

        if missing_pct > 0:
            logger.warning(f"{symbol}: {missing_pct:.2%} missing values in OHLCV columns")
        if price_anomalies:
            logger.warning(f"{symbol}: {price_anomalies} rows with close <= 0")

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

        if raw.empty:
            raise ValueError(f"yfinance returned no data for {symbol!r} [{start} → {end}]")

        # Normalise column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]

        # Keep only OHLCV
        raw = raw[self.REQUIRED_COLS].copy()

        # Ensure UTC-normalised DatetimeIndex
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert("UTC").tz_localize(None)
        else:
            raw.index = pd.to_datetime(raw.index)

        return raw

    def _fetch_binance(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Fetch via Binance public REST API (no API key needed for klines).
        Symbol format: 'BTCUSDT' (not 'BTC-USD').
        Falls back to yfinance for Yahoo-style tickers.
        """
        try:
            from binance.client import Client  # type: ignore
        except ImportError:
            logger.warning("python-binance not installed; falling back to yfinance")
            return self._fetch_yfinance(symbol, start, end, interval)

        interval_map = {
            "1d":  Client.KLINE_INTERVAL_1DAY,
            "1h":  Client.KLINE_INTERVAL_1HOUR,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
        }
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1DAY)

        client = Client()  # unauthenticated
        klines = client.get_historical_klines(symbol, binance_interval, start, end)

        if not klines:
            raise ValueError(f"Binance returned no data for {symbol!r}")

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        df = df[self.REQUIRED_COLS].astype(float)
        df.index.name = "Date"
        return df

    def _save_parquet(self, df: pd.DataFrame, path: Path) -> None:
        os.makedirs(path.parent, exist_ok=True)
        df.to_parquet(path, engine=self.engine)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine=self.engine)
