"""
feature_engineer.py — Feature engineering pipeline.

Groups:
  1. Returns          — log returns, rolling returns (5d/10d/20d/60d)
  2. Volatility       — rolling std of returns, ATR, Bollinger bandwidth
  3. Trend/Momentum   — SMA, EMA, MACD, RSI, ROC, price-relative-to-MA
  4. Bollinger Bands  — %B position, upper/lower distance
  5. Volume           — volume ratio, z-score anomaly
  6. Regime proxy     — simple vol regime label (low/mid/high)
  7. Targets          — forward returns + direction labels (ML Phase 5 only)

No-lookahead guarantee:
  All feature columns at row i use only data up to and including row i.
  Target columns (prefix 'target_') use future data — strip before inference.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ta

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class FeatureEngineer:
    """Build a rich feature matrix from a clean OHLCV DataFrame."""

    # Feature group prefixes — used by get_feature_names() filtering
    FEATURE_GROUPS = ["ret_", "vol_", "trend_", "bb_", "volume_", "regime_", "mtf_"]
    TARGET_PREFIX  = "target_"

    def __init__(self, drop_warmup: bool = True, multi_timeframe: bool = False):
        """
        Parameters
        ----------
        drop_warmup : bool
            If True, drop the first N rows where slow indicators are NaN.
            Recommended True for ML training; False to preserve full index.
        multi_timeframe : bool
            If True, also compute weekly-resampled features and broadcast
            them back to the daily index. Adds the "mtf_" feature group.
        """
        self.drop_warmup     = drop_warmup
        self.multi_timeframe = multi_timeframe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all feature columns (no targets).
        Returns a new DataFrame with OHLCV + all features; no target columns.
        """
        out = df.copy()
        out = self._add_return_features(out)
        out = self._add_volatility_features(out)
        out = self._add_trend_features(out)
        out = self._add_bollinger_features(out)
        out = self._add_volume_features(out)
        out = self._add_regime_features(out)

        if self.multi_timeframe:
            out = self._add_multi_timeframe_features(out)

        if self.drop_warmup:
            feature_cols = self.get_feature_names(out)
            out = out.dropna(subset=feature_cols)

        return out

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add forward-return target columns.
        ⚠️  These use future data — NEVER use in live inference or feature construction.
        Intended only for ML training labels (Phase 5).
        """
        out = df.copy()
        close = out["close"]

        # 1-day forward return
        out["target_ret_1d"]       = close.shift(-1) / close - 1
        # 5-day forward return
        out["target_ret_5d"]       = close.shift(-5) / close - 1
        # Binary direction: 1 = up tomorrow, 0 = down/flat
        out["target_direction_1d"] = (out["target_ret_1d"] > 0).astype(int)
        # Binary direction: 1 = up in 5 days, 0 = down/flat
        out["target_direction_5d"] = (out["target_ret_5d"] > 0).astype(int)

        return out

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features + targets in one DataFrame. Ready for ML training."""
        feat = self.build_features(df)
        out  = self.build_targets(feat)
        # Drop last 5 rows where 5d targets are NaN
        out  = out.dropna(subset=["target_ret_5d"])
        return out

    def get_feature_names(self, df: pd.DataFrame = None) -> list:
        """
        Return list of feature column names (all columns with group prefixes).
        If df is provided, filters to columns actually present.
        """
        all_prefixes = tuple(self.FEATURE_GROUPS)
        if df is None:
            return []
        return [c for c in df.columns if c.startswith(all_prefixes)]

    def get_target_names(self, df: pd.DataFrame = None) -> list:
        if df is None:
            return []
        return [c for c in df.columns if c.startswith(self.TARGET_PREFIX)]

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        features_dir: str | Path,
        interval: str = "1d",
        engine: str = "pyarrow",
    ) -> Path:
        """Save feature DataFrame to Parquet."""
        os.makedirs(features_dir, exist_ok=True)
        path = Path(features_dir) / f"{symbol}_{interval}_features.parquet"
        df.to_parquet(path, engine=engine)
        return path

    def load(
        self,
        symbol: str,
        features_dir: str | Path,
        interval: str = "1d",
        engine: str = "pyarrow",
    ) -> pd.DataFrame:
        path = Path(features_dir) / f"{symbol}_{interval}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No feature file at {path}. Run build_all() + save() first."
            )
        return pd.read_parquet(path, engine=engine)

    # ------------------------------------------------------------------
    # Feature group builders (private)
    # ------------------------------------------------------------------

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log returns and rolling return windows."""
        close = df["close"]

        # Already in processed data — recompute to be safe
        df["ret_1d"]       = close.pct_change()
        df["ret_log_1d"]   = np.log(close / close.shift(1))

        # Rolling cumulative returns (no lookahead: uses close[i] vs close[i-N])
        for w in [5, 10, 20, 60]:
            df[f"ret_{w}d"] = close / close.shift(w) - 1

        # Momentum: return over past 20d minus past 5d (captures intermediate trend)
        df["ret_momentum"] = df["ret_20d"] - df["ret_5d"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realised volatility at multiple windows, ATR."""
        ret = df["ret_log_1d"]

        for w in [5, 10, 20, 60]:
            df[f"vol_realised_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

        # Volatility ratio: short-term vs long-term (regime proxy)
        df["vol_ratio_5_20"] = df["vol_realised_5d"] / df["vol_realised_20d"].replace(0, np.nan)

        # ATR (14-period) — normalised by close for cross-asset comparability
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df["vol_atr_14"]        = atr_indicator.average_true_range()
        df["vol_atr_14_pct"]    = df["vol_atr_14"] / df["close"]  # % of price

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMAs, EMAs, MACD, RSI, ROC, price-relative-to-MA."""
        close = df["close"]

        # --- Moving averages ---
        for w in [10, 20, 50, 200]:
            df[f"trend_sma_{w}"] = close.rolling(w).mean()
            # Price distance from MA (normalised): (close - SMA) / SMA
            df[f"trend_close_vs_sma{w}"] = (close - df[f"trend_sma_{w}"]) / df[f"trend_sma_{w}"]

        for w in [12, 26]:
            df[f"trend_ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        # --- MACD ---
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["trend_macd"]          = macd_ind.macd()
        df["trend_macd_signal"]   = macd_ind.macd_signal()
        df["trend_macd_hist"]     = macd_ind.macd_diff()
        # Normalise MACD by price to make it cross-asset comparable
        df["trend_macd_norm"]     = df["trend_macd"] / close

        # --- RSI ---
        df["trend_rsi_14"]  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["trend_rsi_7"]   = ta.momentum.RSIIndicator(close=close, window=7).rsi()

        # RSI slope: rate of change of RSI over 5 bars
        df["trend_rsi_slope"] = df["trend_rsi_14"].diff(5)

        # --- Rate of Change ---
        for w in [5, 10, 20]:
            df[f"trend_roc_{w}d"] = ta.momentum.ROCIndicator(close=close, window=w).roc()

        # --- SMA crossover signals (continuous: fast/slow ratio) ---
        df["trend_sma_20_50_ratio"] = df["trend_sma_20"] / df["trend_sma_50"] - 1
        df["trend_sma_50_200_ratio"] = (
            df["trend_sma_50"] / df["trend_sma_200"].replace(0, np.nan) - 1
        )

        # --- ADX (trend strength) ---
        adx_ind = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=close, window=14
        )
        df["trend_adx"]  = adx_ind.adx()
        df["trend_dip"]  = adx_ind.adx_neg()
        df["trend_dipos"] = adx_ind.adx_pos()

        return df

    def _add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band position (%B), bandwidth, distance to bands."""
        close = df["close"]
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        mid   = bb.bollinger_mavg()

        df["bb_upper"]     = upper
        df["bb_lower"]     = lower
        df["bb_mid"]       = mid

        # %B: where is close within the bands? 0=lower, 0.5=middle, 1=upper
        band_width = (upper - lower).replace(0, np.nan)
        df["bb_pct_b"]     = (close - lower) / band_width

        # Bandwidth: (upper - lower) / mid — measures band expansion/contraction
        df["bb_bandwidth"] = band_width / mid.replace(0, np.nan)

        # Normalised distance from close to each band
        df["bb_dist_upper"] = (upper - close) / close
        df["bb_dist_lower"] = (close - lower) / close

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume ratio, z-score anomaly, OBV trend."""
        volume = df["volume"].replace(0, np.nan)

        # Volume relative to 20-day mean
        vol_ma20 = volume.rolling(20).mean()
        df["volume_ratio_20d"] = volume / vol_ma20.replace(0, np.nan)

        # Z-score over 20-day window: how unusual is today's volume?
        vol_std20 = volume.rolling(20).std()
        df["volume_zscore_20d"] = (volume - vol_ma20) / vol_std20.replace(0, np.nan)

        # Directional volume: positive when price up, negative when price down
        sign = np.sign(df["close"].diff())
        df["volume_signed"] = volume * sign

        # Rolling 5-day sum of signed volume (buying/selling pressure)
        df["volume_pressure_5d"] = df["volume_signed"].rolling(5).sum()
        # Normalise by 20d avg volume
        df["volume_pressure_norm"] = df["volume_pressure_5d"] / vol_ma20.replace(0, np.nan)

        # OBV (On-Balance Volume) — normalised trend
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"]
        ).on_balance_volume()
        # OBV slope: 10-day change (normalised by its own rolling std)
        obv_std = obv.rolling(20).std().replace(0, np.nan)
        df["volume_obv_slope"] = obv.diff(10) / obv_std

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple volatility-based regime proxy.
        Regime label: 0=low_vol, 1=mid_vol, 2=high_vol
        Based on 20d realised volatility percentile (rolling 252-bar window).
        Also includes a trend-strength label.
        """
        vol20 = df.get("vol_realised_20d")
        if vol20 is None:
            return df

        # Rolling percentile rank of current vol vs past 252 bars
        def rolling_percentile(series, window=252):
            return series.rolling(window, min_periods=30).apply(
                lambda x: (x[-1] > x[:-1]).mean(), raw=True
            )

        df["regime_vol_pct"] = rolling_percentile(vol20)

        # Discretise into 3 buckets: low (0–33%), mid (33–67%), high (67–100%)
        df["regime_vol_label"] = pd.cut(
            df["regime_vol_pct"],
            bins=[-0.001, 0.33, 0.67, 1.001],
            labels=[0, 1, 2],
        ).astype(float)

        # Trend regime: is price above its 50-day SMA?
        sma50 = df.get("trend_sma_50")
        if sma50 is not None:
            df["regime_trend_label"] = (df["close"] > sma50).astype(float)

        return df

    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-timeframe (MTF) features: compute on weekly-resampled data,
        then broadcast back to the daily index using forward-fill.

        Pipeline:
          daily close -> resample to weekly (Friday close)
                      -> compute weekly indicators
                      -> reindex to daily and forward-fill

        ⚠️  No-lookahead: each weekly bar is "released" at its own date
        (the Friday close), and forward-filled into the following week.
        We then `shift(1)` so that today's bar uses LAST week's reading,
        guaranteeing past-only data.
        """
        # Resample close to weekly (Friday close — week-ending convention)
        weekly_close = df["close"].resample("W-FRI").last()

        # Need at least a few weeks of data
        if len(weekly_close) < 30:
            return df

        # --- Weekly returns ---
        weekly_ret_1 = weekly_close.pct_change()
        weekly_ret_4 = weekly_close / weekly_close.shift(4) - 1
        weekly_ret_12 = weekly_close / weekly_close.shift(12) - 1

        # --- Weekly trend: 10-week and 30-week SMAs ---
        weekly_sma_10 = weekly_close.rolling(10).mean()
        weekly_sma_30 = weekly_close.rolling(30).mean()
        weekly_close_vs_sma10 = (weekly_close - weekly_sma_10) / weekly_sma_10
        weekly_sma_ratio = weekly_sma_10 / weekly_sma_30.replace(0, np.nan) - 1

        # --- Weekly RSI ---
        weekly_rsi = ta.momentum.RSIIndicator(close=weekly_close, window=14).rsi()

        # --- Weekly volatility (rolling std of weekly returns) ---
        weekly_vol = weekly_ret_1.rolling(8).std() * np.sqrt(52)

        # Combine into a weekly DataFrame
        weekly_features = pd.DataFrame({
            "mtf_w_ret_1w":         weekly_ret_1,
            "mtf_w_ret_4w":         weekly_ret_4,
            "mtf_w_ret_12w":        weekly_ret_12,
            "mtf_w_close_vs_sma10": weekly_close_vs_sma10,
            "mtf_w_sma_ratio":      weekly_sma_ratio,
            "mtf_w_rsi":            weekly_rsi,
            "mtf_w_vol":            weekly_vol,
        })

        # Broadcast to daily index: reindex with forward-fill
        # Then shift by 1 day so today's bar uses LAST week's value
        # (no lookahead — week-ending Friday becomes available Monday)
        daily_mtf = weekly_features.reindex(df.index, method="ffill").shift(1)

        # Add to df
        for col in daily_mtf.columns:
            df[col] = daily_mtf[col]

        return df
