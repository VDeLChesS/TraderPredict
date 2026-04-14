"""
train_models.py -- Reusable model training script.

Trains XGBoost and LightGBM direction classifiers for all configured symbols.
Each model is saved to models/saved/{sym_file}_{model_type}.pkl.

Usage:
    python scripts/train_models.py                  # Train all symbols
    python scripts/train_models.py --symbols BTC-USD SPY
    python scripts/train_models.py --target 5d       # Use 5-day direction target
"""

import argparse
import os
import sys
import warnings

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

from config import (
    SYMBOLS, START_DATE, END_DATE,
    DATA_RAW_DIR, DATA_PROCESSED_DIR,
)
from data_loader.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.direction_classifier import DirectionClassifier


MODEL_DIR = Path(__file__).parent.parent / "models" / "saved"


def train_and_save(
    symbol: str,
    model_type: str,
    target_col: str = "target_direction_1d",
    model_dir: Path = MODEL_DIR,
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    train_pct: float = 0.80,
    suffix: str = "",
) -> dict:
    """
    Load OHLCV -> build features+targets -> train_test_evaluate -> save .pkl.

    Parameters
    ----------
    symbol     : e.g. "BTC-USD", "AAPL"
    model_type : "xgboost" or "lightgbm"
    target_col : target column name (default: target_direction_1d)
    model_dir  : directory to save .pkl files
    suffix     : optional suffix for filename (e.g. "_5d")

    Returns
    -------
    dict with training metrics + file path
    """
    loader = DataLoader(raw_dir, processed_dir)

    # Load or fetch data
    try:
        df = loader.load(symbol)
    except FileNotFoundError:
        print(f"  Fetching {symbol} from yfinance...")
        raw = loader.fetch(symbol, START_DATE, END_DATE)
        df = loader.clean(raw, symbol)

    print(f"  {symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Build features + targets
    fe = FeatureEngineer(drop_warmup=True)
    df_all = fe.build_all(df)
    feat_cols = fe.get_feature_names(df_all)

    X = df_all[feat_cols]
    y = df_all[target_col]

    print(f"  Features: {len(feat_cols)} columns, {len(X)} rows")
    print(f"  Target: {target_col} (baseline={y.mean():.3f})")

    # Train and evaluate
    clf = DirectionClassifier(model_type=model_type)
    metrics = clf.train_test_evaluate(X, y, train_pct=train_pct)

    # Save model
    sym_file = symbol.replace("-", "_")
    os.makedirs(str(model_dir), exist_ok=True)
    filename = f"{sym_file}_{model_type}{suffix}.pkl"
    path = model_dir / filename
    clf.save(path)

    result = {
        "symbol":    symbol,
        "model":     model_type,
        "target":    target_col,
        "file":      str(path),
        "accuracy":  metrics["accuracy"],
        "auc":       metrics["auc"],
        "f1":        metrics["f1"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "baseline":  metrics["baseline_acc"],
        "train_rows": metrics["train_rows"],
        "test_rows":  metrics["test_rows"],
    }

    print(f"  Saved: {filename}  "
          f"acc={metrics['accuracy']:.4f}  "
          f"auc={metrics['auc']:.4f}  "
          f"f1={metrics['f1']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train ML models for TraderPredict")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to train (default: all from config)")
    parser.add_argument("--target", default="1d", choices=["1d", "5d"],
                        help="Target horizon: 1d or 5d")
    parser.add_argument("--model-types", nargs="+", default=["xgboost", "lightgbm"],
                        help="Model types to train")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    target_col = f"target_direction_{args.target}"
    suffix = "" if args.target == "1d" else f"_{args.target}"

    print("=" * 70)
    print("  TraderPredict -- Model Training")
    print(f"  Symbols: {symbols}")
    print(f"  Target:  {target_col}")
    print(f"  Models:  {args.model_types}")
    print("=" * 70)

    all_results = []

    for sym in symbols:
        print(f"\n{'-' * 50}")
        print(f"  Training: {sym}")
        print(f"{'-' * 50}")

        for model_type in args.model_types:
            try:
                result = train_and_save(
                    symbol=sym,
                    model_type=model_type,
                    target_col=target_col,
                    suffix=suffix,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {sym}/{model_type}: {e}")

    # Summary table
    if all_results:
        df_summary = pd.DataFrame(all_results)
        cols = ["symbol", "model", "accuracy", "auc", "f1", "baseline"]
        print(f"\n{'=' * 70}")
        print("  Training Summary")
        print(f"{'=' * 70}")
        print(df_summary[cols].to_string(index=False))
        print(f"\n  {len(all_results)} models trained and saved to: {MODEL_DIR}")

        # Save summary CSV
        summary_path = MODEL_DIR.parent / "training_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"  Summary CSV: {summary_path}")

    print(f"\n{'=' * 70}")
    print("  Training complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
