"""
train_models.py -- Reusable model training script.

Trains XGBoost and LightGBM direction classifiers for all configured symbols.
Each model is saved to models/saved/{sym_file}_{model_type}{suffix}.pkl.

File-naming suffixes:
  ""        -> default daily target, no MTF, no tuning
  "_5d"     -> 5-day direction target
  "_mtf"    -> multi-timeframe features included
  "_tuned"  -> Optuna-tuned hyperparameters
  "_5d_mtf_tuned" -> all of the above

Usage:
    python scripts/train_models.py                            # Train all symbols (defaults)
    python scripts/train_models.py --symbols BTC-USD SPY
    python scripts/train_models.py --target 5d                # 5-day direction target
    python scripts/train_models.py --multi-timeframe          # Add weekly features
    python scripts/train_models.py --tune --n-trials 30       # Optuna hyperparameter tuning
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
    multi_timeframe: bool = False,
    tune: bool = False,
    n_trials: int = 30,
) -> dict:
    """
    Load OHLCV -> build features+targets -> train_test_evaluate -> save .pkl.

    Parameters
    ----------
    symbol     : e.g. "BTC-USD", "AAPL"
    model_type : "xgboost" or "lightgbm"
    target_col : target column name (default: target_direction_1d)
    model_dir  : directory to save .pkl files
    suffix     : optional suffix for filename (e.g. "_5d", "_mtf", "_tuned")
    multi_timeframe : include weekly-resampled features
    tune       : run Optuna hyperparameter tuning before final training
    n_trials   : Optuna trials when tune=True

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
    fe = FeatureEngineer(drop_warmup=True, multi_timeframe=multi_timeframe)
    df_all = fe.build_all(df)
    feat_cols = fe.get_feature_names(df_all)

    X = df_all[feat_cols]
    y = df_all[target_col]

    print(f"  Features: {len(feat_cols)} columns, {len(X)} rows  "
          f"(MTF={'yes' if multi_timeframe else 'no'})")
    print(f"  Target: {target_col} (baseline={y.mean():.3f})")

    # Optional: Optuna hyperparameter tuning
    best_params = None
    if tune:
        from models.optuna_tuner import OptunaTuner
        # Use only the training portion for tuning, to avoid OOS leakage
        split = int(len(X) * train_pct)
        X_tune = X.iloc[:split]
        y_tune = y.iloc[:split]

        print(f"  Tuning {model_type} with Optuna ({n_trials} trials)...")
        tuner = OptunaTuner(model_type=model_type, n_trials=n_trials)
        best_params, best_auc = tuner.tune(X_tune, y_tune)
        print(f"  Best CV AUC: {best_auc:.4f}")
        print(f"  Best params: {best_params}")
        clf = tuner.build_best_classifier()
    else:
        clf = DirectionClassifier(model_type=model_type)

    # Final train-test evaluation
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
    parser.add_argument("--multi-timeframe", action="store_true",
                        help="Include weekly-resampled (MTF) features")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials when --tune is set")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    target_col = f"target_direction_{args.target}"

    # Build filename suffix from flags
    suffix_parts = []
    if args.target != "1d":
        suffix_parts.append(args.target)
    if args.multi_timeframe:
        suffix_parts.append("mtf")
    if args.tune:
        suffix_parts.append("tuned")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    print("=" * 70)
    print("  TraderPredict -- Model Training")
    print(f"  Symbols: {symbols}")
    print(f"  Target:  {target_col}")
    print(f"  Models:  {args.model_types}")
    print(f"  MTF features: {args.multi_timeframe}")
    print(f"  Optuna tuning: {args.tune}" + (f" ({args.n_trials} trials)" if args.tune else ""))
    print(f"  Suffix:  '{suffix}'")
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
                    multi_timeframe=args.multi_timeframe,
                    tune=args.tune,
                    n_trials=args.n_trials,
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
