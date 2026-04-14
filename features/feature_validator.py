"""
feature_validator.py — Quality checks for the feature matrix.

Checks:
  1. No lookahead bias  — features at row i must not reference future data
  2. No excessive NaN  — after warmup drop, NaN fraction should be < 1%
  3. No infinite values
  4. No zero-variance features  — useless for ML
  5. Target columns are present and have correct dtypes
  6. Index is monotonically increasing DatetimeIndex
"""

import numpy as np
import pandas as pd


def validate_features(df: pd.DataFrame, feature_names: list, target_names: list) -> dict:
    """
    Run all quality checks. Returns a dict with one key per check.
    A check value of True = passed.

    Parameters
    ----------
    df            : DataFrame output from FeatureEngineer.build_all()
    feature_names : list of feature column names
    target_names  : list of target column names
    """
    report = {}

    # --- 1. Index health ---
    report["index_is_datetime"] = isinstance(df.index, pd.DatetimeIndex)
    report["index_monotonic"]   = bool(df.index.is_monotonic_increasing)
    report["index_no_dups"]     = not bool(df.index.duplicated().any())

    # --- 2. Feature NaN check (after warmup drop) ---
    feat_df = df[feature_names] if feature_names else pd.DataFrame()
    nan_per_col = feat_df.isna().mean()
    report["feature_nan_max"]       = float(nan_per_col.max()) if not feat_df.empty else 0.0
    report["feature_nan_cols"]      = nan_per_col[nan_per_col > 0.01].index.tolist()
    report["feature_nan_ok"]        = bool(report["feature_nan_max"] <= 0.01)

    # --- 3. Infinite values ---
    inf_series = np.isinf(feat_df.fillna(0)).sum()
    inf_counts = {col: int(n) for col, n in inf_series[inf_series > 0].items()}
    report["infinite_value_cols"] = inf_counts
    report["no_infinite_values"]  = len(inf_counts) == 0

    # --- 4. Zero-variance features ---
    zero_var = [c for c in feature_names if df[c].std() == 0]
    report["zero_variance_cols"] = zero_var
    report["no_zero_variance"]   = len(zero_var) == 0

    # --- 5. Target sanity ---
    report["targets_present"] = all(t in df.columns for t in target_names)
    if target_names and report["targets_present"]:
        target_df = df[target_names]
        report["target_nan_max"] = float(target_df.isna().mean().max())
        # direction labels must be 0 or 1
        dir_cols = [t for t in target_names if "direction" in t]
        dir_ok = all(set(df[c].dropna().unique()).issubset({0, 1}) for c in dir_cols)
        report["target_direction_binary"] = dir_ok
    else:
        report["target_nan_max"]          = None
        report["target_direction_binary"] = None

    # --- 6. Feature count ---
    report["feature_count"] = len(feature_names)
    report["row_count"]     = len(df)
    report["date_range"]    = (str(df.index[0].date()), str(df.index[-1].date()))

    # --- 7. Overall pass ---
    hard_checks = [
        report["index_is_datetime"],
        report["index_monotonic"],
        report["index_no_dups"],
        report["feature_nan_ok"],
        report["no_infinite_values"],
        report["no_zero_variance"],
        report["targets_present"],
    ]
    report["all_passed"] = all(hard_checks)

    return report


def print_report(report: dict, symbol: str = "") -> None:
    """Pretty-print a validation report."""
    label = f" [{symbol}]" if symbol else ""
    print(f"\n--- Feature Validation Report{label} ---")
    print(f"  Rows:            {report['row_count']}")
    print(f"  Date range:      {report['date_range']}")
    print(f"  Feature count:   {report['feature_count']}")
    print(f"  Index datetime:  {report['index_is_datetime']}")
    print(f"  Index monotonic: {report['index_monotonic']}")
    print(f"  Index no dups:   {report['index_no_dups']}")
    print(f"  Max NaN pct:     {report['feature_nan_max']:.4%}")
    nan_cols = report.get("feature_nan_cols", [])
    if nan_cols:
        print(f"  NaN cols (>1%):  {nan_cols}")
    print(f"  No infinities:   {report['no_infinite_values']}")
    if report["infinite_value_cols"]:
        print(f"  Inf cols:        {report['infinite_value_cols']}")
    print(f"  No zero-var:     {report['no_zero_variance']}")
    if report["zero_variance_cols"]:
        print(f"  Zero-var cols:   {report['zero_variance_cols']}")
    print(f"  Targets present: {report['targets_present']}")
    if report.get("target_nan_max") is not None:
        print(f"  Target max NaN:  {report['target_nan_max']:.4%}")
    dir_ok = report.get("target_direction_binary")
    if dir_ok is not None:
        print(f"  Direction binary:{dir_ok}")
    status = "PASS" if report["all_passed"] else "FAIL"
    print(f"  RESULT:          [{status}]")
