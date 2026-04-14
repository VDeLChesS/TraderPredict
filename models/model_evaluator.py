"""
model_evaluator.py -- Evaluation, plotting, and reporting for DirectionClassifier.
ASCII-only output for Windows console compatibility.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ------------------------------------------------------------------
# Text reports
# ------------------------------------------------------------------

def print_train_test_report(result: dict, symbol: str, model_type: str) -> None:
    sep = "=" * 60
    dash = "-" * 60
    print("")
    print(sep)
    print(f"  {symbol}  |  {model_type.upper()}  |  Train/Test Evaluation")
    print(sep)
    print(f"  Train: {result['train_start']} to {result['train_end']}  ({result['train_rows']} bars)")
    print(f"  Test:  {result['test_start']}  to {result['test_end']}   ({result['test_rows']} bars)")
    print(dash)
    print(f"  {'Metric':<20} {'Model':>10}  {'Baseline (always up)':>20}")
    print(f"  {'Accuracy':<20} {result['accuracy']:>10.4f}  {result['baseline_acc']:>20.4f}")
    print(f"  {'AUC-ROC':<20} {result['auc']:>10.4f}  {'0.5000':>20}")
    print(f"  {'F1 Score':<20} {result['f1']:>10.4f}  {'n/a':>20}")
    print(f"  {'Precision':<20} {result['precision']:>10.4f}  {'n/a':>20}")
    print(f"  {'Recall':<20} {result['recall']:>10.4f}  {'n/a':>20}")
    beat = result['accuracy'] > result['baseline_acc']
    print(dash)
    print(f"  Beats baseline:  {'YES' if beat else 'NO'}")
    print(f"  AUC > 0.52:      {'YES' if result['auc'] > 0.52 else 'NO'}")
    print(sep)


def print_walkforward_report(cv_result: dict, symbol: str, model_type: str) -> None:
    n = len(cv_result["fold_metrics"])
    dash = "-" * 62
    print(f"\n  Walk-Forward CV [{symbol} | {model_type.upper()}]  ({n} folds)")
    print(f"  {'Fold':<6} {'Train rows':>10} {'Test rows':>9} {'Accuracy':>9} {'AUC':>7} {'F1':>7} {'Baseline':>9}")
    print(f"  {dash}")
    for fm in cv_result["fold_metrics"]:
        print(
            f"  {fm['fold']:<6} {fm['train_rows']:>10} {fm['test_rows']:>9} "
            f"{fm['accuracy']:>9.4f} {fm['auc']:>7.4f} {fm['f1']:>7.4f} {fm['baseline']:>9.4f}"
        )
    print(f"  {dash}")
    print(
        f"  {'Mean':<6} {'':>10} {'':>9} "
        f"{cv_result['mean_accuracy']:>9.4f} {cv_result['mean_auc']:>7.4f} {cv_result['mean_f1']:>7.4f}"
    )
    print(
        f"  {'Std':<6} {'':>10} {'':>9} "
        f"{cv_result['std_accuracy']:>9.4f} {cv_result['std_auc']:>7.4f} {cv_result['std_f1']:>7.4f}"
    )


# ------------------------------------------------------------------
# Shared plot helper
# ------------------------------------------------------------------

def _save_fig(fig, results_dir: str, filename: str, dpi: int = 100) -> str:
    """Save figure to results_dir/filename, close it, and return the full path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    symbol: str,
    model_type: str,
    top_n: int = 25,
    results_dir: str = "results",
) -> str:
    top = importance.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top.index, top.values, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{symbol} -- Top {top_n} Feature Importances ({model_type.upper()})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, results_dir, f"{symbol}_{model_type}_feature_importance.png")


def plot_roc_curve(
    y_true: pd.Series,
    y_proba: pd.Series,
    symbol: str,
    model_type: str,
    results_dir: str = "results",
) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = sklearn_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{symbol} -- ROC Curve ({model_type.upper()})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, results_dir, f"{symbol}_{model_type}_roc_curve.png")


def plot_probability_calibration(
    y_true: pd.Series,
    y_proba: pd.Series,
    symbol: str,
    model_type: str,
    results_dir: str = "results",
    n_bins: int = 10,
) -> str:
    """Decile calibration: predicted P(up) vs actual win rate per decile."""
    num_bins = int(n_bins)
    df = pd.DataFrame({"proba": y_proba.values, "actual": y_true.values})
    # Manual rank-based binning
    df["rank"] = df["proba"].rank(method="first")
    max_rank = df["rank"].max()
    df["decile"] = ((df["rank"] - 1) / max_rank * num_bins).astype(int).clip(0, num_bins - 1)
    cal = df.groupby("decile").agg(
        mean_proba=("proba", "mean"),
        actual_win_rate=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(cal["decile"], cal["actual_win_rate"],
           color="#4CAF50", alpha=0.7, label="Actual win rate")
    ax.plot(cal["decile"], cal["mean_proba"],
            color="#F44336", linewidth=2, marker="o", label="Mean predicted prob")
    ax.axhline(float(y_true.mean()), color="grey", linestyle="--", linewidth=1,
               label=f"Overall win rate ({float(y_true.mean()):.2%})")
    ax.set_xlabel("Probability Decile (0=lowest, 9=highest)")
    ax.set_ylabel("Win Rate / Probability")
    ax.set_title(f"{symbol} -- Probability Calibration ({model_type.upper()})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return _save_fig(fig, results_dir, f"{symbol}_{model_type}_calibration.png")


def plot_walkforward_accuracy(
    cv_result: dict,
    symbol: str,
    model_type: str,
    results_dir: str = "results",
) -> str:
    """Bar chart: accuracy + AUC per fold vs always-long baseline."""
    folds    = [m["fold"]     for m in cv_result["fold_metrics"]]
    accs     = [m["accuracy"] for m in cv_result["fold_metrics"]]
    aucs     = [m["auc"]      for m in cv_result["fold_metrics"]]
    baseline = [m["baseline"] for m in cv_result["fold_metrics"]]

    x = np.arange(len(folds))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, accs,     width, label="Accuracy",             color="#2196F3", alpha=0.8)
    ax.bar(x,         aucs,     width, label="AUC-ROC",              color="#4CAF50", alpha=0.8)
    ax.bar(x + width, baseline, width, label="Baseline (always up)", color="#FF9800", alpha=0.6)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Score")
    ax.set_title(f"{symbol} -- Walk-Forward CV ({model_type.upper()})")
    ax.legend()
    ax.set_ylim(0.3, 0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, results_dir, f"{symbol}_{model_type}_walkforward_cv.png")
