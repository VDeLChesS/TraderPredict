"""
report_generator.py -- Generate comprehensive results from pipeline output.

Produces:
  1. Console summary tables (per symbol + cross-symbol leaderboard)
  2. CSV export: results/full_comparison.csv
  3. Equity curve plots: results/{symbol}_multi_equity.png
  4. Best strategy selection per symbol
  5. Walk-forward summary: in-sample vs OOS degradation (Item 4)
  6. Walk-forward CSV: results/walkforward_comparison.csv
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtesting.engine import BacktestEngine


class ReportGenerator:
    """
    Generate reports from MultiStrategyPipeline output.

    Usage:
        gen = ReportGenerator(pipeline_results, results_dir)
        gen.print_summary()
        gen.save_csv()
        gen.plot_equity_curves()
        gen.print_leaderboard()
        gen.print_walkforward_summary()
        gen.save_walkforward_csv()
    """

    def __init__(self, results: dict, results_dir: str = "results"):
        self.results     = results
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print per-symbol comparison tables."""
        for sym, sym_data in self.results.items():
            comp = sym_data["comparison"]
            sep = "=" * 90
            print(f"\n{sep}")
            print(f"  {sym} -- Strategy Comparison")
            print(f"{sep}")

            # Pivot: rows=strategy, columns grouped by risk_config
            for risk_name in comp["risk_config"].unique():
                subset = comp[comp["risk_config"] == risk_name].copy()
                subset = subset.set_index("strategy")
                cols = ["total_return", "sharpe", "sortino", "max_drawdown",
                        "win_rate", "profit_factor", "num_trades"]
                cols = [c for c in cols if c in subset.columns]

                print(f"\n  Risk Config: {risk_name}")
                print(f"  {'-'*86}")
                print(f"  {'Strategy':<25} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} "
                      f"{'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}")
                print(f"  {'-'*86}")

                for strat_name, row in subset.iterrows():
                    print(
                        f"  {strat_name:<25} "
                        f"{row.get('total_return', 0):>7.2%} "
                        f"{row.get('sharpe', 0):>8.3f} "
                        f"{row.get('sortino', 0):>8.3f} "
                        f"{row.get('max_drawdown', 0):>7.2%} "
                        f"{row.get('win_rate', 0):>8.3f} "
                        f"{row.get('profit_factor', 0):>6.2f} "
                        f"{int(row.get('num_trades', 0)):>7}"
                    )

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def print_leaderboard(self) -> None:
        """
        Cross-symbol leaderboard: best strategy per symbol (by Sharpe ratio,
        using 'no_risk' config for fair comparison).
        """
        sep = "=" * 90
        print(f"\n{sep}")
        print(f"  LEADERBOARD -- Best Strategy per Symbol (no-risk, ranked by Sharpe)")
        print(f"{sep}")
        print(f"  {'Symbol':<12} {'Best Strategy':<25} {'Return':>8} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'Trades':>7}")
        print(f"  {'-'*80}")

        for sym, sym_data in self.results.items():
            comp = sym_data["comparison"]
            no_risk = comp[comp["risk_config"] == "no_risk"].copy()
            if no_risk.empty:
                continue

            best_idx = no_risk["sharpe"].idxmax()
            best = no_risk.loc[best_idx]
            print(
                f"  {sym:<12} {best['strategy']:<25} "
                f"{best['total_return']:>7.2%} "
                f"{best['sharpe']:>8.3f} "
                f"{best['max_drawdown']:>7.2%} "
                f"{int(best['num_trades']):>7}"
            )

        # Risk management impact
        print(f"\n{sep}")
        print(f"  RISK MANAGEMENT IMPACT -- Conservative vs No-Risk (MA Crossover)")
        print(f"{sep}")
        print(f"  {'Symbol':<12} {'Config':<15} {'Return':>8} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'Improvement':>12}")
        print(f"  {'-'*75}")

        for sym, sym_data in self.results.items():
            comp = sym_data["comparison"]
            ma_rows = comp[comp["strategy"] == "MA Crossover"]

            nr = ma_rows[ma_rows["risk_config"] == "no_risk"]
            cv = ma_rows[ma_rows["risk_config"] == "conservative"]

            if nr.empty or cv.empty:
                continue

            nr_dd = nr.iloc[0]["max_drawdown"]
            cv_dd = cv.iloc[0]["max_drawdown"]
            improvement = nr_dd - cv_dd  # positive = conservative has less drawdown

            print(
                f"  {sym:<12} {'no_risk':<15} "
                f"{nr.iloc[0]['total_return']:>7.2%} "
                f"{nr.iloc[0]['sharpe']:>8.3f} "
                f"{nr_dd:>7.2%} "
                f"{'--':>12}"
            )
            print(
                f"  {'':<12} {'conservative':<15} "
                f"{cv.iloc[0]['total_return']:>7.2%} "
                f"{cv.iloc[0]['sharpe']:>8.3f} "
                f"{cv_dd:>7.2%} "
                f"{improvement:>+11.2%}"
            )

        print(f"{sep}")

    # ------------------------------------------------------------------
    # Walk-forward summary (Item 4)
    # ------------------------------------------------------------------

    def print_walkforward_summary(self) -> None:
        """
        Print walk-forward evaluation: in-sample vs out-of-sample metrics.
        Shows which strategies are robust vs overfit.
        """
        sep = "=" * 100
        print(f"\n{sep}")
        print(f"  WALK-FORWARD EVALUATION -- In-Sample vs Out-of-Sample (by Sharpe)")
        print(f"{sep}")

        for sym, sym_data in self.results.items():
            wf = sym_data.get("walkforward", {})
            if not wf:
                continue

            print(f"\n  {sym}  (split: {list(wf.values())[0].get('split_date', '?')})")
            print(f"  {'-'*94}")
            print(f"  {'Strategy':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
                  f"{'Degrad%':>9} {'IS Return':>10} {'OOS Return':>11} "
                  f"{'Robust?':>8}")
            print(f"  {'-'*94}")

            for strat_name, wf_result in wf.items():
                train_m = wf_result["train"]["metrics"]
                test_m  = wf_result["test"]["metrics"]

                sharpe_in  = train_m["sharpe"]
                sharpe_oos = test_m["sharpe"]

                if sharpe_in != 0:
                    degradation = (sharpe_in - sharpe_oos) / abs(sharpe_in)
                else:
                    degradation = 0.0

                # Robust if OOS Sharpe > 0 and degradation < 50%
                robust = "YES" if (sharpe_oos > 0 and degradation < 0.50) else "no"

                print(
                    f"  {strat_name:<25} "
                    f"{sharpe_in:>10.3f} "
                    f"{sharpe_oos:>11.3f} "
                    f"{degradation:>+8.1%} "
                    f"{train_m['total_return']:>9.2%} "
                    f"{test_m['total_return']:>10.2%} "
                    f"{robust:>8}"
                )

        print(f"{sep}")

    # ------------------------------------------------------------------
    # Walk-forward CSV export (Item 4)
    # ------------------------------------------------------------------

    def save_walkforward_csv(self) -> str:
        """Save walk-forward comparison to CSV. Returns file path."""
        rows = []
        for sym, sym_data in self.results.items():
            wf = sym_data.get("walkforward", {})
            for strat_name, wf_result in wf.items():
                train_m = wf_result["train"]["metrics"]
                test_m  = wf_result["test"]["metrics"]

                sharpe_in  = train_m["sharpe"]
                sharpe_oos = test_m["sharpe"]
                degradation = ((sharpe_in - sharpe_oos) / abs(sharpe_in)
                               if sharpe_in != 0 else 0.0)

                rows.append({
                    "symbol":           sym,
                    "strategy":         strat_name,
                    "split_date":       wf_result.get("split_date", ""),
                    "is_total_return":  train_m["total_return"],
                    "is_sharpe":        sharpe_in,
                    "is_sortino":       train_m.get("sortino", 0),
                    "is_max_drawdown":  train_m["max_drawdown"],
                    "is_num_trades":    train_m["num_trades"],
                    "oos_total_return": test_m["total_return"],
                    "oos_sharpe":       sharpe_oos,
                    "oos_sortino":      test_m.get("sortino", 0),
                    "oos_max_drawdown": test_m["max_drawdown"],
                    "oos_num_trades":   test_m["num_trades"],
                    "degradation_pct":  round(degradation, 4),
                })

        if not rows:
            return ""

        df = pd.DataFrame(rows)
        path = os.path.join(self.results_dir, "walkforward_comparison.csv")
        df.to_csv(path, index=False)
        print(f"\n  Walk-forward CSV saved: {path} ({len(df)} rows)")
        return path

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_csv(self) -> str:
        """Save full comparison table to CSV. Returns file path."""
        all_comps = []
        for sym, sym_data in self.results.items():
            all_comps.append(sym_data["comparison"])

        if not all_comps:
            return ""

        full = pd.concat(all_comps, ignore_index=True)
        path = os.path.join(self.results_dir, "full_comparison.csv")
        full.to_csv(path, index=False)
        print(f"\n  CSV saved: {path} ({len(full)} rows)")
        return path

    # ------------------------------------------------------------------
    # Equity curve plots
    # ------------------------------------------------------------------

    def plot_equity_curves(self) -> list:
        """
        For each symbol, plot equity curves of all strategies (no_risk config).
        Returns list of saved paths.
        """
        paths = []

        for sym, sym_data in self.results.items():
            df = sym_data["data"]
            strat_results = sym_data["strategies"]

            fig, ax = plt.subplots(figsize=(14, 7))

            # Buy-and-hold benchmark
            bh = df["close"] / df["close"].iloc[0]
            ax.plot(bh.index, bh, label="Buy & Hold", linewidth=1,
                    linestyle="--", color="grey", alpha=0.7)

            # Each strategy (no_risk only for clean comparison)
            colors = [
                "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336",
                "#009688", "#795548", "#607D8B", "#E91E63", "#3F51B5",
                "#CDDC39", "#FF5722",
            ]
            i = 0
            for strat_name, risk_results in strat_results.items():
                if "no_risk" not in risk_results:
                    continue
                eq = risk_results["no_risk"]["equity_curve"]
                normalised = eq / eq.iloc[0]
                ax.plot(eq.index, normalised, label=strat_name,
                        linewidth=1.5, color=colors[i % len(colors)])
                i += 1

            ax.set_title(f"{sym} -- Multi-Strategy Equity Curves (normalised, fees+slippage)")
            ax.set_ylabel("Portfolio Value (normalised to 1.0)")
            ax.set_xlabel("Date")
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(alpha=0.3)
            plt.tight_layout()

            path = os.path.join(self.results_dir, f"{sym}_multi_equity.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            paths.append(path)
            print(f"  Equity plot: {path}")

        return paths

    def plot_risk_comparison(self) -> list:
        """
        For each symbol, compare MA Crossover equity curves under
        different risk configs on a single chart.
        """
        paths = []

        for sym, sym_data in self.results.items():
            strat_results = sym_data["strategies"]
            if "MA Crossover" not in strat_results:
                continue

            ma_results = strat_results["MA Crossover"]

            fig, ax = plt.subplots(figsize=(14, 6))

            colors = {"no_risk": "#2196F3", "conservative": "#4CAF50", "moderate": "#FF9800"}
            for risk_name, result in ma_results.items():
                eq = result["equity_curve"]
                normalised = eq / eq.iloc[0]
                label = f"{risk_name} (Sharpe={result['metrics']['sharpe']:.3f})"
                ax.plot(eq.index, normalised, label=label, linewidth=1.5,
                        color=colors.get(risk_name, "black"))

            ax.set_title(f"{sym} -- MA Crossover: Risk Config Comparison")
            ax.set_ylabel("Portfolio Value (normalised)")
            ax.set_xlabel("Date")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()

            path = os.path.join(self.results_dir, f"{sym}_risk_comparison.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            paths.append(path)
            print(f"  Risk comparison plot: {path}")

        return paths

    def plot_walkforward_degradation(self) -> list:
        """
        Bar chart: IS Sharpe vs OOS Sharpe per strategy for each symbol.
        Visualises overfitting / robustness.
        """
        paths = []

        for sym, sym_data in self.results.items():
            wf = sym_data.get("walkforward", {})
            if not wf:
                continue

            strats = list(wf.keys())
            is_sharpes  = [wf[s]["train"]["metrics"]["sharpe"] for s in strats]
            oos_sharpes = [wf[s]["test"]["metrics"]["sharpe"] for s in strats]

            fig, ax = plt.subplots(figsize=(max(12, len(strats) * 0.8), 6))

            x = range(len(strats))
            width = 0.35
            bars1 = ax.bar([i - width/2 for i in x], is_sharpes, width,
                          label="In-Sample", color="#2196F3", alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], oos_sharpes, width,
                          label="Out-of-Sample", color="#FF9800", alpha=0.8)

            ax.set_title(f"{sym} -- Walk-Forward: IS vs OOS Sharpe Ratio")
            ax.set_ylabel("Sharpe Ratio")
            ax.set_xticks(list(x))
            ax.set_xticklabels(strats, rotation=45, ha="right", fontsize=7)
            ax.legend(fontsize=9)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()

            path = os.path.join(self.results_dir, f"{sym}_walkforward.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            paths.append(path)
            print(f"  Walk-forward plot: {path}")

        return paths
