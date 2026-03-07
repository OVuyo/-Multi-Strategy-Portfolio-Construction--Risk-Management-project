"""
================================================================================
MODULE 8: Performance Analytics & Visualization
================================================================================
PURPOSE:
    Generate quality performance reports and visualizations.

OUTPUT:
    - Cumulative return charts
    - Drawdown analysis
    - Rolling metrics (Sharpe, volatility)
    - Weight allocation over time
    - Factor exposure charts
    - Risk decomposition
    - Strategy comparison dashboard


================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
          '#3B1F2B', '#44BBA4', '#E94F37', '#393E41',
          '#5C6B73', '#9DB4C0', '#DAA520', '#2F4F4F']


class PerformanceAnalytics:
    """
    Generates comprehensive performance analytics and visualizations.
    """

    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 1: Strategy Comparison Dashboard
    # ══════════════════════════════════════════════════════════════════

    def plot_strategy_comparison(
        self, results: Dict[str, Dict], save_path: str = None
    ) -> str:
        """
        Create a 2x2 dashboard comparing all strategies.

        Panels:
        1. Cumulative Returns (log scale)
        2. Drawdown Analysis
        3. Rolling 1-Year Sharpe Ratio
        4. Performance Summary Bar Chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Strategy Portfolio Comparison Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        # Panel 1: Cumulative Returns
        ax = axes[0, 0]
        for i, (name, res) in enumerate(results.items()):
            cum = res['cumulative_returns']
            ax.plot(cum.index, cum.values, label=name,
                   color=COLORS[i % len(COLORS)], linewidth=1.5)
        # Benchmark
        if 'benchmark_cumulative' in list(results.values())[0]:
            bench = list(results.values())[0]['benchmark_cumulative']
            ax.plot(bench.index, bench.values, label='Benchmark',
                   color='black', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.set_ylabel('Growth of $1')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_yscale('log')

        # Panel 2: Drawdowns
        ax = axes[0, 1]
        for i, (name, res) in enumerate(results.items()):
            cum = res['cumulative_returns']
            dd = (cum - cum.cummax()) / cum.cummax()
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3,
                          color=COLORS[i % len(COLORS)], label=name)
        ax.set_title('Drawdown Analysis', fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.legend(fontsize=8)
        ax.set_ylim([-0.6, 0.05])

        # Panel 3: Rolling Sharpe
        ax = axes[1, 0]
        for i, (name, res) in enumerate(results.items()):
            rets = res['portfolio_returns']
            rolling_sharpe = (
                rets.rolling(252).mean() / rets.rolling(252).std()
            ) * np.sqrt(252)
            ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                   label=name, color=COLORS[i % len(COLORS)], linewidth=1.2)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.axhline(y=1, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(fontsize=8)

        # Panel 4: Performance Bars
        ax = axes[1, 1]
        names = list(results.keys())
        sharpes = [results[n]['metrics']['sharpe_ratio'] for n in names]
        colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
        bars = ax.barh(names, sharpes, color=colors, alpha=0.8)
        ax.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax.set_xlabel('Sharpe Ratio')
        for bar, val in zip(bars, sharpes):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=10)

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/01_strategy_comparison.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 2: Efficient Frontier
    # ══════════════════════════════════════════════════════════════════

    def plot_efficient_frontier(
        self, frontier: pd.DataFrame,
        strategy_points: Dict[str, Dict] = None,
        save_path: str = None
    ) -> str:
        """Plot the efficient frontier with strategy portfolio positions."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Frontier curve
        ax.plot(frontier['Volatility'] * 100, frontier['Return'] * 100,
               'b-', linewidth=2, label='Efficient Frontier')

        # Color by Sharpe ratio
        sc = ax.scatter(
            frontier['Volatility'] * 100, frontier['Return'] * 100,
            c=frontier['Sharpe'], cmap='RdYlGn', s=20, zorder=5
        )
        plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

        # Plot strategy positions
        if strategy_points:
            for i, (name, point) in enumerate(strategy_points.items()):
                ax.scatter(
                    point['vol'] * 100, point['ret'] * 100,
                    marker='*', s=300, color=COLORS[i % len(COLORS)],
                    edgecolors='black', linewidth=1, zorder=10, label=name
                )

        ax.set_title('Efficient Frontier & Strategy Positions',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/02_efficient_frontier.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 3: Risk Decomposition
    # ══════════════════════════════════════════════════════════════════

    def plot_risk_decomposition(
        self, risk_data: Dict, save_path: str = None
    ) -> str:
        """Pie chart of risk decomposition and bar chart of factor contributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie: Systematic vs Idiosyncratic
        sizes = [risk_data['systematic_pct'], risk_data['idiosyncratic_pct']]
        labels = ['Systematic\nRisk', 'Idiosyncratic\nRisk']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
               colors=['#2E86AB', '#F18F01'], startangle=90,
               textprops={'fontsize': 12})
        ax1.set_title('Risk Decomposition', fontweight='bold', fontsize=13)

        # Bar: Factor contributions
        fc = risk_data['factor_contributions']
        factors = list(fc.keys())
        pcts = [fc[f]['pct_of_total'] for f in factors]
        ax2.barh(factors, pcts, color=COLORS[:len(factors)])
        ax2.set_title('Factor Risk Contributions', fontweight='bold', fontsize=13)
        ax2.set_xlabel('% of Total Risk')

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/03_risk_decomposition.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 4: Weight Allocations
    # ══════════════════════════════════════════════════════════════════

    def plot_weight_comparison(
        self, strategies: Dict[str, np.ndarray],
        tickers: List[str], save_path: str = None
    ) -> str:
        """Stacked bar chart comparing strategy allocations."""
        fig, ax = plt.subplots(figsize=(14, 8))

        n_strategies = len(strategies)
        x = np.arange(len(tickers))
        width = 0.8 / n_strategies

        for i, (name, weights) in enumerate(strategies.items()):
            ax.bar(x + i * width, weights * 100, width,
                  label=name, color=COLORS[i % len(COLORS)], alpha=0.8)

        ax.set_xlabel('Asset', fontsize=12)
        ax.set_ylabel('Weight (%)', fontsize=12)
        ax.set_title('Portfolio Weight Allocations by Strategy',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * n_strategies / 2)
        ax.set_xticklabels(tickers, rotation=45)
        ax.legend(fontsize=9, loc='upper right')

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/04_weight_allocations.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 5: Regime Analysis
    # ══════════════════════════════════════════════════════════════════

    def plot_regime_analysis(
        self, returns: pd.Series, regimes: pd.Series,
        regime_stats: pd.DataFrame, save_path: str = None
    ) -> str:
        """Plot return series colored by detected regime."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       height_ratios=[2, 1])

        # Cumulative returns colored by regime
        cum = (1 + returns.loc[regimes.index]).cumprod()
        regime_colors = {0: '#44BBA4', 1: '#F18F01', 2: '#C73E1D'}
        regime_names = {0: 'Bull', 1: 'Bear', 2: 'Crisis'}

        for regime_id, color in regime_colors.items():
            mask = regimes == regime_id
            ax1.fill_between(
                cum.index, cum.min(), cum.max(),
                where=mask, alpha=0.15, color=color,
                label=regime_names[regime_id]
            )
        ax1.plot(cum.index, cum.values, 'k-', linewidth=1)
        ax1.set_title('Cumulative Returns with Detected Market Regimes',
                      fontweight='bold', fontsize=13)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(fontsize=10, loc='upper left')

        # Regime statistics
        if not regime_stats.empty:
            stats_display = regime_stats[
                ['ann_return', 'ann_volatility', 'sharpe']
            ].copy()
            stats_display.columns = ['Ann. Return', 'Ann. Vol', 'Sharpe']
            stats_display = stats_display.T
            colors_list = [regime_colors[i] for i in range(len(stats_display.columns))]

            stats_display.plot(kind='bar', ax=ax2, color=colors_list, alpha=0.8)
            ax2.set_title('Regime-Conditional Statistics', fontweight='bold')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
            ax2.legend(fontsize=9)

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/05_regime_analysis.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 6: Stress Test Results
    # ══════════════════════════════════════════════════════════════════

    def plot_stress_test(
        self, stress_results: pd.DataFrame, save_path: str = None
    ) -> str:
        """Horizontal bar chart of stress test scenario losses."""
        fig, ax = plt.subplots(figsize=(12, 6))

        scenarios = stress_results.index
        losses = stress_results['Portfolio Return (%)']

        colors = ['#C73E1D' if v < 0 else '#44BBA4' for v in losses]
        ax.barh(scenarios, losses, color=colors, alpha=0.85)

        ax.set_title('Stress Test: Portfolio Impact by Scenario',
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Portfolio Return (%)')
        ax.axvline(x=0, color='black', linewidth=0.5)

        for i, (v, s) in enumerate(zip(losses, scenarios)):
            ax.text(v - 1 if v < 0 else v + 0.5, i, f'{v:.1f}%',
                   va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/06_stress_test.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 7: Correlation Heatmap
    # ══════════════════════════════════════════════════════════════════

    def plot_correlation_matrix(
        self, returns: pd.DataFrame, save_path: str = None
    ) -> str:
        """Asset correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        corr = returns.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5
        )
        ax.set_title('Asset Return Correlations', fontweight='bold', fontsize=14)

        plt.tight_layout()
        path = save_path or f'{self.output_dir}/07_correlation_matrix.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
