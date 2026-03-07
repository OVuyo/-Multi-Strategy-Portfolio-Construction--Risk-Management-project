"""
================================================================================
MAIN PIPELINE:  Multi-Strategy Portfolio Engine
================================================================================

This is the master orchestrator that connects all modules into a cohesive
end-to-end quantitative research pipeline.

Pipeline Steps:
    1. Data Generation     → Synthetic multi-asset returns with factor structure
    2. Factor Analysis     → Fama-French-style factor model & decomposition
    3. Regime Detection    → Hidden state identification (Bull/Bear/Crisis)
    4. ML Signal Generation → Feature engineering & return prediction
    5. Portfolio Optimization → 7 strategies (MVO, BL, RP, CVaR, HRP, etc.)
    6. Walk-Forward Backtest → Rigorous out-of-sample evaluation
    7. Risk Analytics      → VaR, CVaR, stress testing, drawdowns
    8. Performance Report  → Institutional-quality visualizations

Run:
    python main.py
================================================================================
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import MarketDataGenerator
from factor_model import FactorModel
from risk_engine import RiskEngine
from optimization_engine import PortfolioOptimizer
from regime_detection import RegimeDetector
from ml_signals import MLSignalGenerator
from backtester import WalkForwardBacktester
from performance_analytics import PerformanceAnalytics


def separator(title):
    print(f"\n{'='*75}")
    print(f"  STEP: {title}")
    print(f"{'='*75}")


def main():
    """Execute the full quantitative research pipeline."""

    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   MULTI-STRATEGY PORTFOLIO ENGINE                               ║
    ║   ─────────────────────────────────────────────────────────     ║
    ║   A Complete Quantitative Research Pipeline                     ║
    ║   Modules: Data | Factors | Regimes | ML | Optimization |       ║
    ║            Backtesting | Risk | Analytics                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    analytics = PerformanceAnalytics(output_dir)

    # ──────────────────────────────────────────────────────────────
    # STEP 1: DATA GENERATION
    # ──────────────────────────────────────────────────────────────
    separator("1/8 — DATA GENERATION")
    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(n_days=2520)
    benchmark = gen.generate_benchmark(returns)

    print(f"  Date Range  : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Assets      : {returns.shape[1]}")
    print(f"  Observations: {returns.shape[0]}")
    print(f"\n  Asset Summary Statistics:")
    summary = gen.summary(returns)
    print(summary.to_string())

    # ──────────────────────────────────────────────────────────────
    # STEP 2: FACTOR MODEL
    # ──────────────────────────────────────────────────────────────
    separator("2/8 — MULTI-FACTOR MODEL")
    fm = FactorModel()
    fm.fit(returns, factors)

    print("  Factor Loadings (Betas):")
    loadings = fm.get_factor_loadings_table()
    print(loadings[['Alpha', 'MKT', 'HML', 'MOM', 'R_squared']].round(4).to_string())

    print("\n  Information Ratios:")
    ir = fm.compute_information_ratio()
    print(ir.round(4).to_string())

    # ──────────────────────────────────────────────────────────────
    # STEP 3: REGIME DETECTION
    # ──────────────────────────────────────────────────────────────
    separator("3/8 — MARKET REGIME DETECTION")
    market_returns = returns.mean(axis=1)
    detector = RegimeDetector(n_regimes=3, lookback=63)
    detector.fit(market_returns)

    print("  Regime Summary:")
    regime_summary = detector.get_regime_summary()
    print(regime_summary.round(4).to_string())

    current = detector.predict_current_regime()
    print(f"\n  Current Regime: {current['regime']}")
    print(f"  Probabilities : {current['probabilities']}")

    # ──────────────────────────────────────────────────────────────
    # STEP 4: ML SIGNAL GENERATION
    # ──────────────────────────────────────────────────────────────
    separator("4/8 — ML SIGNAL GENERATION")
    ml = MLSignalGenerator(prediction_horizon=21)
    ml_results = ml.fit(returns)

    print("  Model Performance (Information Coefficient):")
    print(ml.get_model_metrics().round(4).to_string())

    print("\n  Top 10 Predictive Features:")
    print(ml.get_top_features(10).round(4).to_string())

    # ──────────────────────────────────────────────────────────────
    # STEP 5: PORTFOLIO OPTIMIZATION
    # ──────────────────────────────────────────────────────────────
    separator("5/8 — MULTI-STRATEGY PORTFOLIO OPTIMIZATION")
    opt = PortfolioOptimizer(returns, risk_free_rate=0.04/252)
    strategies = opt.optimize_all_strategies()

    print("  Weight Allocations by Strategy:")
    weights_table = opt.weights_table(strategies)
    print(weights_table.to_string())

    # Efficient Frontier
    print("\n  Computing Efficient Frontier...")
    frontier = opt.compute_efficient_frontier(n_points=40)
    print(f"  Frontier computed with {len(frontier)} portfolios.")

    # Strategy risk-return points for plotting
    strategy_points = {}
    for name, w in strategies.items():
        ret = w @ returns.mean().values * 252
        vol = np.sqrt(w @ returns.cov().values @ w) * np.sqrt(252)
        strategy_points[name] = {'ret': ret, 'vol': vol}

    # ──────────────────────────────────────────────────────────────
    # STEP 6: WALK-FORWARD BACKTESTING
    # ──────────────────────────────────────────────────────────────
    separator("6/8 — WALK-FORWARD BACKTESTING")

    # Define strategy functions for backtesting
    def equal_weight_fn(rets):
        return np.ones(rets.shape[1]) / rets.shape[1]

    def max_sharpe_fn(rets):
        o = PortfolioOptimizer(rets, risk_free_rate=0.04/252)
        return o.mean_variance_optimize('max_sharpe')

    def min_var_fn(rets):
        o = PortfolioOptimizer(rets, risk_free_rate=0.04/252)
        return o.mean_variance_optimize('min_variance')

    def risk_parity_fn(rets):
        o = PortfolioOptimizer(rets, risk_free_rate=0.04/252)
        return o.risk_parity()

    def hrp_fn(rets):
        o = PortfolioOptimizer(rets, risk_free_rate=0.04/252)
        return o.hierarchical_risk_parity()

    def min_cvar_fn(rets):
        o = PortfolioOptimizer(rets, risk_free_rate=0.04/252)
        return o.minimum_cvar()

    bt = WalkForwardBacktester(
        returns, benchmark,
        initial_capital=1_000_000,
        transaction_cost_bps=10,
        rebalance_frequency=21
    )

    strategy_fns = {
        'Equal Weight': equal_weight_fn,
        'Max Sharpe': max_sharpe_fn,
        'Min Variance': min_var_fn,
        'Risk Parity': risk_parity_fn,
        'HRP': hrp_fn,
        'Min CVaR': min_cvar_fn,
    }

    bt_results = bt.run_multiple_strategies(strategy_fns, min_training_days=504)

    print("\n  Strategy Comparison:")
    comparison = bt.compare_strategies(bt_results)
    print(comparison.to_string())

    # ──────────────────────────────────────────────────────────────
    # STEP 7: RISK ANALYTICS
    # ──────────────────────────────────────────────────────────────
    separator("7/8 — RISK ANALYTICS")
    risk = RiskEngine(confidence_level=0.95)

    # Analyze best strategy by Sharpe
    best_strategy = comparison['sharpe_ratio'].idxmax()
    best_weights = strategies.get(best_strategy, strategies['Risk Parity'])
    print(f"  Analyzing risk for best strategy: {best_strategy}")

    risk_metrics = risk.portfolio_risk_metrics(returns, best_weights)
    print("\n  Portfolio Risk Metrics:")
    for k, v in risk_metrics.items():
        if isinstance(v, float):
            print(f"    {k:35s}: {v:+.6f}")

    # Stress Test
    print("\n  Stress Test Results:")
    stress = risk.stress_test(returns, best_weights)
    print(stress.to_string())

    # Factor Risk Decomposition
    factor_cov = factors[['MKT', 'HML', 'MOM']].cov().values * 252
    risk_decomp = fm.compute_factor_risk_decomposition(best_weights, factor_cov)
    print(f"\n  Risk Decomposition:")
    print(f"    Systematic Risk : {risk_decomp['systematic_pct']:.1f}%")
    print(f"    Idiosyncratic   : {risk_decomp['idiosyncratic_pct']:.1f}%")
    print(f"    Total Ann. Vol  : {risk_decomp['total_volatility']:.4f}")

    # Marginal Risk Contributions
    print("\n  Marginal Risk Contributions:")
    mrc = risk.marginal_risk_contribution(returns, best_weights)
    print(mrc.round(4).to_string())

    # ──────────────────────────────────────────────────────────────
    # STEP 8: GENERATE VISUALIZATIONS
    # ──────────────────────────────────────────────────────────────
    separator("8/8 — GENERATING VISUALIZATIONS")

    paths = []

    # 1. Strategy Comparison Dashboard
    print("  [1/7] Strategy Comparison Dashboard...")
    p = analytics.plot_strategy_comparison(bt_results)
    paths.append(p)

    # 2. Efficient Frontier
    print("  [2/7] Efficient Frontier...")
    p = analytics.plot_efficient_frontier(frontier, strategy_points)
    paths.append(p)

    # 3. Risk Decomposition
    print("  [3/7] Risk Decomposition...")
    p = analytics.plot_risk_decomposition(risk_decomp)
    paths.append(p)

    # 4. Weight Allocations
    print("  [4/7] Weight Allocations...")
    p = analytics.plot_weight_comparison(strategies, returns.columns.tolist())
    paths.append(p)

    # 5. Regime Analysis
    print("  [5/7] Regime Analysis...")
    p = analytics.plot_regime_analysis(
        market_returns, detector.labels,
        regime_summary
    )
    paths.append(p)

    # 6. Stress Test
    print("  [6/7] Stress Test...")
    p = analytics.plot_stress_test(stress)
    paths.append(p)

    # 7. Correlation Matrix
    print("  [7/7] Correlation Matrix...")
    p = analytics.plot_correlation_matrix(returns)
    paths.append(p)

    # ──────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*75}")
    print(f"\n  Generated {len(paths)} visualization files in: {output_dir}/")
    for p in paths:
        print(f"    → {os.path.basename(p)}")

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  KEY FINDINGS                                               │
  │  ─────────────────────────────────────────────────────────  │
  │  Best Strategy (Sharpe): {best_strategy:38s}│
  │  Sharpe Ratio          : {comparison.loc[best_strategy, 'sharpe_ratio']:+38.4f}│
  │  Ann. Return           : {comparison.loc[best_strategy, 'ann_return']:+37.2f}%│
  │  Max Drawdown          : {comparison.loc[best_strategy, 'max_drawdown']:+37.2f}%│
  │  Current Regime        : {current['regime']:38s}│
  └─────────────────────────────────────────────────────────────┘
    """)

    # Save summary data
    comparison.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'))
    weights_table.to_csv(os.path.join(output_dir, 'strategy_weights.csv'))
    loadings.to_csv(os.path.join(output_dir, 'factor_loadings.csv'))

    return bt_results, comparison


if __name__ == '__main__':
    results, comparison = main()
