"""
================================================================================
MODULE 7: Walk-Forward Backtesting Engine
================================================================================
PURPOSE:
    Implement a rigorous walk-forward backtesting framework that avoids
    the most common pitfalls: look-ahead bias, survivorship bias, and
    overfitting to in-sample data.

THEORY:
    Walk-Forward Testing:
    - Train model on [0, T], rebalance at T, test on [T, T+h]
    - Expand training window, repeat
    - No future information ever leaks into past decisions

    Rebalancing:
    - Portfolio is rebalanced periodically (monthly by default)
    - Transaction costs are applied at each rebalance
    - Turnover is tracked to assess implementation feasibility

    Key Metrics Computed:
    - Cumulative returns and drawdowns
    - Rolling Sharpe ratio
    - Turnover and transaction cost drag
    - Alpha and beta vs. benchmark

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine with transaction cost modeling.

    Simulates portfolio rebalancing over time using expanding-window
    optimization, tracks all relevant performance and risk metrics.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 10,
        rebalance_frequency: int = 21,  # Monthly
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (T x N).
        benchmark_returns : pd.Series
            Benchmark return series.
        initial_capital : float
            Starting capital.
        transaction_cost_bps : float
            One-way transaction cost in basis points.
        rebalance_frequency : int
            Rebalancing period in trading days.
        """
        self.returns = returns
        self.benchmark = benchmark_returns
        self.initial_capital = initial_capital
        self.tc_rate = transaction_cost_bps / 10000
        self.rebal_freq = rebalance_frequency
        self.n_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()

    def run(
        self,
        strategy_fn: Callable,
        strategy_name: str = 'Strategy',
        min_training_days: int = 504,  # ~2 years
        **strategy_kwargs
    ) -> Dict:
        """
        Execute walk-forward backtest.

        At each rebalance date:
        1. Use all data up to date t to compute optimal weights
        2. Apply weights for the next rebalancing period
        3. Track returns, turnover, and transaction costs

        Parameters
        ----------
        strategy_fn : Callable
            Function that takes (returns_so_far) → weights array.
        strategy_name : str
            Name for the strategy.
        min_training_days : int
            Minimum training window before first trade.

        Returns
        -------
        dict
            Complete backtest results.
        """
        T = len(self.returns)
        dates = self.returns.index

        # Initialize tracking
        portfolio_returns = pd.Series(0.0, index=dates, name=strategy_name)
        weights_history = pd.DataFrame(
            0.0, index=dates, columns=self.tickers
        )
        turnover_history = pd.Series(0.0, index=dates, name='turnover')
        tc_history = pd.Series(0.0, index=dates, name='transaction_costs')

        current_weights = np.zeros(self.n_assets)
        rebalance_dates = []

        for t in range(min_training_days, T):
            # Daily portfolio return (using current weights)
            daily_ret = self.returns.iloc[t].values
            port_ret = current_weights @ daily_ret

            # Transaction costs (if rebalance day)
            tc = 0.0
            if (t - min_training_days) % self.rebal_freq == 0:
                # Compute new weights using data up to t
                train_returns = self.returns.iloc[:t]
                try:
                    new_weights = strategy_fn(train_returns, **strategy_kwargs)
                    new_weights = np.array(new_weights)
                    new_weights = np.clip(new_weights, 0, 1)
                    new_weights = new_weights / new_weights.sum() if new_weights.sum() > 0 else np.ones(self.n_assets) / self.n_assets
                except Exception:
                    new_weights = current_weights.copy()

                # Compute turnover
                turnover = np.sum(np.abs(new_weights - current_weights))
                tc = turnover * self.tc_rate

                current_weights = new_weights
                rebalance_dates.append(dates[t])
                turnover_history.iloc[t] = turnover
                tc_history.iloc[t] = tc

            # Net return after costs
            portfolio_returns.iloc[t] = port_ret - tc
            weights_history.iloc[t] = current_weights

            # Drift weights based on returns (between rebalances)
            if (t - min_training_days) % self.rebal_freq != 0:
                drifted = current_weights * (1 + daily_ret)
                current_weights = drifted / drifted.sum() if drifted.sum() > 0 else current_weights

        # Trim to active period
        active_mask = portfolio_returns.index >= dates[min_training_days]
        portfolio_returns = portfolio_returns[active_mask]
        benchmark_active = self.benchmark[active_mask]

        # Compute performance metrics
        metrics = self._compute_metrics(
            portfolio_returns, benchmark_active, strategy_name
        )
        metrics['rebalance_dates'] = rebalance_dates
        metrics['total_turnover'] = turnover_history.sum()
        metrics['total_transaction_costs'] = tc_history.sum()
        metrics['avg_turnover_per_rebalance'] = (
            turnover_history[turnover_history > 0].mean()
        )

        return {
            'name': strategy_name,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_active,
            'cumulative_returns': (1 + portfolio_returns).cumprod(),
            'benchmark_cumulative': (1 + benchmark_active).cumprod(),
            'weights_history': weights_history[active_mask],
            'metrics': metrics,
        }

    def _compute_metrics(
        self, port_rets: pd.Series, bench_rets: pd.Series, name: str
    ) -> Dict:
        """Compute comprehensive performance metrics."""
        ann = 252
        excess = port_rets - bench_rets

        # Basic metrics
        ann_ret = port_rets.mean() * ann
        ann_vol = port_rets.std() * np.sqrt(ann)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Benchmark comparison
        bench_ann_ret = bench_rets.mean() * ann
        bench_ann_vol = bench_rets.std() * np.sqrt(ann)

        # Alpha and Beta (CAPM regression)
        if bench_rets.std() > 0:
            beta = np.cov(port_rets, bench_rets)[0, 1] / bench_rets.var()
            alpha = ann_ret - beta * bench_ann_ret
        else:
            beta, alpha = 0, ann_ret

        # Information Ratio
        tracking_error = excess.std() * np.sqrt(ann)
        info_ratio = excess.mean() * ann / tracking_error if tracking_error > 0 else 0

        # Drawdown
        cum = (1 + port_rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()

        # Sortino Ratio (downside deviation)
        downside = port_rets[port_rets < 0]
        downside_dev = downside.std() * np.sqrt(ann) if len(downside) > 0 else 1e-10
        sortino = ann_ret / downside_dev

        # Calmar Ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

        # Win rate
        win_rate = (port_rets > 0).mean()

        return {
            'strategy': name,
            'ann_return': ann_ret,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': info_ratio,
            'tracking_error': tracking_error,
            'win_rate': win_rate,
            'bench_ann_return': bench_ann_ret,
            'bench_sharpe': bench_ann_ret / bench_ann_vol if bench_ann_vol > 0 else 0,
        }

    def run_multiple_strategies(
        self, strategy_fns: Dict[str, Callable],
        min_training_days: int = 504
    ) -> Dict[str, Dict]:
        """
        Run multiple strategies and return comparative results.

        Parameters
        ----------
        strategy_fns : Dict[str, Callable]
            {strategy_name: strategy_function}

        Returns
        -------
        Dict[str, Dict]
            Results for each strategy.
        """
        all_results = {}
        for name, fn in strategy_fns.items():
            print(f"  Running: {name}...")
            result = self.run(fn, strategy_name=name,
                            min_training_days=min_training_days)
            all_results[name] = result
        return all_results

    @staticmethod
    def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table of strategy metrics."""
        metrics_list = {}
        for name, result in results.items():
            m = result['metrics'].copy()
            # Remove non-scalar values
            m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            metrics_list[name] = m

        df = pd.DataFrame(metrics_list).T
        df.index.name = 'strategy'

        # Format percentages
        pct_cols = ['ann_return', 'ann_volatility', 'max_drawdown',
                    'alpha', 'tracking_error', 'win_rate', 'bench_ann_return']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col] * 100

        return df.round(4)


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from data_generator import MarketDataGenerator

    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(2520)
    benchmark = gen.generate_benchmark(returns)

    def equal_weight_strategy(rets):
        n = rets.shape[1]
        return np.ones(n) / n

    bt = WalkForwardBacktester(
        returns, benchmark,
        transaction_cost_bps=10,
        rebalance_frequency=21
    )

    result = bt.run(equal_weight_strategy, 'Equal Weight', min_training_days=504)

    print("=" * 70)
    print("BACKTEST RESULTS — EQUAL WEIGHT")
    print("=" * 70)
    for k, v in result['metrics'].items():
        if isinstance(v, (int, float)):
            print(f"  {k:30s}: {v:+.4f}")
