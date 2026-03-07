"""
================================================================================
MODULE 4: Portfolio Optimization Engine
================================================================================
PURPOSE:
    Implement 5 distinct portfolio optimization strategies, from classical
    Markowitz Mean-Variance Optimization to modern machine-learning-based
    Hierarchical Risk Parity. This demonstrates mastery of the full spectrum
    of portfolio construction methodologies used in industry.

STRATEGIES IMPLEMENTED:
    1. Mean-Variance Optimization (MVO) — Markowitz (1952)
    2. Black-Litterman Model — Black & Litterman (1992)
    3. Risk Parity — Qian (2005)
    4. Minimum CVaR Optimization — Rockafellar & Uryasev (2000)
    5. Hierarchical Risk Parity (HRP) — Lopez de Prado (2016)


================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, Tuple, Optional, List


class PortfolioOptimizer:
    """
    Multi-strategy portfolio optimization engine.

    Implements 5 optimization methodologies with a unified interface.
    All optimizers return a weight vector w ∈ R^N with sum(w) = 1.
    """

    def __init__(
        self, returns: pd.DataFrame,
        risk_free_rate: float = 0.04 / 252,
        allow_short: bool = False
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Historical asset returns (T x N).
        risk_free_rate : float
            Daily risk-free rate.
        allow_short : bool
            Whether to allow short positions.
        """
        self.returns = returns
        self.rf = risk_free_rate
        self.allow_short = allow_short
        self.n_assets = returns.shape[1]
        self.tickers = returns.columns.tolist()

        # Pre-compute statistics
        self.mu = returns.mean().values  # Expected returns
        self.cov = returns.cov().values  # Covariance matrix
        self.corr = returns.corr().values  # Correlation matrix

    # ══════════════════════════════════════════════════════════════════
    # 1. MEAN-VARIANCE OPTIMIZATION (MARKOWITZ)
    # ══════════════════════════════════════════════════════════════════

    def mean_variance_optimize(
        self, target: str = 'max_sharpe',
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """
        Classical Mean-Variance Optimization.

        Three objectives:
        - 'max_sharpe': Maximize Sharpe ratio (tangency portfolio)
        - 'min_variance': Minimize portfolio variance (global min var)
        - 'target_return': Minimize variance subject to target return

        The optimization problem (max Sharpe):
            max  (w'mu - rf) / sqrt(w'Sigma w)
            s.t. sum(w) = 1, w >= 0  (if long-only)
        """
        bounds = (
            (None, None) if self.allow_short else (0, 1)
            for _ in range(self.n_assets)
        )
        bounds = list(bounds)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if target == 'max_sharpe':
            def neg_sharpe(w):
                port_ret = w @ self.mu - self.rf
                port_vol = np.sqrt(w @ self.cov @ w)
                return -port_ret / port_vol if port_vol > 1e-10 else 0

            result = minimize(
                neg_sharpe, x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )

        elif target == 'min_variance':
            def portfolio_variance(w):
                return w @ self.cov @ w

            result = minimize(
                portfolio_variance, x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP', bounds=bounds, constraints=constraints
            )

        elif target == 'target_return':
            assert target_return is not None
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ self.mu - target_return
            })

            def portfolio_variance(w):
                return w @ self.cov @ w

            result = minimize(
                portfolio_variance, x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP', bounds=bounds, constraints=constraints
            )

        return result.x

    def compute_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Compute the efficient frontier — the set of optimal portfolios.

        For each target return between the min-var return and the max
        feasible return, finds the minimum-variance portfolio.

        Returns
        -------
        pd.DataFrame
            Columns: [Return, Volatility, Sharpe, Weights...]
        """
        # Find return range
        min_var_w = self.mean_variance_optimize('min_variance')
        min_ret = min_var_w @ self.mu
        max_ret = self.mu.max()

        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

        frontier = []
        for tr in target_returns:
            try:
                w = self.mean_variance_optimize('target_return', target_return=tr)
                port_ret = w @ self.mu * 252
                port_vol = np.sqrt(w @ self.cov @ w) * np.sqrt(252)
                sharpe = (port_ret - self.rf * 252) / port_vol if port_vol > 0 else 0

                row = {
                    'Return': port_ret, 'Volatility': port_vol, 'Sharpe': sharpe
                }
                for i, t in enumerate(self.tickers):
                    row[t] = w[i]
                frontier.append(row)
            except Exception:
                continue

        return pd.DataFrame(frontier)

    # ══════════════════════════════════════════════════════════════════
    # 2. BLACK-LITTERMAN MODEL
    # ══════════════════════════════════════════════════════════════════

    def black_litterman(
        self,
        market_caps: Optional[np.ndarray] = None,
        views: Optional[Dict] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5
    ) -> np.ndarray:
        """
        Black-Litterman model for combining equilibrium with active views.

        The BL model:
        1. Starts with CAPM equilibrium returns (implied by market caps)
        2. Blends in investor views with uncertainty
        3. Produces posterior expected returns
        4. Optimizes using posterior (much more stable than raw MVO)

        Posterior returns:
            mu_BL = [(tau*Sigma)^{-1} + P'Omega^{-1}P]^{-1}
                    * [(tau*Sigma)^{-1}*pi + P'Omega^{-1}*Q]

        Parameters
        ----------
        market_caps : np.ndarray, optional
            Market capitalizations (for equilibrium weights).
        views : dict, optional
            Investor views: {'absolute': [(asset, return, confidence)],
                            'relative': [(asset1, asset2, spread, confidence)]}
        tau : float
            Uncertainty scaling parameter (typically 0.01 to 0.1).
        risk_aversion : float
            Risk aversion parameter (lambda).
        """
        N = self.n_assets
        Sigma = self.cov * 252  # Annualize

        # Step 1: Equilibrium returns from market cap weights
        if market_caps is None:
            market_caps = np.random.dirichlet(np.ones(N) * 5)
        w_mkt = market_caps / market_caps.sum()
        pi = risk_aversion * Sigma @ w_mkt  # Implied equilibrium returns

        # Step 2: Construct view matrices
        if views is None:
            # Default views: tech outperforms, bonds underperform
            views = {
                'absolute': [
                    (0, 0.12, 0.7),   # Asset 0 returns 12% with 70% confidence
                    (11, 0.02, 0.8),   # Asset 11 returns 2% with 80% confidence
                ],
                'relative': [
                    (0, 5, 0.05, 0.6),  # Asset 0 outperforms Asset 5 by 5%
                ]
            }

        # Build P matrix (view portfolio) and Q vector (view returns)
        P_rows = []
        Q_list = []
        Omega_diags = []

        for asset_idx, view_ret, conf in views.get('absolute', []):
            p = np.zeros(N)
            p[asset_idx] = 1.0
            P_rows.append(p)
            Q_list.append(view_ret)
            Omega_diags.append((1 - conf) * 0.01)

        for a1, a2, spread, conf in views.get('relative', []):
            p = np.zeros(N)
            p[a1] = 1.0
            p[a2] = -1.0
            P_rows.append(p)
            Q_list.append(spread)
            Omega_diags.append((1 - conf) * 0.01)

        if len(P_rows) == 0:
            return w_mkt

        P = np.array(P_rows)
        Q = np.array(Q_list)
        Omega = np.diag(Omega_diags)

        # Step 3: Posterior returns (BL formula)
        tau_Sigma = tau * Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)

        M = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        mu_BL = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)

        # Step 4: Optimize using posterior returns
        self_backup_mu = self.mu.copy()
        self.mu = mu_BL / 252  # Convert to daily

        weights = self.mean_variance_optimize('max_sharpe')

        self.mu = self_backup_mu  # Restore
        return weights

    # ══════════════════════════════════════════════════════════════════
    # 3. RISK PARITY
    # ══════════════════════════════════════════════════════════════════

    def risk_parity(self) -> np.ndarray:
        """
        Risk Parity (Equal Risk Contribution) portfolio.

        Find w such that each asset contributes equally to portfolio risk:
            RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w) = sigma_p / N

        Solved by minimizing:
            sum_i (RC_i - sigma_p/N)^2

        Risk Parity is widely used at firms like Bridgewater (All Weather).
        """
        N = self.n_assets
        Sigma = self.cov

        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / port_vol
            target_rc = port_vol / N
            return np.sum((risk_contrib - target_rc) ** 2)

        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 1.0)] * N
        x0 = np.ones(N) / N

        result = minimize(
            risk_budget_objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        return result.x

    # ══════════════════════════════════════════════════════════════════
    # 4. MINIMUM CVaR OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════

    def minimum_cvar(self, alpha: float = 0.05) -> np.ndarray:
        """
        Minimize Conditional Value-at-Risk (Expected Shortfall).

        Uses the Rockafellar-Uryasev (2000) linear programming reformulation:
            min CVaR = min [ zeta + 1/(alpha*T) * sum max(0, -r_t'w - zeta) ]

        We use a smooth approximation since scipy doesn't have LP.

        CVaR minimization produces portfolios more robust to tail risk than
        variance minimization.
        """
        T = len(self.returns)
        N = self.n_assets
        R = self.returns.values

        def cvar_objective(params):
            w = params[:N]
            zeta = params[N]  # VaR threshold

            # Portfolio returns
            port_rets = R @ w
            losses = -port_rets - zeta
            excess_losses = np.maximum(losses, 0)

            cvar = zeta + (1 / (alpha * T)) * np.sum(excess_losses)
            return cvar

        # Initial guess
        x0 = np.append(np.ones(N) / N, 0.0)

        # Constraints: weights sum to 1
        constraints = [{
            'type': 'eq',
            'fun': lambda p: np.sum(p[:N]) - 1
        }]

        # Bounds: weights [0, 1], zeta unconstrained
        bounds = [(0, 1)] * N + [(None, None)]

        result = minimize(
            cvar_objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        return result.x[:N]

    # ══════════════════════════════════════════════════════════════════
    # 5. HIERARCHICAL RISK PARITY (HRP)
    # ══════════════════════════════════════════════════════════════════

    def hierarchical_risk_parity(self) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado, 2016).

        HRP uses machine learning (hierarchical clustering) to build a
        portfolio that doesn't require covariance matrix inversion.

        Three steps:
        1. Tree Clustering — cluster assets by correlation distance
        2. Quasi-Diagonalization — reorder assets along cluster tree
        3. Recursive Bisection — allocate risk top-down through the tree

        Advantages over MVO:
        - No matrix inversion → numerically stable
        - No expected return estimation needed
        - Respects hierarchical correlation structure
        - Better out-of-sample performance empirically
        """
        # Step 1: Distance matrix from correlations
        dist = np.sqrt(0.5 * (1 - self.corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist)
        link = linkage(condensed_dist, method='single')

        # Step 2: Quasi-diagonalization (reorder by cluster tree)
        sort_ix = leaves_list(link).astype(int)

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(
            self.cov, sort_ix
        )

        # Map back to original order
        w = np.zeros(self.n_assets)
        for i, ix in enumerate(sort_ix):
            w[ix] = weights[i]

        return w

    def _recursive_bisection(
        self, cov: np.ndarray, sort_ix: np.ndarray
    ) -> np.ndarray:
        """
        Recursive bisection step of HRP.

        At each level, splits the sorted assets into two clusters
        and allocates risk inversely proportional to cluster variance.
        """
        w = np.ones(len(sort_ix))

        cluster_items = [sort_ix]

        while len(cluster_items) > 0:
            # Split each cluster in half
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue

                half = len(cluster) // 2
                left = cluster[:half]
                right = cluster[half:]

                # Cluster variances (inverse-variance allocation)
                var_left = self._cluster_variance(cov, left)
                var_right = self._cluster_variance(cov, right)

                alpha = 1 - var_left / (var_left + var_right)

                # Scale weights
                for i in range(len(sort_ix)):
                    if sort_ix[i] in left:
                        w[i] *= alpha
                    elif sort_ix[i] in right:
                        w[i] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            cluster_items = new_clusters

        return w / w.sum()

    @staticmethod
    def _cluster_variance(cov: np.ndarray, indices: np.ndarray) -> float:
        """Compute the variance of an equal-weight cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        n = len(indices)
        w = np.ones(n) / n
        return w @ sub_cov @ w

    # ══════════════════════════════════════════════════════════════════
    # COMPARISON UTILITIES
    # ══════════════════════════════════════════════════════════════════

    def optimize_all_strategies(self) -> Dict[str, np.ndarray]:
        """Run all optimization strategies and return weights."""
        strategies = {
            'Equal Weight': np.ones(self.n_assets) / self.n_assets,
            'Max Sharpe (MVO)': self.mean_variance_optimize('max_sharpe'),
            'Min Variance': self.mean_variance_optimize('min_variance'),
            'Black-Litterman': self.black_litterman(),
            'Risk Parity': self.risk_parity(),
            'Min CVaR': self.minimum_cvar(),
            'HRP': self.hierarchical_risk_parity(),
        }
        return strategies

    def weights_table(self, strategies: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Format strategy weights into a comparison table."""
        df = pd.DataFrame(strategies, index=self.tickers)
        df = df.round(4)
        return df


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from data_generator import MarketDataGenerator

    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(2520)

    opt = PortfolioOptimizer(returns, risk_free_rate=0.04/252)
    strategies = opt.optimize_all_strategies()

    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 70)
    print("\nWeight Allocations:")
    print(opt.weights_table(strategies).to_string())
