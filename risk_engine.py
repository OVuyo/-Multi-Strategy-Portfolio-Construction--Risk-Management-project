"""
================================================================================
MODULE 3: Risk Analytics Engine
================================================================================
PURPOSE:
    Comprehensive risk measurement framework implementing parametric,
    historical, and Monte Carlo Value-at-Risk (VaR), Conditional VaR (CVaR),
    stress testing, and drawdown analytics.

THEORY:
    Value-at-Risk (VaR):
        The maximum loss over a holding period at a given confidence level.
        VaR_alpha = -F^{-1}(alpha) where F is the return distribution CDF.

    Conditional VaR (CVaR / Expected Shortfall):
        The expected loss given that loss exceeds VaR.
        CVaR_alpha = E[L | L > VaR_alpha]
        CVaR is a coherent risk measure (satisfies subadditivity).

    Three approaches:
    1. Parametric (Gaussian) — fast but assumes normality
    2. Historical — non-parametric, uses actual return distribution
    3. Monte Carlo — flexible, can incorporate complex dynamics

INTERVIEW RELEVANCE:
    - VaR/CVaR is daily work at every risk desk and hedge fund
    - Understanding coherent risk measures (CVaR > VaR) is critical
    - Stress testing is a regulatory requirement (Basel III/IV)
    - Drawdown analysis matters for fund evaluation
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List


class RiskEngine:
    """
    Institutional-grade risk analytics engine.

    Computes multiple risk metrics across different methodologies,
    performs stress tests, and analyzes drawdown characteristics.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Parameters
        ----------
        confidence_level : float
            Confidence level for VaR/CVaR calculations (default: 95%).
        """
        self.confidence = confidence_level
        self.alpha = 1 - confidence_level  # tail probability

    # ══════════════════════════════════════════════════════════════════
    # VALUE-AT-RISK METHODS
    # ══════════════════════════════════════════════════════════════════

    def parametric_var(
        self, returns: pd.Series, holding_period: int = 1
    ) -> float:
        """
        Parametric (Gaussian) VaR.

        VaR = -(mu + z_alpha * sigma) * sqrt(T)

        Assumes returns are normally distributed — a simplification that
        underestimates tail risk due to fat tails in real returns.
        """
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(self.alpha)
        return -(mu + z * sigma) * np.sqrt(holding_period)

    def historical_var(self, returns: pd.Series) -> float:
        """
        Historical (non-parametric) VaR.

        Simply takes the alpha-quantile of the empirical return distribution.
        No distributional assumptions required.
        """
        return -np.percentile(returns, self.alpha * 100)

    def cornish_fisher_var(self, returns: pd.Series) -> float:
        """
        Cornish-Fisher VaR — adjusts for skewness and kurtosis.

        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*(K-3)/24 - (2*z^3 - 5*z)*S^2/36

        where S = skewness, K = excess kurtosis.

        This is superior to Gaussian VaR for fat-tailed distributions.
        """
        z = stats.norm.ppf(self.alpha)
        s = returns.skew()
        k = returns.kurtosis()  # excess kurtosis

        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
        mu = returns.mean()
        sigma = returns.std()
        return -(mu + z_cf * sigma)

    def monte_carlo_var(
        self, returns: pd.Series, n_simulations: int = 10000,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Monte Carlo VaR using fitted Student-t distribution.

        Fits a Student-t to capture fat tails, then simulates.

        Returns
        -------
        var : float
            Monte Carlo VaR.
        cvar : float
            Monte Carlo CVaR (Expected Shortfall).
        """
        rng = np.random.RandomState(seed)

        # Fit Student-t distribution to returns
        df, loc, scale = stats.t.fit(returns)

        # Simulate
        simulated = stats.t.rvs(df, loc=loc, scale=scale,
                                size=n_simulations, random_state=rng)

        var = -np.percentile(simulated, self.alpha * 100)
        # CVaR: average of losses beyond VaR
        losses = -simulated
        cvar = np.mean(losses[losses >= var])

        return var, cvar

    # ══════════════════════════════════════════════════════════════════
    # CONDITIONAL VALUE-AT-RISK (EXPECTED SHORTFALL)
    # ══════════════════════════════════════════════════════════════════

    def historical_cvar(self, returns: pd.Series) -> float:
        """
        Historical CVaR (Expected Shortfall).

        CVaR = E[Loss | Loss > VaR]

        CVaR is a coherent risk measure satisfying:
        1. Monotonicity
        2. Translation invariance
        3. Positive homogeneity
        4. Subadditivity (VaR fails this!)
        """
        var = self.historical_var(returns)
        losses = -returns
        return float(losses[losses >= var].mean())

    def parametric_cvar(self, returns: pd.Series) -> float:
        """
        Parametric (Gaussian) CVaR.

        CVaR = mu - sigma * phi(z_alpha) / alpha

        where phi is the standard normal PDF.
        """
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(self.alpha)
        return -(mu - sigma * stats.norm.pdf(z) / self.alpha)

    # ══════════════════════════════════════════════════════════════════
    # PORTFOLIO RISK
    # ══════════════════════════════════════════════════════════════════

    def portfolio_risk_metrics(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive risk metrics for a portfolio.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (T x N).
        weights : np.ndarray
            Portfolio weights (N,).

        Returns
        -------
        dict
            Complete risk profile including VaR, CVaR, drawdowns, etc.
        """
        port_returns = (returns * weights).sum(axis=1)

        # Core risk metrics
        metrics = {
            'annualized_return': port_returns.mean() * 252,
            'annualized_volatility': port_returns.std() * np.sqrt(252),
            'sharpe_ratio': (port_returns.mean() / port_returns.std()
                          * np.sqrt(252)) if port_returns.std() > 0 else 0,
            'skewness': port_returns.skew(),
            'excess_kurtosis': port_returns.kurtosis(),

            # VaR (daily, 95%)
            'parametric_var': self.parametric_var(port_returns),
            'historical_var': self.historical_var(port_returns),
            'cornish_fisher_var': self.cornish_fisher_var(port_returns),

            # CVaR (daily, 95%)
            'parametric_cvar': self.parametric_cvar(port_returns),
            'historical_cvar': self.historical_cvar(port_returns),

            # Drawdown analysis
            **self._drawdown_analysis(port_returns),

            # Tail risk
            'tail_ratio': self._tail_ratio(port_returns),
            'calmar_ratio': self._calmar_ratio(port_returns),
        }

        # Monte Carlo VaR
        mc_var, mc_cvar = self.monte_carlo_var(port_returns)
        metrics['monte_carlo_var'] = mc_var
        metrics['monte_carlo_cvar'] = mc_cvar

        return metrics

    def _drawdown_analysis(self, returns: pd.Series) -> Dict:
        """Compute drawdown statistics."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        max_dd = drawdown.min()
        dd_duration = self._max_drawdown_duration(drawdown)

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'max_drawdown_duration_days': dd_duration,
            'current_drawdown': drawdown.iloc[-1]
        }

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Find the longest drawdown period in days."""
        is_dd = drawdown < 0
        if not is_dd.any():
            return 0
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        if len(dd_groups) == 0:
            return 0
        return dd_groups.value_counts().max()

    @staticmethod
    def _tail_ratio(returns: pd.Series) -> float:
        """Ratio of right tail (95th percentile) to left tail (5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else np.inf

    @staticmethod
    def _calmar_ratio(returns: pd.Series) -> float:
        """Annualized return / max drawdown."""
        cum = (1 + returns).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        ann_ret = returns.mean() * 252
        return ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    # ══════════════════════════════════════════════════════════════════
    # STRESS TESTING
    # ══════════════════════════════════════════════════════════════════

    def stress_test(
        self, returns: pd.DataFrame, weights: np.ndarray,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Run stress test scenarios on the portfolio.

        Predefined scenarios model historical crises with
        correlated asset shocks.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset return history for calibration.
        weights : np.ndarray
            Portfolio weights.
        scenarios : dict, optional
            Custom scenarios {name: {ticker: shock}}.

        Returns
        -------
        pd.DataFrame
            Stress test results for each scenario.
        """
        if scenarios is None:
            scenarios = self._default_stress_scenarios(returns.columns.tolist())

        results = []
        for name, shocks in scenarios.items():
            # Apply shocks to each asset
            portfolio_shock = sum(
                weights[i] * shocks.get(ticker, 0)
                for i, ticker in enumerate(returns.columns)
            )

            results.append({
                'Scenario': name,
                'Portfolio Return (%)': portfolio_shock * 100,
                'Portfolio Loss ($1M)': -portfolio_shock * 1_000_000,
            })

        return pd.DataFrame(results).set_index('Scenario')

    @staticmethod
    def _default_stress_scenarios(tickers: List[str]) -> Dict:
        """
        Predefined stress scenarios modeled on historical crises.

        These are calibrated to approximate the magnitude and
        cross-asset correlations observed during actual events.
        """
        # Map tickers to default shocks by broad category
        tech = {'AAPL', 'MSFT', 'AMZN'}
        fin = {'JPM', 'GS'}
        defensive = {'JNJ', 'PG', 'NEE'}
        energy = {'XOM'}
        realestate = {'SPG'}
        gold = {'GLD'}
        bonds = {'TLT'}

        def make_scenario(tech_s, fin_s, def_s, ene_s, re_s, gold_s, bond_s):
            shocks = {}
            for t in tickers:
                if t in tech: shocks[t] = tech_s
                elif t in fin: shocks[t] = fin_s
                elif t in defensive: shocks[t] = def_s
                elif t in energy: shocks[t] = ene_s
                elif t in realestate: shocks[t] = re_s
                elif t in gold: shocks[t] = gold_s
                elif t in bonds: shocks[t] = bond_s
                else: shocks[t] = (tech_s + fin_s + def_s) / 3
            return shocks

        return {
            '2008 GFC':           make_scenario(-0.40, -0.55, -0.15, -0.35, -0.45, 0.05, 0.20),
            '2020 COVID Crash':   make_scenario(-0.30, -0.35, -0.12, -0.50, -0.40, 0.03, 0.08),
            'Tech Bubble Burst':  make_scenario(-0.50, -0.20, -0.05, -0.10, -0.15, 0.10, 0.15),
            'Rising Rates Shock': make_scenario(-0.15, -0.10, -0.08, 0.05, -0.20, -0.05, -0.15),
            'Stagflation':        make_scenario(-0.20, -0.25, -0.10, 0.15, -0.15, 0.20, -0.10),
            'Flash Crash (-5%)':  make_scenario(-0.05, -0.06, -0.03, -0.04, -0.05, 0.01, 0.02),
        }

    # ══════════════════════════════════════════════════════════════════
    # MARGINAL RISK CONTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════

    def marginal_risk_contribution(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute each asset's marginal contribution to portfolio risk.

        Euler decomposition:
            MCTR_i = (Sigma @ w)_i / sigma_p

        Risk contribution:
            RC_i = w_i * MCTR_i

        Sum of RC_i = sigma_p (Euler's theorem for homogeneous functions).
        """
        cov = returns.cov().values * 252  # Annualize
        port_vol = np.sqrt(weights @ cov @ weights)

        # Marginal contribution to risk
        mctr = (cov @ weights) / port_vol

        # Component risk contribution
        rc = weights * mctr

        # Percentage contribution
        pct_rc = rc / port_vol * 100

        return pd.DataFrame({
            'Weight': weights,
            'MCTR': mctr,
            'Risk Contribution': rc,
            '% of Total Risk': pct_rc
        }, index=returns.columns)


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from data_generator import MarketDataGenerator

    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(2520)

    engine = RiskEngine(confidence_level=0.95)

    # Equal-weight portfolio
    n = returns.shape[1]
    w = np.ones(n) / n

    metrics = engine.portfolio_risk_metrics(returns, w)

    print("=" * 70)
    print("RISK ANALYTICS — EQUAL-WEIGHT PORTFOLIO")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:+.6f}")

    print("\nStress Test Results:")
    print(engine.stress_test(returns, w).to_string())

    print("\nMarginal Risk Contributions:")
    print(engine.marginal_risk_contribution(returns, w).to_string())
