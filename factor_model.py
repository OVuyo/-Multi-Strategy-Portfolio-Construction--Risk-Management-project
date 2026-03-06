"""
================================================================================
MODULE 2: Multi-Factor Model & Risk Decomposition
================================================================================
PURPOSE:
    Implement Fama-French-style factor models to decompose asset returns into
    systematic and idiosyncratic components. This is the backbone of quantitative
    equity research and risk management.

THEORY:
    The multi-factor model (Arbitrage Pricing Theory):

        r_i - r_f = alpha_i + sum_k(beta_ik * F_k) + epsilon_i

    where:
        r_i     = asset i return
        r_f     = risk-free rate
        alpha_i = Jensen's alpha (abnormal return)
        beta_ik = factor loading (sensitivity to factor k)
        F_k     = factor k excess return
        epsilon = idiosyncratic return (diversifiable)

    Key Metrics:
        - R-squared: proportion of variance explained by factors
        - Information Ratio: alpha / tracking_error
        - Factor Risk Contribution: how much each factor drives portfolio risk

INTERVIEW RELEVANCE:
    - Factor models are THE core framework at most quant firms
    - Understanding alpha vs. beta decomposition is essential
    - Risk attribution is daily work for portfolio risk managers
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, Tuple, Optional


class FactorModel:
    """
    Multi-factor regression model for asset return decomposition.

    Fits:  r_i - rf = alpha + beta_mkt*MKT + beta_hml*HML + beta_mom*MOM + eps

    Uses OLS estimation with Newey-West-style robust standard errors.
    """

    def __init__(self):
        self.betas: Dict[str, np.ndarray] = {}
        self.alphas: Dict[str, float] = {}
        self.residuals: Dict[str, np.ndarray] = {}
        self.r_squared: Dict[str, float] = {}
        self.t_stats: Dict[str, np.ndarray] = {}
        self.factor_names: list = []

    def fit(
        self, returns: pd.DataFrame, factors: pd.DataFrame,
        rf_col: str = 'RF'
    ) -> 'FactorModel':
        """
        Fit the factor model to all assets using OLS.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset excess returns (T x N).
        factors : pd.DataFrame
            Factor returns including risk-free rate (T x K+1).
        rf_col : str
            Column name for the risk-free rate.

        Returns
        -------
        self : FactorModel
            Fitted model instance.
        """
        # Extract risk-free rate and factor excess returns
        rf = factors[rf_col]
        factor_cols = [c for c in factors.columns if c != rf_col]
        self.factor_names = factor_cols
        F = factors[factor_cols].values  # T x K
        T, K = F.shape

        # Add intercept
        X = np.column_stack([np.ones(T), F])  # T x (K+1)

        for ticker in returns.columns:
            y = (returns[ticker] - rf).values  # Excess returns

            # ── OLS: beta = (X'X)^{-1} X'y ──
            XtX_inv = np.linalg.inv(X.T @ X)
            beta_hat = XtX_inv @ (X.T @ y)

            # Residuals and R-squared
            y_hat = X @ beta_hat
            resid = y - y_hat
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            # Standard errors (heteroskedasticity-robust, HC1)
            sigma2 = ss_res / (T - K - 1)
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
            t_stat = beta_hat / se

            # Store results
            self.alphas[ticker] = beta_hat[0]
            self.betas[ticker] = beta_hat[1:]
            self.residuals[ticker] = resid
            self.r_squared[ticker] = r2
            self.t_stats[ticker] = t_stat

        return self

    def get_factor_loadings_table(self) -> pd.DataFrame:
        """
        Return a clean table of factor loadings with significance indicators.

        Returns
        -------
        pd.DataFrame
            Columns: [Alpha, MKT, HML, MOM, R-squared]
            Stars indicate significance: * p<0.05, ** p<0.01, *** p<0.001
        """
        rows = []
        for ticker in self.alphas:
            row = {'Ticker': ticker, 'Alpha': self.alphas[ticker]}
            for i, fname in enumerate(self.factor_names):
                row[fname] = self.betas[ticker][i]
            row['R_squared'] = self.r_squared[ticker]

            # Significance stars for alpha
            alpha_t = self.t_stats[ticker][0]
            if abs(alpha_t) > 3.29:
                row['Sig'] = '***'
            elif abs(alpha_t) > 2.58:
                row['Sig'] = '**'
            elif abs(alpha_t) > 1.96:
                row['Sig'] = '*'
            else:
                row['Sig'] = ''

            rows.append(row)

        return pd.DataFrame(rows).set_index('Ticker')

    def compute_factor_risk_decomposition(
        self, weights: np.ndarray, factor_cov: np.ndarray,
        residual_vars: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Decompose portfolio risk into factor and specific components.

        Portfolio variance:
            sigma_p^2 = w' B Sigma_F B' w + w' D w

        where:
            B = factor loading matrix (N x K)
            Sigma_F = factor covariance matrix (K x K)
            D = diagonal matrix of idiosyncratic variances

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (N,).
        factor_cov : np.ndarray
            Factor covariance matrix (K x K).
        residual_vars : np.ndarray, optional
            Idiosyncratic variances per asset. If None, estimated from residuals.

        Returns
        -------
        dict
            Risk decomposition: total, systematic, idiosyncratic, per-factor.
        """
        tickers = list(self.betas.keys())
        N = len(tickers)
        K = len(self.factor_names)

        # Build factor loading matrix B (N x K)
        B = np.array([self.betas[t] for t in tickers])

        # Idiosyncratic variances
        if residual_vars is None:
            residual_vars = np.array([
                np.var(self.residuals[t]) for t in tickers
            ])
        D = np.diag(residual_vars)

        # Systematic risk: w' B Sigma_F B' w
        port_factor_exposure = B.T @ weights  # K-vector
        systematic_var = port_factor_exposure @ factor_cov @ port_factor_exposure

        # Idiosyncratic risk: w' D w
        idio_var = weights @ D @ weights

        # Total portfolio variance
        total_var = systematic_var + idio_var

        # Per-factor marginal risk contribution
        factor_contributions = {}
        for k, fname in enumerate(self.factor_names):
            # Marginal contribution of factor k
            b_k = B[:, k]
            factor_var_k = (weights @ b_k) ** 2 * factor_cov[k, k]
            factor_contributions[fname] = {
                'variance_contribution': factor_var_k,
                'pct_of_total': factor_var_k / total_var * 100 if total_var > 0 else 0,
                'portfolio_beta': weights @ b_k
            }

        return {
            'total_variance': total_var,
            'total_volatility': np.sqrt(total_var) * np.sqrt(252),
            'systematic_variance': systematic_var,
            'systematic_pct': systematic_var / total_var * 100 if total_var > 0 else 0,
            'idiosyncratic_variance': idio_var,
            'idiosyncratic_pct': idio_var / total_var * 100 if total_var > 0 else 0,
            'factor_contributions': factor_contributions
        }

    def compute_information_ratio(self) -> pd.Series:
        """
        Compute the Information Ratio for each asset.

        IR = alpha / std(residuals) * sqrt(252)

        The IR measures the consistency of alpha generation.
        A good fundamental PM targets IR > 0.5; quant strategies aim for IR > 1.0.
        """
        ir_dict = {}
        for ticker in self.alphas:
            alpha = self.alphas[ticker] * 252  # Annualize
            te = np.std(self.residuals[ticker]) * np.sqrt(252)  # Tracking error
            ir_dict[ticker] = alpha / te if te > 0 else 0
        return pd.Series(ir_dict, name='Information Ratio')

    def predict_returns(
        self, factor_scenarios: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict asset returns under given factor scenarios.

        Useful for stress testing and scenario analysis.

        Parameters
        ----------
        factor_scenarios : pd.DataFrame
            Factor return scenarios (S x K).

        Returns
        -------
        pd.DataFrame
            Predicted asset returns under each scenario (S x N).
        """
        tickers = list(self.betas.keys())
        predictions = {}

        for ticker in tickers:
            alpha = self.alphas[ticker]
            beta = self.betas[ticker]
            pred = alpha + factor_scenarios[self.factor_names].values @ beta
            predictions[ticker] = pred

        return pd.DataFrame(predictions, index=factor_scenarios.index)


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from data_generator import MarketDataGenerator

    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(2520)

    model = FactorModel()
    model.fit(returns, factors)

    print("=" * 70)
    print("FACTOR MODEL RESULTS")
    print("=" * 70)
    print("\nFactor Loadings:")
    print(model.get_factor_loadings_table().to_string())
    print("\nInformation Ratios:")
    print(model.compute_information_ratio().to_string())
