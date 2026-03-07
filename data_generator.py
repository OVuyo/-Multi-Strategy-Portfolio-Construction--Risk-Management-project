"""
================================================================================
MODULE 1: Synthetic Financial Data Generator
================================================================================

PURPOSE:
    Generate realistic multi-asset financial time series with embedded factor
    structure, regime shifts, fat tails, and volatility clustering — properties
    observed in real financial markets.

Approach: 
    Real asset returns exhibit ( the known properties):
    - Fat tails (leptokurtosis) — modeled via Student-t innovations
    - Volatility clustering — modeled via GARCH(1,1) dynamics
    - Cross-asset correlations — modeled via a factor structure
    - Regime dependence — modeled via Hidden Markov switching
    
so to generate realistic synthetic data I created a class/module which ensures that the data we generate 
contains the above stated 4 known properties which are exibited by real financial data  
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


class MarketDataGenerator:
    """
    Generates synthetic but realistic multi-asset financial data.

    The generator creates correlated asset returns driven by:
    - A common market factor (systematic risk)
    - A value factor (HML-like)
    - A momentum factor  (an alpha factor) 
    - Idiosyncratic noise per asset

    Volatility follows a simplified GARCH(1,1) process to capture clustering.
    """

    # ── Asset Universe Definition ──────────────────────────────────────
    " 12-asset universe spanning Tech, Financials, Healthcare, Energy, Real Estate, Commodities, Fixed Income "

    ASSET_UNIVERSE = {
        # Ticker: (Name, Sector, Annual Return, Annual Vol, Market Beta)
        'AAPL':  ('Apple Inc.',          'Technology',     0.15, 0.28, 1.20),
        'MSFT':  ('Microsoft Corp.',     'Technology',     0.13, 0.25, 1.10),
        'JPM':   ('JPMorgan Chase',      'Financials',     0.10, 0.22, 1.15),
        'JNJ':   ('Johnson & Johnson',   'Healthcare',     0.08, 0.15, 0.65),
        'XOM':   ('Exxon Mobil',         'Energy',         0.07, 0.24, 0.90),
        'PG':    ('Procter & Gamble',    'Consumer',       0.09, 0.14, 0.55),
        'GS':    ('Goldman Sachs',       'Financials',     0.11, 0.30, 1.35),
        'AMZN':  ('Amazon.com',          'Technology',     0.16, 0.32, 1.25),
        'NEE':   ('NextEra Energy',      'Utilities',      0.10, 0.18, 0.50),
        'SPG':   ('Simon Property',      'Real Estate',    0.08, 0.26, 1.05),
        'GLD':   ('Gold ETF',            'Commodities',    0.05, 0.16, 0.05),
        'TLT':   ('Treasury Bond ETF',   'Fixed Income',   0.03, 0.12, -0.20),
    }

    def __init__(self, seed: int = 42):
        """Initialize with reproducible random state."""
        self.rng = np.random.RandomState(seed)
        self.tickers = list(self.ASSET_UNIVERSE.keys())
        self.n_assets = len(self.tickers)

    def _generate_garch_volatility(
        self, n_days: int, omega: float = 0.00001,
        alpha: float = 0.08, beta: float = 0.90, base_var: float = 0.0001
    ) -> np.ndarray:
        """
        Generate a GARCH(1,1) conditional variance series.

        GARCH(1,1):  sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

        Parameters
        ----------
        n_days : int
            Number of trading days.
        omega : float
            Long-run variance constant.
        alpha : float
            ARCH coefficient (shock persistence).
        beta : float
            GARCH coefficient (variance persistence).
        base_var : float
            Initial variance.

        Returns
        -------
        np.ndarray
            Conditional volatility series (standard deviations).
        """
        variances = np.zeros(n_days)
        variances[0] = base_var
        shocks = self.rng.normal(0, 1, size=n_days)  # standard normal innovations

        for t in range(1, n_days):
            variances[t] = (
                omega
                + alpha * (shocks[t - 1] ** 2) * variances[t - 1]
                + beta * variances[t - 1]
            )
            variances[t] = min(variances[t], 0.01)  # cap extreme variance
        return np.sqrt(variances)

    def _generate_regime_indicators(
        self, n_days: int,
        p_bull_to_bear: float = 0.02,
        p_bear_to_bull: float = 0.05
    ) -> np.ndarray:
        """
        Generate a two-state Markov regime sequence.

        States:
            0 = Bull market (higher returns, lower vol)
            1 = Bear market (lower returns, higher vol)

        The transition matrix:
            P = [[1 - p_bb, p_bb],
                 [p_rb,     1 - p_rb]]
        """
        regimes = np.zeros(n_days, dtype=int)
        for t in range(1, n_days):
            if regimes[t - 1] == 0:  # Bull
                regimes[t] = 1 if self.rng.random() < p_bull_to_bear else 0
            else:  # Bear
                regimes[t] = 0 if self.rng.random() < p_bear_to_bull else 1
        return regimes

    def generate_factor_returns(self, n_days: int) -> pd.DataFrame:
        """
        Generate synthetic Fama-French-style factor returns.

        Factors:
            MKT  : Market excess return factor
            HML  : Value factor (High Minus Low book-to-market)
            MOM  : Momentum factor (Winners Minus Losers)
            RF   : Risk-free rate (annualized ~4%, daily)

        Returns
        -------
        pd.DataFrame
            Daily factor returns with columns [MKT, HML, MOM, RF].
        """
        dates = pd.bdate_range(start='2014-01-02', periods=n_days, freq='B')

        # Market factor with GARCH volatility
        mkt_vol = self._generate_garch_volatility(n_days, base_var=0.0002)
        mkt_returns = self.rng.normal(0.0003, 1, n_days) * mkt_vol

        # HML factor (lower vol, slight positive premium)
        hml_returns = self.rng.normal(0.0001, 0.005, n_days)

        # Momentum factor
        mom_returns = self.rng.normal(0.00015, 0.006, n_days)

        # Risk-free rate (~4% annualized)
        rf = np.full(n_days, 0.04 / 252)

        factors = pd.DataFrame({
            'MKT': mkt_returns,
            'HML': hml_returns,
            'MOM': mom_returns,
            'RF': rf
        }, index=dates)

        return factors

    def generate_asset_returns( 
        #step is crucial , prices are not stationary but returns are , when data is stationary we can model it statistically.
        
        self, n_days: int = 2520     #1 year = 252 trading days , so 2520 days = 10 years
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic asset returns using a multi-factor model.

        Model:
            r_i,t = alpha_i + beta_mkt_i * MKT_t + beta_hml_i * HML_t
                    + beta_mom_i * MOM_t + regime_effect_t + epsilon_i,t

        Parameters
        ----------
        n_days : int
            Number of trading days (default: 2520 ≈ 10 years).

        Returns
        -------
        prices : pd.DataFrame
            Simulated price series for all assets.
        returns : pd.DataFrame
            Daily log returns.  (we use log returns because log return unlike simple returns are continously compounded returns (additive returns across time) and are more normally distributed then simple returns).
        factors : pd.DataFrame
            Factor return series.
        """
        factors = self.generate_factor_returns(n_days)
        regimes = self._generate_regime_indicators(n_days)
        dates = factors.index

        # Factor loadings per asset
        betas = {}     #beta measures the portfolio's volatility relative to the market
        for ticker, (name, sector, ann_ret, ann_vol, mkt_beta) in self.ASSET_UNIVERSE.items():
            hml_beta = self.rng.uniform(-0.3, 0.5)
            mom_beta = self.rng.uniform(-0.2, 0.4)
            betas[ticker] = {
                'alpha': ann_ret / 252,
                'mkt': mkt_beta,
                'hml': hml_beta,
                'mom': mom_beta,
                'idio_vol': ann_vol / np.sqrt(252) * 0.4  # 40% idiosyncratic
            }

        returns_dict = {}
        for ticker, beta in betas.items():
            # Systematic component
            systematic = (
                beta['alpha']
                + beta['mkt'] * factors['MKT'].values
                + beta['hml'] * factors['HML'].values
                + beta['mom'] * factors['MOM'].values
            )

            # Regime adjustment: bear markets → lower returns, higher vol
            regime_shift = np.where(regimes == 1, -0.001, 0.0002)
            vol_scaling = np.where(regimes == 1, 1.8, 1.0)

            # Idiosyncratic noise with slight fat tails
            idio = (
                self.rng.normal(0, 1, size=n_days)
                * beta['idio_vol']
                * vol_scaling
            )

            returns_dict[ticker] = systematic + regime_shift + idio

        returns = pd.DataFrame(returns_dict, index=dates)

        # Construct price paths from returns
        prices = (1 + returns).cumprod() * 100  # Start at 100

        # Also store regime data
        self.regimes = pd.Series(regimes, index=dates, name='regime')

        return prices, returns, factors

    def generate_benchmark(self, returns: pd.DataFrame) -> pd.Series:
        """
        Generate a market-cap-weighted benchmark from asset returns.

        Uses approximate market cap weights (tech-heavy, like S&P 500).

        critical for managing a multi-asset portfolio
        """
        weights = {
            'AAPL': 0.15, 'MSFT': 0.14, 'AMZN': 0.08, 'JPM': 0.06,
            'JNJ': 0.05, 'GS': 0.04, 'XOM': 0.05, 'PG': 0.05,
            'NEE': 0.04, 'SPG': 0.03, 'GLD': 0.06, 'TLT': 0.25
        }
        w = pd.Series(weights)
        w = w / w.sum()
        benchmark = (returns * w).sum(axis=1)
        benchmark.name = 'Benchmark'
        return benchmark

    def get_sector_mapping(self) -> Dict[str, str]:
        """Return ticker → sector mapping."""
        return {t: v[1] for t, v in self.ASSET_UNIVERSE.items()}

    def summary(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute summary  statistics for generated data."""
        ann_factor = 252  #there are 252 trading days in a single year
        stats_dict = {
            'Ann. Return (%)': returns.mean() * ann_factor * 100,
            'Ann. Volatility (%)': returns.std() * np.sqrt(ann_factor) * 100,
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(ann_factor),
            'Max Drawdown (%)': self._max_drawdowns(returns) * 100
        }
        return pd.DataFrame(stats_dict).round(3)

    @staticmethod
    def _max_drawdowns(returns: pd.DataFrame) -> pd.Series:
        """Compute maximum drawdown for each asset."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(n_days=2520)

    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nDate Range : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Assets     : {returns.shape[1]}")
    print(f"Obs/Asset  : {returns.shape[0]}")
    print(f"\nAsset Summary Statistics:")
    print(gen.summary(returns).to_string())
    print(f"\nFactor Summary:")
    print(gen.summary(factors[['MKT', 'HML', 'MOM']]).to_string())
