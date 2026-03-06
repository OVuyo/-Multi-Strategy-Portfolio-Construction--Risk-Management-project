# Institutional-Grade Multi-Strategy Portfolio Construction & Risk Management Engine project


## PROJECT OVERVIEW

### Problem Statement

*"Design and implement an end-to-end quantitative research pipeline that constructs, optimizes, backtests, and risk-manages a multi-asset portfolio using 7 distinct optimization strategies — from classical Markowitz Mean-Variance to modern machine-learning-based Hierarchical Risk Parity — with institutional-quality risk analytics including parametric/historical/Monte Carlo VaR, CVaR, stress testing, factor risk decomposition, and regime-aware allocation."*

This project demonstrates the **complete workflow of a quantitative analyst** . It covers every stage from data analysis to portfolio construction to risk reporting.

## PROJECT ARCHITECTURE

```
quant_project/
├── main.py                    # Master orchestrator pipeline
├── data_generator.py          # Module 1: Synthetic data with factor structure
├── factor_model.py            # Module 2: Fama-French factor decomposition
├── risk_engine.py             # Module 3: VaR, CVaR, stress testing
├── optimization_engine.py     # Module 4: 7 optimization strategies
├── regime_detection.py        # Module 5: GMM-based regime detection
├── ml_signals.py              # Module 6: ML return prediction
├── backtester.py              # Module 7: Walk-forward backtesting
├── performance_analytics.py   # Module 8: Visualization & reporting
├── README.md                  # This guide
└── output/                    # Generated charts and CSVs
    ├── 01_strategy_comparison.png
    ├── 02_efficient_frontier.png
    ├── 03_risk_decomposition.png
    ├── 04_weight_allocations.png
    ├── 05_regime_analysis.png
    ├── 06_stress_test.png
    ├── 07_correlation_matrix.png
    ├── strategy_comparison.csv
    ├── strategy_weights.csv
    └── factor_loadings.csv
```
## PROJECT structure - MODULE-BY-MODULE GUIDE

### Module 1: Data Generator (`data_generator.py`)

**Question:** How do we generate realistic synthetic financial data that exhibits the stylized facts observed in real markets?

**Theory:** Real asset returns exhibit four key properties:
- **Fat tails** (leptokurtosis) — extreme returns occur more frequently than a normal distribution predicts
- **Volatility clustering** — large returns tend to be followed by large returns (GARCH effect)
- **Cross-asset correlations** — driven by common factors
- **Regime dependence** — markets alternate between bull, bear, and crisis states

**Implementation:**
- Multi-factor return model: `r_i = alpha_i + beta_i * F + epsilon_i`
- GARCH(1,1) conditional variance for volatility clustering
- Two-state Markov chain for regime switching
- 12-asset universe spanning Tech, Financials, Healthcare, Energy, Real Estate, Commodities, Fixed Income
---
### Module 2: Factor Model (`factor_model.py`)

**Question:** How do we decompose asset returns into systematic (factor) and idiosyncratic components?

**Theory:** The Arbitrage Pricing Theory (APT) multi-factor model:

```
r_i - r_f = alpha_i + beta_MKT * MKT + beta_HML * HML + beta_MOM * MOM + epsilon_i
```

where:
- `alpha_i` = Jensen's alpha (abnormal return not explained by factors)
- `beta_ik` = factor loading (sensitivity to factor k)
- `R-squared` = proportion of variance explained by systematic factors

**Key Metrics Computed:**
- Factor loadings (betas) with significance tests
- R-squared per asset
- Information Ratio: `IR = alpha / tracking_error`
- Factor risk decomposition using Euler's theorem

---

### Module 3: Risk Engine (`risk_engine.py`)

**Question:** How do we measure and manage portfolio risk across multiple methodologies?

**Theory:** Three approaches to Value-at-Risk:

1. **Parametric VaR**: `VaR = -(mu + z_alpha * sigma)` — assumes normality
2. **Historical VaR**: alpha-quantile of empirical distribution — no distributional assumptions
3. **Cornish-Fisher VaR**: adjusts for skewness and kurtosis — superior for fat tails

**Conditional VaR (Expected Shortfall):**
```
CVaR = E[Loss | Loss > VaR]
```
CVaR is a *coherent* risk measure (satisfies subadditivity), unlike VaR.

**Stress Testing Scenarios:**
- 2008 Global Financial Crisis
- 2020 COVID Crash
- Tech Bubble Burst
- Rising Rates Shock
- Stagflation
- Flash Crash

**Marginal Risk Contributions:** Using Euler decomposition:
```
MCTR_i = (Sigma @ w)_i / sigma_p
RC_i = w_i * MCTR_i

```
---

### Module 4: Optimization Engine (`optimization_engine.py`)

**Question:** Given N assets, how do we allocate capital optimally?

**5 Strategies Implemented:**

#### 1. Mean-Variance Optimization (Markowitz, 1952)
```
max  (w'mu - rf) / sqrt(w'Sigma w)    [Max Sharpe]
min  w'Sigma w                          [Min Variance]
```
**Weakness:** Highly sensitive to estimation errors in mu and Sigma.

#### 2. Black-Litterman (1992)
Combines CAPM equilibrium returns with investor views:
```
mu_BL = [(tau*Sigma)^{-1} + P'Omega^{-1}P]^{-1} * [(tau*Sigma)^{-1}*pi + P'Omega^{-1}*Q]
```
**Strength:** Produces much more stable, intuitive weights than raw MVO.

#### 3. Risk Parity (Qian, 2005)
Each asset contributes equally to portfolio risk:
```
RC_i = w_i * (Sigma w)_i / sigma_p = sigma_p / N  for all i
```
**Used by:** Bridgewater's All Weather fund.

#### 4. Minimum CVaR (Rockafellar & Uryasev, 2000)
Minimizes expected loss in the tail:
```
min CVaR = min [ zeta + 1/(alpha*T) * sum max(0, -r_t'w - zeta) ]
```
**Advantage:** More robust to tail risk than variance minimization.

#### 5. Hierarchical Risk Parity (Lopez de Prado, 2016)
Uses hierarchical clustering — no matrix inversion needed:
1. Tree Clustering — cluster assets by correlation distance
2. Quasi-Diagonalization — reorder assets along cluster tree
3. Recursive Bisection — allocate risk top-down
---

### Module 5: Regime Detection (`regime_detection.py`)

**Question:** Can we identify latent market states and adapt portfolio allocation accordingly?

**Theory:** Hidden Markov Models assume returns are generated by an unobservable Markov chain:
- State: `S_t in {Bull, Bear, Crisis}`
- Each state has its own return distribution: `r_t ~ N(mu_{S_t}, sigma_{S_t})` 
  
**Implementation:**
- Gaussian Mixture Model as a practical HMM approximation
- Features: rolling momentum, volatility, skewness, drawdown, vol acceleration
- 3 regimes identified with distinct risk/return characteristics

**Regime-Adaptive Allocation:**
```python
risk_scaling = {Bull': 1.0, 'Bear': 0.7, 'Crisis': 0.4}
```
**Advantage:** Numerically stable, better out-of-sample performance.

---


---

### Module 6: ML Signals (`ml_signals.py`)

**Question:** Can machine learning predict forward returns to generate alpha?

**Feature Engineering (45+ features per asset):**
- **Momentum:** 5/21/63-day rolling returns, momentum crossover
- **Volatility:** realized vol, vol ratio (short/long)
- **Technical:** RSI, Bollinger Band width
- **Statistical:** rolling skewness, kurtosis, autocorrelation
- **Cross-sectional:** relative momentum rank

**Models:**
- Ridge Regression (L2 regularization)
- Random Forest (non-linear, captures interactions)
- Ensemble average (robustness)

**Key Insight:** In quantitative finance, an Information Coefficient (rank correlation between prediction and realization) of 0.05-0.10 is considered good. The signal-to-noise ratio is extremely low.

---

### Module 7: Backtester (`backtester.py`)

**Question:** How do we rigorously evaluate strategy performance without look-ahead bias?

**Walk-Forward Protocol:**
1. Train on `[0, T]`, optimize weights
2. Trade from `T` to `T + rebalance_period`
3. Expand window, repeat
4. Apply transaction costs at each rebalance

**Metrics Computed:**
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Alpha, Beta (vs. benchmark)
- Information Ratio, Tracking Error
- Maximum Drawdown, Win Rate
- Total Turnover, Transaction Cost Drag

---

### Module 8: Performance Analytics (`performance_analytics.py`)

**Generates 7 quality visualizations:**
1. Strategy Comparison Dashboard (cumulative returns, drawdowns, rolling Sharpe)
2. Efficient Frontier with strategy positions
3. Risk Decomposition (systematic vs. idiosyncratic)
4. Weight Allocations by strategy
5. Regime Analysis (returns colored by detected regime)
6. Stress Test results
7. Correlation Heatmap

---

## KEY RESULTS

### Strategy Performance (Walk-Forward Backtest)

| Strategy     | Ann. Return | Ann. Vol | Sharpe | Max DD   | Alpha  |
|-------------|------------|---------|--------|----------|--------|
| Min CVaR    | 2.51%      | 4.16%   | 0.604  | -12.21%  | 2.43%  |
| Max Sharpe  | 2.43%      | 5.57%   | 0.436  | -12.24%  | 2.34%  |
| HRP         | 2.02%      | 6.39%   | 0.316  | -15.30%  | 1.71%  |
| Risk Parity | 0.30%      | 15.45%  | 0.019  | -39.86%  | -0.54% |
| Equal Wt    | 0.14%      | 26.22%  | 0.005  | -68.92%  | -1.33% |
| Min Var     | -0.44%     | 10.76%  | -0.041 | -27.20%  | -1.00% |

### Key Findings
- **Min CVaR** produced the highest Sharpe ratio (0.604) with the tightest max drawdown
- **HRP** performed well without requiring covariance matrix inversion
- **Equal Weight** underperformed — naive diversification is insufficient
- Regime detection identified 3 distinct states with meaningfully different characteristics

---




## HOW TO RUN

```bash
# Install dependencies (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn)
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Run the full pipeline
python main.py

# Run individual modules
python data_generator.py
python factor_model.py
python risk_engine.py
python optimization_engine.py
python regime_detection.py
python ml_signals.py
python backtester.py
```

---

---

## REFERENCES

1. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*.
2. Black, F. & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*.
3. Qian, E. (2005). "Risk Parity Portfolios." *PanAgora Asset Management*.
4. Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*.
5. Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*.
6. Fama, E.F. & French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*.

---
