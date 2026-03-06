# FRM Exam Prep — Quantitative Financial Risk Management
## Quantitative Financial Risk Management — Complete Study Guide

---

### What's in this folder

| File | What it covers |
|------|---------------|
| `WORKPLACE_SCENARIOS_AND_PIPELINES.md` | 9 real-world scenarios with step-by-step pipelines and interpretation guides |
| `ch02_standard_deviation.py` | EWMA, GARCH, rectangular window volatility |
| `ch03_value_at_risk.py` | Delta-normal, historical, Monte Carlo, Cornish-Fisher VaR + backtesting |
| `ch04_expected_shortfall_evt.py` | Expected Shortfall, EVT, GEV/GPD fitting |
| `ch05_portfolio_correlation.py` | Covariance matrices, beta, minimum variance portfolios |
| `ch06_beyond_correlation.py` | Copulas, joint distributions, stress testing |
| `ch07_risk_attribution.py` | Component VaR, marginal VaR, incremental VaR |
| `ch08_credit_risk.py` | Merton model, default probability, credit spreads, CDS |
| `ch09_to_ch12_advanced_topics.py` | Liquidity risk, Bayesian analysis, behavioral biases, MLE, copulas |
| `run_all.py` | Master runner — executes all scripts in sequence |

### How to use

1. **Read the guide first** — open `COMPLETE_GUIDE_*.md` and go chapter by chapter
2. **Run the scripts** — each script generates charts and prints key outputs
3. **Check workplace scenarios** — `WORKPLACE_SCENARIOS_AND_PIPELINES.md` shows you how everything connects to real tasks

### Requirements

```bash
pip install numpy pandas scipy matplotlib statsmodels
```

### Quick start

```bash
python run_all.py          # runs everything
python ch02_standard_deviation.py  # run individual chapters
```

---
*Based on "Quantitative Financial Risk Management" by Michael B. Miller (Wiley Finance)*
