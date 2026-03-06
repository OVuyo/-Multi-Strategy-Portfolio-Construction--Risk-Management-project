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
