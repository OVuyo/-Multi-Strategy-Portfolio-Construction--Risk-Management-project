"""
================================================================================
MODULE 6: Machine Learning Signal Generator
================================================================================
PURPOSE:
    Build ML-based return prediction signals using engineered features.
    Combines technical indicators, cross-sectional features, and macro
    signals to generate alpha forecasts.

THEORY:
    ML in quant finance differs from general ML:
    - Signal-to-noise ratio is extremely low (~0.01 R-squared is good!)
    - Non-stationarity means features drift over time
    - Overfitting is the #1 enemy — walk-forward validation is essential
    - Feature engineering matters more than model complexity

    Features used:
    1. Technical: RSI, MACD, Bollinger Band width, volatility ratio
    2. Cross-sectional: relative momentum, relative value
    3. Statistical: rolling skewness, kurtosis, autocorrelation

    Models:
    - Ridge Regression (L2 regularization prevents overfitting)
    - Random Forest (non-linear, handles interactions)
    - Ensemble (average of both for robustness)

INTERVIEW RELEVANCE:
    - ML in finance is the hottest area in quant hiring
    - Shows understanding of the unique challenges (low SNR, non-stationarity)
    - Feature engineering is where domain expertise creates edge
    - Walk-forward testing shows awareness of look-ahead bias
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List, Optional


class MLSignalGenerator:
    """
    Machine learning signal generator for return prediction.

    Engineers financial features and trains models to predict
    forward returns for use in portfolio allocation.
    """

    def __init__(self, prediction_horizon: int = 21):
        """
        Parameters
        ----------
        prediction_horizon : int
            Forward return horizon in trading days (default: 21 ≈ 1 month).
        """
        self.horizon = prediction_horizon
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.feature_names: List[str] = []
        self.feature_importance: Dict = {}

    # ══════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════

    def engineer_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create a rich feature set from raw returns.

        Feature Categories:
        1. Momentum features (trend following)
        2. Mean-reversion features (contrarian signals)
        3. Volatility features (risk regime)
        4. Cross-sectional features (relative value)
        5. Statistical features (distribution shape)

        Parameters
        ----------
        returns : pd.DataFrame
            Daily asset returns (T x N).

        Returns
        -------
        pd.DataFrame
            Feature matrix (T x F) where F is the number of features.
        """
        features = pd.DataFrame(index=returns.index)

        for ticker in returns.columns:
            r = returns[ticker]
            prefix = ticker

            # ── Momentum Features ──
            # Short-term momentum (5-day)
            features[f'{prefix}_mom_5'] = r.rolling(5).mean()
            # Medium-term momentum (21-day)
            features[f'{prefix}_mom_21'] = r.rolling(21).mean()
            # Long-term momentum (63-day)
            features[f'{prefix}_mom_63'] = r.rolling(63).mean()
            # Momentum crossover (short vs long)
            features[f'{prefix}_mom_cross'] = (
                features[f'{prefix}_mom_5'] - features[f'{prefix}_mom_63']
            )

            # ── Volatility Features ──
            # Realized volatility (21-day)
            features[f'{prefix}_vol_21'] = r.rolling(21).std() * np.sqrt(252)
            # Volatility ratio (short/long)
            vol_5 = r.rolling(5).std()
            vol_63 = r.rolling(63).std()
            features[f'{prefix}_vol_ratio'] = vol_5 / vol_63.replace(0, np.nan)

            # ── RSI (Relative Strength Index) ──
            delta = r.copy()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            features[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))

            # ── Bollinger Band Width ──
            ma20 = r.rolling(20).mean()
            std20 = r.rolling(20).std()
            features[f'{prefix}_bb_width'] = (2 * std20) / ma20.abs().replace(0, np.nan)

            # ── Statistical Features ──
            features[f'{prefix}_skew_21'] = r.rolling(21).skew()
            features[f'{prefix}_kurt_21'] = r.rolling(21).kurt()

            # ── Autocorrelation (mean-reversion signal) ──
            features[f'{prefix}_autocorr'] = r.rolling(21).apply(
                lambda x: x.autocorr(), raw=False
            )

        # ── Cross-Sectional Features ──
        # Relative momentum rank (cross-sectional)
        mom_21 = returns.rolling(21).mean()
        ranks = mom_21.rank(axis=1, pct=True)
        for ticker in returns.columns:
            features[f'{ticker}_cs_rank'] = ranks[ticker]

        # ── Macro Features ──
        # Market average return and vol (common factors)
        mkt = returns.mean(axis=1)
        features['mkt_mom_21'] = mkt.rolling(21).mean()
        features['mkt_vol_21'] = mkt.rolling(21).std() * np.sqrt(252)
        features['mkt_skew'] = mkt.rolling(63).skew()

        # Clean
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()

        self.feature_names = features.columns.tolist()
        return features

    # ══════════════════════════════════════════════════════════════════
    # TARGET VARIABLE
    # ══════════════════════════════════════════════════════════════════

    def compute_forward_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward returns for each asset.

        target_i,t = sum(r_i, t+1 : t+H)

        This is the variable we're predicting — the cumulative
        return over the next `prediction_horizon` days.
        """
        fwd = returns.rolling(self.horizon).sum().shift(-self.horizon)
        return fwd

    # ══════════════════════════════════════════════════════════════════
    # MODEL TRAINING WITH WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════

    def fit(
        self, returns: pd.DataFrame,
        train_pct: float = 0.7
    ) -> Dict:
        """
        Train ML models using walk-forward (expanding window) validation.

        Walk-forward avoids look-ahead bias:
        - Train on [0, t], predict t+1
        - Expand window, repeat

        For efficiency, we use a single train/test split here,
        but the backtester module implements full walk-forward.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns.
        train_pct : float
            Fraction used for training.

        Returns
        -------
        dict
            Training results including metrics and feature importance.
        """
        features = self.engineer_features(returns)
        targets = self.compute_forward_returns(returns)

        # Align features and targets
        common_idx = features.index.intersection(targets.dropna().index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]

        # Train/test split (temporal — no shuffling!)
        split_idx = int(len(X) * train_pct)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

        results = {}

        for ticker in returns.columns:
            y = y_all[ticker]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            # Drop rows with NaN targets
            valid_train = ~y_train.isna()
            valid_test = ~y_test.isna()

            Xtr = X_train[valid_train]
            ytr = y_train[valid_train]
            Xte = X_test[valid_test]
            yte = y_test[valid_test]

            if len(Xtr) < 100 or len(Xte) < 50:
                continue

            # Standardize features
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            # ── Model 1: Ridge Regression ──
            ridge = Ridge(alpha=10.0)
            ridge.fit(Xtr_s, ytr)
            ridge_pred = ridge.predict(Xte_s)

            # ── Model 2: Random Forest ──
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=20, random_state=42, n_jobs=-1
            )
            rf.fit(Xtr_s, ytr)
            rf_pred = rf.predict(Xte_s)

            # ── Ensemble: Average ──
            ensemble_pred = (ridge_pred + rf_pred) / 2

            # ── Evaluate ──
            from scipy.stats import spearmanr
            corr_ridge = spearmanr(yte, ridge_pred)[0]
            corr_rf = spearmanr(yte, rf_pred)[0]
            corr_ensemble = spearmanr(yte, ensemble_pred)[0]

            # Store models
            self.models[ticker] = {
                'ridge': ridge, 'rf': rf, 'scaler': scaler
            }

            # Feature importance (from RF)
            self.feature_importance[ticker] = pd.Series(
                rf.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)

            results[ticker] = {
                'ridge_ic': corr_ridge,
                'rf_ic': corr_rf,
                'ensemble_ic': corr_ensemble,
                'n_train': len(Xtr),
                'n_test': len(Xte),
            }

        self.results = results
        return results

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate return predictions using trained models.

        Returns
        -------
        pd.DataFrame
            Predicted forward returns for each asset.
        """
        features = self.engineer_features(returns)
        predictions = pd.DataFrame(index=features.index)

        for ticker, model_dict in self.models.items():
            scaler = model_dict['scaler']
            ridge = model_dict['ridge']
            rf = model_dict['rf']

            X_scaled = scaler.transform(features)
            pred_ridge = ridge.predict(X_scaled)
            pred_rf = rf.predict(X_scaled)
            predictions[ticker] = (pred_ridge + pred_rf) / 2

        return predictions

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Return top-N most important features averaged across assets."""
        if not self.feature_importance:
            return pd.DataFrame()

        avg_importance = pd.DataFrame(self.feature_importance).mean(axis=1)
        return avg_importance.sort_values(ascending=False).head(n)

    def get_model_metrics(self) -> pd.DataFrame:
        """Return model performance metrics."""
        return pd.DataFrame(self.results).T


# ── Module self-test ──────────────────────────────────────────────────
if __name__ == '__main__':
    from data_generator import MarketDataGenerator

    gen = MarketDataGenerator(seed=42)
    prices, returns, factors = gen.generate_asset_returns(2520)

    ml = MLSignalGenerator(prediction_horizon=21)
    results = ml.fit(returns)

    print("=" * 70)
    print("ML SIGNAL GENERATOR RESULTS")
    print("=" * 70)
    print("\nModel Performance (Information Coefficient):")
    print(ml.get_model_metrics().round(4).to_string())
    print("\nTop 10 Features:")
    print(ml.get_top_features(10).round(4).to_string())
