#!/usr/bin/env python3
"""
ML-Enhanced Pair Trading Strategy - Quant Research Approach
- Cointegration for pair selection (lower threshold for more pairs)
- XGBoost and Random Forest for signal generation
- Realistic target labeling: entry/hold/exit based on returns
- Simple and clean implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
from datetime import datetime, timedelta
from scipy.stats import binomtest
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

class MLEnhancedStrategy:
    def __init__(self, transaction_cost=0.001, max_pairs=30, short_selling=True):
        self.transaction_cost = transaction_cost
        self.max_pairs = max_pairs
        self.short_selling = short_selling
        self.cointegrated_pairs = []
        self.models = {}
        self.trades = []
        self.performance = {}
        
    def load_data(self, data_path):
        """Load formation and trading data"""
        data_path = Path(data_path)
        
        # Find files
        formation_file = list(data_path.glob("*_in_sample_formation.csv"))[0]
        trading_file = list(data_path.glob("*_out_sample_trading.csv"))[0]
        
        print("🚀 ML-ENHANCED PAIR TRADING STRATEGY")
        print("="*60)
        print(f"📈 Formation: {formation_file.name}")
        print(f"💰 Trading: {trading_file.name}")
        
        # Load data
        self.formation_data = pd.read_csv(formation_file)
        self.trading_data = pd.read_csv(trading_file)
        
        # Set date index
        for df in [self.formation_data, self.trading_data]:
            date_col = 'period' if 'period' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Get price columns
        self.price_columns = [col for col in self.formation_data.columns if col.startswith('p_adjclose_')]
        if not self.price_columns:
            self.price_columns = [col for col in self.formation_data.columns if col.startswith('P_')]
        
        self.formation_prices = self.formation_data[self.price_columns]
        self.trading_prices = self.trading_data[self.price_columns]
        
        print(f"✅ Loaded {len(self.price_columns)} stocks")
        return self
    
    def find_cointegrated_pairs(self, significance_level=0.10):  # Lower threshold for more pairs
        """Find cointegrated pairs using Engle-Granger test"""
        print(f"\n🔍 Finding cointegrated pairs (significance: {significance_level*100}%)")
        
        all_pairs = []
        tested_pairs = 0
        
        # Test top 150 stocks for more pairs
        top_stocks = self.price_columns[:250]
        
        for i, stock1 in enumerate(top_stocks):
            for j, stock2 in enumerate(top_stocks):
                if i < j:
                    tested_pairs += 1
                    
                    # Test cointegration
                    s1 = self.formation_prices[stock1].dropna()
                    s2 = self.formation_prices[stock2].dropna()
                    common_dates = s1.index.intersection(s2.index)
                    
                    if len(common_dates) < 30:
                        continue
                    
                    s1_clean = s1[common_dates]
                    s2_clean = s2[common_dates]
                    
                    try:
                        coint_stat, p_value, critical_values = coint(s1_clean, s2_clean)
                        ols_result = OLS(s1_clean, s2_clean).fit()
                        hedge_ratio = ols_result.params[0]
                        r_squared = ols_result.rsquared
                        
                        if p_value < significance_level:
                            all_pairs.append({
                                'stock1': stock1, 'stock2': stock2,
                                'stock1_name': stock1.replace('p_adjclose_', '').replace('P_', ''),
                                'stock2_name': stock2.replace('p_adjclose_', '').replace('P_', ''),
                                'p_value': p_value, 'hedge_ratio': hedge_ratio,
                                'r_squared': r_squared, 'coint_stat': coint_stat
                            })
                    except:
                        continue
        
        # Sort by cointegration strength and take top pairs
        all_pairs = sorted(all_pairs, key=lambda x: x['p_value'])
        self.cointegrated_pairs = all_pairs[:self.max_pairs]
        
        print(f"   Found {len(self.cointegrated_pairs)} cointegrated pairs")
        for i, pair in enumerate(self.cointegrated_pairs[:15]):
            print(f"   {i+1}. {pair['stock1_name']}-{pair['stock2_name']}: p={pair['p_value']:.4f}, β={pair['hedge_ratio']:.3f}")
        
        return self
    
    def create_features_and_targets(self, prices, pair_info):
        """Create features and realistic targets based on returns (supports long/short)"""
        stock1, stock2 = pair_info['stock1'], pair_info['stock2']
        hedge_ratio = pair_info['hedge_ratio']
        
        # Get price data
        p1 = prices[stock1].dropna()
        p2 = prices[stock2].dropna()
        common_dates = p1.index.intersection(p2.index)
        
        if len(common_dates) < 100:
            return None, None, None, None
        
        p1, p2 = p1[common_dates], p2[common_dates]
        
        # Calculate cointegrating spread
        spread = p1 - hedge_ratio * p2
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        zscore = (spread - spread_mean) / spread_std
        
        # Simple features
        features = pd.DataFrame(index=common_dates)
        features['zscore'] = zscore
        features['zscore_ma5'] = zscore.rolling(5).mean()
        features['zscore_ma10'] = zscore.rolling(10).mean()
        features['zscore_std'] = zscore.rolling(10).std()
        
        # Price ratio features
        features['price_ratio'] = p1 / p2
        features['price_ratio_ma20'] = features['price_ratio'].rolling(20).mean()
        features['price_ratio_std'] = features['price_ratio'].rolling(10).std()
        
        # Volatility features
        features['p1_volatility'] = p1.pct_change().rolling(10).std()
        features['p2_volatility'] = p2.pct_change().rolling(10).std()
        
        # Target: Multi-class (0=hold, 1=long, 2=short) or binary (0=hold, 1=long)
        if self.short_selling:
            features['target'] = 0  # Default: hold
        else:
            features['target'] = 0  # Default: hold
        
        for i in range(len(features) - 10):  # Look ahead 10 days
            current_zscore = features['zscore'].iloc[i]
            
            # Calculate future spread return (next 10 days)
            future_spread = spread.iloc[i+10]
            current_spread = spread.iloc[i]
            spread_return = (future_spread - current_spread) / abs(current_spread) if abs(current_spread) > 0 else 0
            
            if self.short_selling:
                # Multi-class: 0=hold, 1=long, 2=short
                if current_zscore < -1.5 and spread_return > 0.02:  # Low zscore, spread increases
                    features['target'].iloc[i] = 1  # entry_long
                elif current_zscore > 1.5 and spread_return < -0.02:  # High zscore, spread decreases
                    features['target'].iloc[i] = 2  # entry_short
                else:
                    features['target'].iloc[i] = 0  # hold
            else:
                # Binary: 0=hold, 1=long only
                if current_zscore < -1.5 and spread_return > 0.02:  # Low zscore, spread increases
                    features['target'].iloc[i] = 1  # entry_long
                else:
                    features['target'].iloc[i] = 0  # hold
        
        # Remove NaN values
        features = features.dropna()
        
        return features, p1, p2, spread
    
    def train_xgboost(self, X, y):
        """Train XGBoost with enhanced hyperparameter tuning"""
        print("   Training XGBoost...")
        
        # Enhanced parameter space
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Enhanced random search
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        random_search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=50, cv=tscv, 
            scoring='f1_weighted', random_state=42, n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X, y)
        
        print(f"     Best score: {random_search.best_score_:.3f}")
        print(f"     Best params: {random_search.best_params_}")
        
        return random_search.best_estimator_
    
    def train_random_forest(self, X, y):
        """Train Random Forest with enhanced hyperparameter tuning"""
        print("   Training Random Forest...")
        
        # Enhanced parameter space
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
            'bootstrap': [True, False],
            'max_samples': [0.7, 0.8, 0.9, None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        rf_model = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            rf_model, param_dist, n_iter=50, cv=tscv,
            scoring='f1_weighted', random_state=42, n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X, y)
        
        print(f"     Best score: {random_search.best_score_:.3f}")
        print(f"     Best params: {random_search.best_params_}")
        
        return random_search.best_estimator_
    
    def train_models(self):
        """Train all ML models"""
        print(f"\n🤖 Training ML models on {len(self.cointegrated_pairs)} pairs...")
        
        # Combine data from all pairs
        all_features = []
        all_targets = []
        
        for pair in self.cointegrated_pairs:
            features, _, _, _ = self.create_features_and_targets(self.formation_prices, pair)
            if features is not None:
                feature_cols = [col for col in features.columns if col != 'target']
                all_features.append(features[feature_cols])
                all_targets.append(features['target'])
        
        if not all_features:
            print("❌ No training data found")
            return self
        
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)
        
        print(f"   Training on {len(X)} samples")
        print(f"   Target distribution:")
        for target in y.value_counts().items():
            print(f"     {target[0]}: {target[1]} ({target[1]/len(y)*100:.1f}%)")
        
        # Train models
        self.models['xgboost'] = self.train_xgboost(X, y)
        self.models['random_forest'] = self.train_random_forest(X, y)
        
        return self
    
    def generate_signals(self, features, pair_info):
        """Generate trading signals (supports long/short)"""
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols]
        
        # Get predictions from both models
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                if self.short_selling and pred_proba.shape[1] == 3:
                    # Multi-class: [hold, long, short]
                    predictions[name] = {
                        'long': pred_proba[:, 1],  # Class 1 = long entry
                        'short': pred_proba[:, 2]  # Class 2 = short entry
                    }
                else:
                    # Binary: [hold, long] or fallback
                    long_prob = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                    predictions[name] = {
                        'long': long_prob,
                        'short': np.zeros_like(long_prob)  # No short signals
                    }
            else:
                pred = model.predict(X)
                if self.short_selling:
                    long_signal = (pred == 1).astype(float)
                    short_signal = (pred == 2).astype(float)
                    predictions[name] = {
                        'long': long_signal,
                        'short': short_signal
                    }
                else:
                    long_signal = (pred == 1).astype(float)
                    predictions[name] = {
                        'long': long_signal,
                        'short': np.zeros_like(long_signal)
                    }
        
        # Ensemble probabilities
        ensemble_long = np.mean([pred['long'] for pred in predictions.values()], axis=0)
        ensemble_short = np.mean([pred['short'] for pred in predictions.values()], axis=0)
        
        return ensemble_long, ensemble_short
    
    def backtest_strategy(self):
        """Backtest the ML-enhanced strategy (supports long/short)"""
        mode = "LONG & SHORT" if self.short_selling else "LONG ONLY"
        print(f"\n💰 Backtesting ML-enhanced strategy ({mode})...")
        
        self.trades_by_model = {name: [] for name in self.models}
        self.trades_by_model['ensemble'] = []
        
        for pair in self.cointegrated_pairs:
            features, p1, p2, spread = self.create_features_and_targets(self.trading_prices, pair)
            if features is None:
                continue
            feature_cols = [col for col in features.columns if col != 'target']
            X = features[feature_cols]
            
            # Get model probabilities for both long and short
            model_probs = {}
            for name, model in self.models.items():
                long_probs, short_probs = self.generate_signals(features, pair)
                model_probs[name] = {'long': long_probs, 'short': short_probs}
            
            # Ensemble
            ensemble_long = np.mean([probs['long'] for probs in model_probs.values()], axis=0)
            ensemble_short = np.mean([probs['short'] for probs in model_probs.values()], axis=0)
            model_probs['ensemble'] = {'long': ensemble_long, 'short': ensemble_short}

            # For each model (and ensemble), backtest
            for model_name, probs in model_probs.items():
                long_probs = probs['long']
                short_probs = probs['short']
                
                # Generate signals
                long_signals = (long_probs > 0.6) & (features['zscore'] < -1.0)
                short_signals = (short_probs > 0.6) & (features['zscore'] > 1.0) if self.short_selling else np.zeros_like(long_signals, dtype=bool)
                
                position = 0  # 0=no position, 1=long, -1=short
                entry_info = {}
                trades = self.trades_by_model[model_name]
                
                for i, date in enumerate(features.index):
                    if position == 0:  # No position
                        if long_signals[i]:  # Enter long position
                            position = 1
                            entry_info = {
                                'entry_date': date,
                                'entry_zscore': features['zscore'].iloc[i],
                                'entry_p1': p1[date],
                                'entry_p2': p2[date],
                                'confidence': long_probs[i],
                                'position_type': 'Long',
                                'pair': f"{pair['stock1_name']}-{pair['stock2_name']}"
                            }
                        elif short_signals[i] and self.short_selling:  # Enter short position
                            position = -1
                            entry_info = {
                                'entry_date': date,
                                'entry_zscore': features['zscore'].iloc[i],
                                'entry_p1': p1[date],
                                'entry_p2': p2[date],
                                'confidence': short_probs[i],
                                'position_type': 'Short',
                                'pair': f"{pair['stock1_name']}-{pair['stock2_name']}"
                            }
                    
                    elif position == 1:  # In long position, check exit
                        current_zscore = features['zscore'].iloc[i]
                        
                        # Exit conditions for long position
                        exit_signal = (
                            current_zscore > 0 or  # Spread above mean
                            abs(current_zscore) < 0.5  # Close to mean
                        )
                        
                        if exit_signal:
                            # Calculate P&L for long spread position
                            exit_p1, exit_p2 = p1[date], p2[date]
                            hedge_ratio = pair['hedge_ratio']
                            
                            # Long spread: Long P1, Short hedge_ratio*P2
                            pnl = (exit_p1 - entry_info['entry_p1']) - hedge_ratio * (exit_p2 - entry_info['entry_p2'])
                            
                            # Transaction costs
                            trade_value = entry_info['entry_p1'] + hedge_ratio * entry_info['entry_p2']
                            costs = 2 * self.transaction_cost * trade_value
                            net_pnl = pnl - costs
                            
                            trades.append({
                                'pair': entry_info['pair'],
                                'entry_date': entry_info['entry_date'],
                                'exit_date': date,
                                'position': entry_info['position_type'],
                                'entry_zscore': entry_info['entry_zscore'],
                                'exit_zscore': current_zscore,
                                'confidence': entry_info['confidence'],
                                'pnl': net_pnl,
                                'days_held': (date - entry_info['entry_date']).days
                            })
                            
                            position = 0
                            entry_info = {}
                    
                    elif position == -1:  # In short position, check exit
                        current_zscore = features['zscore'].iloc[i]
                        
                        # Exit conditions for short position
                        exit_signal = (
                            current_zscore < 0 or  # Spread below mean
                            abs(current_zscore) < 0.5  # Close to mean
                        )
                        
                        if exit_signal:
                            # Calculate P&L for short spread position
                            exit_p1, exit_p2 = p1[date], p2[date]
                            hedge_ratio = pair['hedge_ratio']
                            
                            # Short spread: Short P1, Long hedge_ratio*P2
                            pnl = (entry_info['entry_p1'] - exit_p1) + hedge_ratio * (exit_p2 - entry_info['entry_p2'])
                            
                            # Transaction costs
                            trade_value = entry_info['entry_p1'] + hedge_ratio * entry_info['entry_p2']
                            costs = 2 * self.transaction_cost * trade_value
                            net_pnl = pnl - costs
                            
                            trades.append({
                                'pair': entry_info['pair'],
                                'entry_date': entry_info['entry_date'],
                                'exit_date': date,
                                'position': entry_info['position_type'],
                                'entry_zscore': entry_info['entry_zscore'],
                                'exit_zscore': current_zscore,
                                'confidence': entry_info['confidence'],
                                'pnl': net_pnl,
                                'days_held': (date - entry_info['entry_date']).days
                            })
                            
                            position = 0
                            entry_info = {}
        
        # Calculate performance for each model
        self.performance_by_model = {}
        for model_name, trades in self.trades_by_model.items():
            if not trades:
                self.performance_by_model[model_name] = None
                continue
            
            # Split trades by position type
            long_trades = [t for t in trades if t['position'] == 'Long']
            short_trades = [t for t in trades if t['position'] == 'Short']
            
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            total_pnl = sum(trade['pnl'] for trade in trades)
            avg_conf = np.mean([trade['confidence'] for trade in trades])
            
            self.performance_by_model[model_name] = {
                'total_trades': total_trades,
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total_trades,
                'avg_confidence': avg_conf
            }
        
        return self
    
    def print_results(self):
        """Print comprehensive results for each model and ensemble"""
        mode = "LONG & SHORT" if self.short_selling else "LONG ONLY"
        print(f"\n" + "="*60)
        print(f"📊 ML-ENHANCED STRATEGY RESULTS BY MODEL ({mode})")
        print("="*60)
        
        for model_name, perf in self.performance_by_model.items():
            if perf is None:
                print(f"\n❌ No trades executed for {model_name}")
                continue
            print(f"\n💰 PERFORMANCE ({model_name.upper()}):")
            print(f"   Total trades: {perf['total_trades']}")
            if self.short_selling:
                print(f"   Long trades: {perf['long_trades']}")
                print(f"   Short trades: {perf['short_trades']}")
            print(f"   Winning trades: {perf['winning_trades']} ({perf['win_rate']*100:.1f}%)")
            print(f"   Total P&L: ${perf['total_pnl']:.2f}")
            print(f"   Average P&L per trade: ${perf['avg_pnl']:.2f}")
            print(f"   Average confidence: {perf['avg_confidence']:.3f}")
            
            # Top pairs
            trades = self.trades_by_model[model_name]
            pair_pnl = {}
            for trade in trades:
                pair = trade['pair']
                if pair not in pair_pnl:
                    pair_pnl[pair] = 0
                pair_pnl[pair] += trade['pnl']
            top_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   🏆 Top pairs:")
            for pair, pnl in top_pairs:
                print(f"      {pair}: ${pnl:.2f}")

def main():
    parser = argparse.ArgumentParser(description='ML-Enhanced Pair Trading Strategy')
    parser.add_argument('data_path', help='Path to data directory')
    parser.add_argument('--max-pairs', type=int, default=30, help='Maximum pairs')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--significance', type=float, default=0.05, help='Cointegration significance')
    parser.add_argument('--short-selling', action='store_true', help='Allow short selling')
    
    args = parser.parse_args()
    
    strategy = MLEnhancedStrategy(
        transaction_cost=args.transaction_cost,
        max_pairs=args.max_pairs,
        short_selling=args.short_selling
    )
    
    strategy.load_data(args.data_path)
    strategy.find_cointegrated_pairs(significance_level=args.significance)
    strategy.train_models()
    strategy.backtest_strategy()
    strategy.print_results()

if __name__ == "__main__":
    main() 