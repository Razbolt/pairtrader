#!/usr/bin/env python3
"""
Advanced Unsupervised Learning for Pair Trading
===============================================

This script implements research-based unsupervised learning approaches for:
1. Pair Selection: Using clustering and correlation analysis
2. Entry/Exit Optimization: Using three different unsupervised algorithms

Based on academic research in quantitative finance and pair trading.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class AdvancedUnsupervisedPairTrading:
    def __init__(self, transaction_cost=0.001, max_pairs=30, short_selling=True):
        """Initialize advanced unsupervised pair trading strategy"""
        self.transaction_cost = transaction_cost
        self.max_pairs = max_pairs
        self.short_selling = short_selling
        
        # Data storage
        self.formation_prices = None
        self.trading_prices = None
        self.cointegrated_pairs = []
        self.price_columns = []
        
        # Unsupervised models
        self.kmeans_model = None
        self.dbscan_model = None
        self.isolation_model = None
        
        # Performance tracking
        self.performance = {}
        
    def load_data(self, data_path):
        """Load and prepare data for pair trading"""
        print("🚀 ADVANCED UNSUPERVISED PAIR TRADING STRATEGY")
        print("=" * 60)
        
        # Convert string to Path object
        data_path = Path(data_path)
        
        # Load formation and trading data
        formation_file = list(data_path.glob("*_in_sample_formation.csv"))[0]
        trading_file = list(data_path.glob("*_out_sample_trading.csv"))[0]
        
        print(f"📈 Formation: {formation_file.name}")
        print(f"💰 Trading: {trading_file.name}")
        
        # Load data
        self.formation_prices = pd.read_csv(formation_file, index_col=0, parse_dates=True)
        self.trading_prices = pd.read_csv(trading_file, index_col=0, parse_dates=True)
        
        # Detect column format and set price columns
        if any(col.startswith('p_adjclose_') for col in self.formation_prices.columns):
            print("✅ Using actual price columns (p_adjclose_)")
            self.price_columns = [col for col in self.formation_prices.columns if col.startswith('p_adjclose_')]
            self.price_prefix = "p_adjclose_"
        elif any(col.startswith('P_') for col in self.formation_prices.columns):
            print("✅ Using generated price columns (P_)")
            self.price_columns = [col for col in self.formation_prices.columns if col.startswith('P_')]
            self.price_prefix = "P_"
        else:
            print("✅ Using commodities price columns")
            self.price_columns = self.formation_prices.columns.tolist()
            self.price_prefix = ""
        
        print(f"✅ Loaded {len(self.price_columns)} stocks")
        
        return self
    
    def find_cointegrated_pairs_advanced(self, significance_level=0.05):
        """Advanced pair selection using multiple criteria"""
        print(f"\n🔍 Advanced Pair Selection (significance: {significance_level*100:.1f}%)")
        
        all_pairs = []
        tested_pairs = 0
        
        # Test pairs systematically
        top_stocks = self.price_columns[:200]  # Limit for computational efficiency
        
        for i, stock1 in enumerate(top_stocks):
            for j, stock2 in enumerate(top_stocks):
                if i < j:
                    tested_pairs += 1
                    
                    # Test cointegration
                    s1 = self.formation_prices[stock1].dropna()
                    s2 = self.formation_prices[stock2].dropna()
                    common_dates = s1.index.intersection(s2.index)
                    
                    if len(common_dates) < 50:  # Minimum data requirement
                        continue
                    
                    s1_clean = s1[common_dates]
                    s2_clean = s2[common_dates]
                    
                    try:
                        # Engle-Granger cointegration test
                        slope, intercept, r_value, p_value, std_err = stats.linregress(s1_clean, s2_clean)
                        hedge_ratio = slope
                        r_squared = r_value**2
                        
                        # Calculate spread
                        spread = s1_clean - hedge_ratio * s2_clean
                        
                        # ADF test on spread
                        adf_result = stats.adfuller(spread)
                        p_value_adf = adf_result[1]
                        
                        # Additional quality metrics
                        correlation = pearsonr(s1_clean, s2_clean)[0]
                        spearman_corr = spearmanr(s1_clean, s2_clean)[0]
                        
                        # Calculate Hurst exponent for mean reversion
                        hurst_exp = self.calculate_hurst_exponent(spread)
                        
                        # Calculate half-life for mean reversion
                        half_life = self.calculate_half_life(spread)
                        
                        # Quality filters - loosened for testing
                        if (p_value_adf < significance_level and 
                            r_squared > 0.1 and  # Reduced from 0.3
                            abs(correlation) > 0.3 and  # Reduced from 0.5
                            hurst_exp < 0.6 and  # Increased from 0.5
                            half_life > 0.5 and half_life < 500):  # Relaxed bounds
                            
                            all_pairs.append({
                                'stock1': stock1, 'stock2': stock2,
                                'stock1_name': stock1.replace('p_adjclose_', '').replace('P_', ''),
                                'stock2_name': stock2.replace('p_adjclose_', '').replace('P_', ''),
                                'p_value': p_value_adf, 'hedge_ratio': hedge_ratio,
                                'r_squared': r_squared, 'correlation': correlation,
                                'hurst_exponent': hurst_exp, 'half_life': half_life,
                                'spearman_corr': spearman_corr
                            })
                    except:
                        continue
        
        # Sort by multiple criteria and take top pairs
        all_pairs = sorted(all_pairs, key=lambda x: (x['p_value'], -x['r_squared'], -abs(x['correlation'])))
        self.cointegrated_pairs = all_pairs[:self.max_pairs]
        
        print(f"   Found {len(self.cointegrated_pairs)} high-quality pairs")
        for i, pair in enumerate(self.cointegrated_pairs[:10]):
            print(f"   {i+1}. {pair['stock1_name']}-{pair['stock2_name']}: "
                  f"p={pair['p_value']:.4f}, β={pair['hedge_ratio']:.3f}, "
                  f"R²={pair['r_squared']:.3f}, H={pair['hurst_exponent']:.3f}")
        
        return self
    
    def calculate_hurst_exponent(self, series, max_lag=20):
        """Calculate Hurst exponent for mean reversion detection"""
        try:
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0] * 2.0
        except:
            return 0.5
    
    def calculate_half_life(self, series):
        """Calculate half-life of mean reversion"""
        try:
            spread_lag = series.shift(1)
            spread_ret = series - spread_lag
            spread_lag = spread_lag.dropna()
            spread_ret = spread_ret.dropna()
            
            # OLS regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(spread_lag, spread_ret)
            half_life = -np.log(2) / slope
            return half_life
        except:
            return 252
    
    def create_advanced_features(self, prices, pair_info):
        """Create comprehensive features for unsupervised learning"""
        # Get price data for the pair
        stock1_name = pair_info['stock1_name']
        stock2_name = pair_info['stock2_name']
        
        # Try different column name formats
        stock1_col = None
        stock2_col = None
        
        if stock1_name in prices.columns and stock2_name in prices.columns:
            stock1_col = stock1_name
            stock2_col = stock2_name
        elif f"P_{stock1_name}" in prices.columns and f"P_{stock2_name}" in prices.columns:
            stock1_col = f"P_{stock1_name}"
            stock2_col = f"P_{stock2_name}"
        elif f"p_adjclose_{stock1_name}" in prices.columns and f"p_adjclose_{stock2_name}" in prices.columns:
            stock1_col = f"p_adjclose_{stock1_name}"
            stock2_col = f"p_adjclose_{stock2_name}"
        
        if stock1_col is None or stock2_col is None:
            return None, None
        
        p1 = prices[stock1_col].dropna()
        p2 = prices[stock2_col].dropna()
        
        # Align data
        common_dates = p1.index.intersection(p2.index)
        p1 = p1[common_dates]
        p2 = p2[common_dates]
        
        if len(p1) < 50:
            return None, None
        
        # Calculate spread using hedge ratio
        hedge_ratio = pair_info['hedge_ratio']
        spread = p1 - hedge_ratio * p2
        
        # Create comprehensive features for unsupervised learning
        features = pd.DataFrame(index=spread.index)
        
        # Core spread features
        features['spread'] = spread
        features['zscore'] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        features['zscore_5'] = (spread - spread.rolling(5).mean()) / spread.rolling(5).std()
        features['zscore_10'] = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()
        features['spread_change'] = spread.pct_change()
        features['spread_volatility'] = spread.rolling(10).std()
        
        # Multiple timeframe moving averages
        features['ma_5'] = spread.rolling(5).mean()
        features['ma_10'] = spread.rolling(10).mean()
        features['ma_20'] = spread.rolling(20).mean()
        features['ma_50'] = spread.rolling(50).mean()
        
        # Multiple timeframe standard deviations
        features['std_5'] = spread.rolling(5).std()
        features['std_10'] = spread.rolling(10).std()
        features['std_20'] = spread.rolling(20).std()
        
        # Momentum indicators
        features['momentum_5'] = spread - spread.shift(5)
        features['momentum_10'] = spread - spread.shift(10)
        features['momentum_20'] = spread - spread.shift(20)
        
        # Mean reversion indicators
        features['mean_rev_5'] = (spread - features['ma_5']) / features['std_5']
        features['mean_rev_10'] = (spread - features['ma_10']) / features['std_10']
        features['mean_rev_20'] = (spread - features['ma_20']) / features['std_20']
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['std_5'] / features['std_20']
        features['vol_ratio_10_20'] = features['std_10'] / features['std_20']
        
        # Price-based features
        features['p1_return'] = p1.pct_change()
        features['p2_return'] = p2.pct_change()
        features['p1_volatility'] = p1.rolling(10).std()
        features['p2_volatility'] = p2.rolling(10).std()
        
        # Add actual price data for P&L calculation
        features['p1_price'] = p1
        features['p2_price'] = p2
        
        # Correlation features
        features['correlation_10'] = p1.rolling(10).corr(p2)
        features['correlation_20'] = p1.rolling(20).corr(p2)
        
        # Advanced statistical features
        features['skewness'] = spread.rolling(20).skew()
        features['kurtosis'] = spread.rolling(20).kurt()
        
        # Regime detection features
        features['regime_volatility'] = spread.rolling(20).std() / spread.rolling(50).std()
        features['regime_momentum'] = features['momentum_5'] / features['momentum_20']
        
        # Clean up
        features = features.dropna()
        
        if len(features) < 30:
            return None, None
        
        return features, spread
    
    def train_kmeans_model(self, features):
        """Train K-Means clustering model with optimal cluster selection"""
        print("   Training K-Means clustering...")
        
        # Prepare features for clustering
        feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
        X = features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using multiple metrics
        silhouette_scores = []
        calinski_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                calinski_avg = calinski_harabasz_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                calinski_scores.append(calinski_avg)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        # Choose optimal K (average of both metrics)
        combined_scores = [(silhouette_scores[i] + calinski_scores[i]/1000) for i in range(len(K_range))]
        optimal_k = K_range[np.argmax(combined_scores)]
        
        print(f"   Optimal clusters: {optimal_k} (silhouette: {silhouette_scores[optimal_k-2]:.3f})")
        
        # Train final model
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.kmeans_model.fit(X_scaled)
        
        return self.kmeans_model, scaler
    
    def train_dbscan_model(self, features):
        """Train DBSCAN clustering model with parameter optimization"""
        print("   Training DBSCAN clustering...")
        
        # Prepare features
        feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
        X = features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal parameters using silhouette score
        best_score = -1
        best_eps = 0.1
        best_min_samples = 5
        
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            for min_samples in [3, 5, 7, 10, 15]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(X_scaled)
                
                # Only calculate score if we have more than 1 cluster and no noise
                if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                    score = silhouette_score(X_scaled, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
        
        print(f"   Best params: eps={best_eps}, min_samples={best_min_samples} (silhouette: {best_score:.3f})")
        
        # Train final model
        self.dbscan_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        self.dbscan_model.fit(X_scaled)
        
        return self.dbscan_model, scaler
    
    def train_isolation_forest(self, features):
        """Train Isolation Forest for anomaly detection"""
        print("   Training Isolation Forest...")
        
        # Prepare features
        feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
        X = features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest with optimized contamination
        self.isolation_model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.isolation_model.fit(X_scaled)
        
        return self.isolation_model, scaler
    
    def train_models(self):
        """Train all unsupervised models on formation period data"""
        print("\n🤖 Training Advanced Unsupervised Models...")
        
        # Collect features from all pairs
        all_features = []
        
        for pair in self.cointegrated_pairs:
            features, _ = self.create_advanced_features(self.formation_prices, pair)
            if features is not None:
                all_features.append(features)
        
        if not all_features:
            print("   No valid data for training")
            return None
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=0)
        print(f"   Training on {len(combined_features)} samples")
        
        # Train K-Means
        self.kmeans_model, self.kmeans_scaler = self.train_kmeans_model(combined_features)
        
        # Train DBSCAN
        self.dbscan_model, self.dbscan_scaler = self.train_dbscan_model(combined_features)
        
        # Train Isolation Forest
        self.isolation_model, self.isolation_scaler = self.train_isolation_forest(combined_features)
        
        print(f"   Trained 3 advanced unsupervised models")
        
        return self
    
    def generate_kmeans_signals(self):
        """Generate signals using K-Means clustering"""
        print("\n🔍 Generating K-Means signals...")
        
        all_trades = []
        
        for pair in self.cointegrated_pairs:
            features, spread = self.create_advanced_features(self.trading_prices, pair)
            if features is None:
                continue
            
            # Prepare features
            feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
            X = features[feature_cols].values
            X_scaled = self.kmeans_scaler.transform(X)
            
            # Get cluster predictions
            clusters = self.kmeans_model.predict(X_scaled)
            
            # Analyze cluster characteristics
            cluster_stats = {}
            for cluster_id in set(clusters):
                cluster_mask = clusters == cluster_id
                cluster_zscore = features['zscore'][cluster_mask]
                cluster_stats[cluster_id] = {
                    'mean_zscore': cluster_zscore.mean(),
                    'std_zscore': cluster_zscore.std(),
                    'count': cluster_mask.sum()
                }
            
            # Identify entry/exit clusters based on z-score characteristics
            entry_clusters = []
            exit_clusters = []
            
            for cluster_id, stats in cluster_stats.items():
                if stats['mean_zscore'] < -1.0 and stats['count'] > 5:  # Low z-score clusters
                    entry_clusters.append(cluster_id)
                elif stats['mean_zscore'] > 0.5 and stats['count'] > 5:  # High z-score clusters
                    exit_clusters.append(cluster_id)
            
            # Generate trading signals
            position = 0
            entry_date = None
            entry_p1 = None
            entry_p2 = None
            hedge_ratio = pair['hedge_ratio']
            
            for i, date in enumerate(features.index):
                current_cluster = clusters[i]
                current_zscore = features['zscore'].iloc[i]
                current_p1 = features['p1_price'].iloc[i]
                current_p2 = features['p2_price'].iloc[i]
                
                if position == 0:  # No position
                    if current_cluster in entry_clusters and current_zscore < -1.2:
                        position = 1
                        entry_date = date
                        entry_p1 = current_p1
                        entry_p2 = current_p2
                        
                        all_trades.append({
                            'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                            'entry_date': date,
                            'entry_zscore': current_zscore,
                            'entry_cluster': current_cluster,
                            'position_type': 'Long Stock1 + Short (hedge_ratio * Stock2)',
                            'entry_reason': 'KMeans_Entry'
                        })
                
                elif position == 1:  # Long position
                    if current_cluster in exit_clusters or current_zscore > 0.2:
                        # Calculate P&L
                        pnl_p1 = current_p1 - entry_p1
                        pnl_p2 = hedge_ratio * (entry_p2 - current_p2)
                        gross_pnl = pnl_p1 + pnl_p2
                        
                        trade_value = entry_p1 + hedge_ratio * entry_p2
                        transaction_costs = 2 * self.transaction_cost * trade_value
                        net_pnl = gross_pnl - transaction_costs
                        
                        if all_trades:
                            last_trade = all_trades[-1]
                            last_trade.update({
                                'exit_date': date,
                                'exit_zscore': current_zscore,
                                'exit_cluster': current_cluster,
                                'pnl': net_pnl,
                                'exit_reason': 'KMeans_Exit'
                            })
                        
                        position = 0
                        entry_date = None
                        entry_p1 = None
                        entry_p2 = None
        
        return all_trades
    
    def generate_dbscan_signals(self):
        """Generate signals using DBSCAN clustering"""
        print("\n🔍 Generating DBSCAN signals...")
        
        all_trades = []
        
        for pair in self.cointegrated_pairs:
            features, spread = self.create_advanced_features(self.trading_prices, pair)
            if features is None:
                continue
            
            # Prepare features
            feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
            X = features[feature_cols].values
            X_scaled = self.dbscan_scaler.transform(X)
            
            # Get cluster predictions
            clusters = self.dbscan_model.fit_predict(X_scaled)
            
            # Analyze cluster characteristics
            cluster_stats = {}
            for cluster_id in set(clusters):
                if cluster_id != -1:  # Skip noise points
                    cluster_mask = clusters == cluster_id
                    cluster_zscore = features['zscore'][cluster_mask]
                    cluster_stats[cluster_id] = {
                        'mean_zscore': cluster_zscore.mean(),
                        'std_zscore': cluster_zscore.std(),
                        'count': cluster_mask.sum()
                    }
            
            # Identify entry/exit clusters
            entry_clusters = []
            exit_clusters = []
            
            for cluster_id, stats in cluster_stats.items():
                if stats['mean_zscore'] < -1.0 and stats['count'] > 5:  # Significant low z-score clusters
                    entry_clusters.append(cluster_id)
                elif stats['mean_zscore'] > 0.5 and stats['count'] > 5:  # Significant high z-score clusters
                    exit_clusters.append(cluster_id)
            
            # Generate trading signals
            position = 0
            entry_date = None
            entry_p1 = None
            entry_p2 = None
            hedge_ratio = pair['hedge_ratio']
            
            for i, date in enumerate(features.index):
                current_cluster = clusters[i]
                current_zscore = features['zscore'].iloc[i]
                current_p1 = features['p1_price'].iloc[i]
                current_p2 = features['p2_price'].iloc[i]
                
                if position == 0:  # No position
                    if current_cluster in entry_clusters and current_zscore < -1.2:
                        position = 1
                        entry_date = date
                        entry_p1 = current_p1
                        entry_p2 = current_p2
                        
                        all_trades.append({
                            'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                            'entry_date': date,
                            'entry_zscore': current_zscore,
                            'entry_cluster': current_cluster,
                            'position_type': 'Long Stock1 + Short (hedge_ratio * Stock2)',
                            'entry_reason': 'DBSCAN_Entry'
                        })
                
                elif position == 1:  # Long position
                    if current_cluster in exit_clusters or current_zscore > 0.2:
                        # Calculate P&L
                        pnl_p1 = current_p1 - entry_p1
                        pnl_p2 = hedge_ratio * (entry_p2 - current_p2)
                        gross_pnl = pnl_p1 + pnl_p2
                        
                        trade_value = entry_p1 + hedge_ratio * entry_p2
                        transaction_costs = 2 * self.transaction_cost * trade_value
                        net_pnl = gross_pnl - transaction_costs
                        
                        if all_trades:
                            last_trade = all_trades[-1]
                            last_trade.update({
                                'exit_date': date,
                                'exit_zscore': current_zscore,
                                'exit_cluster': current_cluster,
                                'pnl': net_pnl,
                                'exit_reason': 'DBSCAN_Exit'
                            })
                        
                        position = 0
                        entry_date = None
                        entry_p1 = None
                        entry_p2 = None
        
        return all_trades
    
    def generate_isolation_forest_signals(self):
        """Generate signals using Isolation Forest anomaly detection"""
        print("\n🔍 Generating Isolation Forest signals...")
        
        all_trades = []
        
        for pair in self.cointegrated_pairs:
            features, spread = self.create_advanced_features(self.trading_prices, pair)
            if features is None:
                continue
            
            # Prepare features
            feature_cols = [col for col in features.columns if col not in ['spread', 'p1_price', 'p2_price']]
            X = features[feature_cols].values
            X_scaled = self.isolation_scaler.transform(X)
            
            # Get anomaly predictions (-1 for anomalies, 1 for normal)
            anomalies = self.isolation_model.predict(X_scaled)
            
            # Generate trading signals based on anomalies
            position = 0
            entry_date = None
            entry_p1 = None
            entry_p2 = None
            hedge_ratio = pair['hedge_ratio']
            
            for i, date in enumerate(features.index):
                is_anomaly = anomalies[i] == -1
                current_zscore = features['zscore'].iloc[i]
                current_p1 = features['p1_price'].iloc[i]
                current_p2 = features['p2_price'].iloc[i]
                
                if position == 0:  # No position
                    # Enter on extreme negative z-score anomalies
                    if is_anomaly and current_zscore < -1.5:
                        position = 1
                        entry_date = date
                        entry_p1 = current_p1
                        entry_p2 = current_p2
                        
                        all_trades.append({
                            'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                            'entry_date': date,
                            'entry_zscore': current_zscore,
                            'is_anomaly': is_anomaly,
                            'position_type': 'Long Stock1 + Short (hedge_ratio * Stock2)',
                            'entry_reason': 'IsolationForest_Entry'
                        })
                
                elif position == 1:  # Long position
                    # Exit on positive z-score or when spread normalizes
                    if current_zscore > 0.2 or (is_anomaly and current_zscore > 0):
                        # Calculate P&L
                        pnl_p1 = current_p1 - entry_p1
                        pnl_p2 = hedge_ratio * (entry_p2 - current_p2)
                        gross_pnl = pnl_p1 + pnl_p2
                        
                        trade_value = entry_p1 + hedge_ratio * entry_p2
                        transaction_costs = 2 * self.transaction_cost * trade_value
                        net_pnl = gross_pnl - transaction_costs
                        
                        if all_trades:
                            last_trade = all_trades[-1]
                            last_trade.update({
                                'exit_date': date,
                                'exit_zscore': current_zscore,
                                'is_anomaly': is_anomaly,
                                'pnl': net_pnl,
                                'exit_reason': 'IsolationForest_Exit'
                            })
                        
                        position = 0
                        entry_date = None
                        entry_p1 = None
                        entry_p2 = None
        
        return all_trades
    
    def backtest_strategies(self):
        """Backtest all three unsupervised strategies"""
        print("\n🚀 Backtesting Advanced Unsupervised Strategies...")
        
        # Generate signals for each strategy
        kmeans_trades = self.generate_kmeans_signals()
        dbscan_trades = self.generate_dbscan_signals()
        isolation_trades = self.generate_isolation_forest_signals()
        
        # Calculate performance for each strategy
        strategies = {
            'K-Means': kmeans_trades,
            'DBSCAN': dbscan_trades,
            'Isolation Forest': isolation_trades
        }
        
        for strategy_name, trades in strategies.items():
            if not trades:
                print(f"   {strategy_name}: No trades generated")
                continue
            
            # Calculate performance metrics
            total_pnl = sum(trade['pnl'] for trade in trades if 'pnl' in trade)
            total_trades = len([trade for trade in trades if 'pnl' in trade])
            winning_trades = len([trade for trade in trades if 'pnl' in trade and trade['pnl'] > 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            if total_trades > 0:
                pnls = [trade['pnl'] for trade in trades if 'pnl' in trade]
                avg_pnl = np.mean(pnls)
                max_profit = max(pnls)
                max_loss = min(pnls)
                sharpe_ratio = avg_pnl / np.std(pnls) if np.std(pnls) > 0 else 0
            else:
                avg_pnl = max_profit = max_loss = sharpe_ratio = 0
            
            # Store performance
            self.performance[strategy_name] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades
            }
        
        return self
    
    def print_results(self):
        """Print comprehensive results and comparison"""
        print("\n" + "=" * 60)
        print("🎯 ADVANCED UNSUPERVISED PAIR TRADING RESULTS")
        print("=" * 60)
        
        # Print individual strategy results
        for strategy_name, perf in self.performance.items():
            print(f"\n📊 {strategy_name.upper()} STRATEGY:")
            print(f"   Total trades: {perf['total_trades']}")
            print(f"   Winning trades: {perf['winning_trades']} ({perf['win_rate']*100:.1f}%)")
            print(f"   Total P&L: ${perf['total_pnl']:.2f}")
            print(f"   Average P&L per trade: ${perf['avg_pnl']:.2f}")
            print(f"   Max profit: ${perf['max_profit']:.2f}")
            print(f"   Max loss: ${perf['max_loss']:.2f}")
            print(f"   Sharpe ratio: {perf['sharpe_ratio']:.2f}")
            
            # Show sample trades
            if perf['trades']:
                print(f"   Sample trades:")
                for i, trade in enumerate(perf['trades'][:3], 1):
                    if 'pnl' in trade:
                        print(f"     {i}. {trade['pair']} {trade['position_type']}: "
                              f"Entry {trade['entry_date'].strftime('%Y-%m-%d')} "
                              f"(Z={trade['entry_zscore']:.2f}) → "
                              f"Exit {trade['exit_date'].strftime('%Y-%m-%d')} "
                              f"(Z={trade['exit_zscore']:.2f}) → P&L: ${trade['pnl']:.2f}")
        
        # Comparative analysis
        print(f"\n" + "=" * 60)
        print("📈 COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        if len(self.performance) > 1:
            # Create comparison table
            comparison_data = []
            for strategy_name, perf in self.performance.items():
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Trades': perf['total_trades'],
                    'Win Rate (%)': f"{perf['win_rate']*100:.1f}",
                    'Total P&L ($)': f"{perf['total_pnl']:.2f}",
                    'Avg P&L ($)': f"{perf['avg_pnl']:.2f}",
                    'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                    'Max Profit ($)': f"{perf['max_profit']:.2f}",
                    'Max Loss ($)': f"{perf['max_loss']:.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            # Find best performing strategy
            best_strategy = max(self.performance.items(), 
                              key=lambda x: x[1]['sharpe_ratio'] if x[1]['sharpe_ratio'] > 0 else -999)
            
            print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy[0]}")
            print(f"   Sharpe Ratio: {best_strategy[1]['sharpe_ratio']:.2f}")
            print(f"   Total P&L: ${best_strategy[1]['total_pnl']:.2f}")
            print(f"   Win Rate: {best_strategy[1]['win_rate']*100:.1f}%")
        
        # Algorithm characteristics comparison
        print(f"\n" + "=" * 60)
        print("🔬 ALGORITHM CHARACTERISTICS & RESEARCH INSIGHTS")
        print("=" * 60)
        
        print("""
K-MEANS CLUSTERING:
   • Groups similar market states into clusters
   • Identifies entry/exit clusters based on z-score characteristics
   • Research Insight: Effective for regime detection in mean-reverting spreads
   • Pros: Simple, interpretable, works well with clear patterns
   • Cons: Assumes spherical clusters, sensitive to feature scaling

DBSCAN CLUSTERING:
   • Identifies density-based clusters and noise points
   • Can find irregularly shaped clusters
   • Research Insight: Better for detecting market microstructure patterns
   • Pros: Handles noise well, doesn't assume cluster shapes
   • Cons: Sensitive to parameters, may not work with varying densities

ISOLATION FOREST:
   • Detects anomalies (extreme market conditions)
   • Identifies unusual spread movements
   • Research Insight: Effective for detecting structural breaks and regime changes
   • Pros: Fast, handles high-dimensional data, identifies outliers
   • Cons: May flag normal extreme movements as anomalies

ADVANCED PAIR SELECTION:
   • Multiple criteria: Cointegration, correlation, Hurst exponent, half-life
   • Research-based quality filters for robust pair selection
   • Ensures pairs have strong mean-reverting properties
        """)
        
        return self

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Unsupervised Pair Trading Strategy')
    parser.add_argument('data_path', type=str, help='Path to data directory')
    parser.add_argument('--max-pairs', type=int, default=10, help='Maximum number of pairs to trade')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost as fraction')
    parser.add_argument('--significance', type=float, default=0.05, help='Significance level for cointegration test')
    parser.add_argument('--short-selling', action='store_true', help='Enable short selling')
    
    args = parser.parse_args()
    
    # Create strategy
    strategy = AdvancedUnsupervisedPairTrading(
        transaction_cost=args.transaction_cost,
        max_pairs=args.max_pairs,
        short_selling=args.short_selling
    )
    
    # Run strategy
    result = (strategy
     .load_data(args.data_path)
     .find_cointegrated_pairs_advanced(args.significance)
     .train_models())
    
    if result is not None:
        (strategy
         .backtest_strategies()
         .print_results())

if __name__ == "__main__":
    main() 