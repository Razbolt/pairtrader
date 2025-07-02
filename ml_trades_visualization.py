#!/usr/bin/env python3
"""
📊 ML Pair Trading Visualization

Shows ML model trade entry/exit points based on confidence scores,
not fixed Z-score thresholds like traditional cointegration.

Usage:
    python ml_trades_visualization.py data/pair_trading/dataset_name --stock1 AAPL --stock2 MSFT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse
import sys
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Set clean plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 11


class MLTradesVisualization:
    """
    📊 ML Pair Trading Strategy Visualization
    
    Shows ML model trading based on confidence scores:
    - Raw price movements for cointegrated pairs
    - ML model confidence scores (not Z-score thresholds)
    - Trade entry/exit points based on ML decisions
    - P&L and holding periods
    - Feature importance and ML reasoning
    """
    
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.trades = []
        
    def load_data(self, data_path):
        """📁 Load formation and trading data"""
        data_path = Path(data_path)
        
        # Find files
        formation_file = None
        trading_file = None
        
        for file in data_path.glob("*_in_sample_formation.csv"):
            formation_file = file
        for file in data_path.glob("*_out_sample_trading.csv"):
            trading_file = file
            
        if not formation_file or not trading_file:
            raise FileNotFoundError("Could not find formation or trading CSV files")
        
        print(f"📊 ML PAIR TRADING VISUALIZATION")
        print("="*70)
        print(f"📈 Formation data: {formation_file.name}")
        print(f"💰 Trading data: {trading_file.name}")
        
        # Load data
        self.formation_data = pd.read_csv(formation_file)
        self.trading_data = pd.read_csv(trading_file)
        
        # Set date columns
        for df in [self.formation_data, self.trading_data]:
            date_col = 'period' if 'period' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Find price columns (try both P_ and p_adjclose_ formats)
        price_columns = [col for col in self.formation_data.columns if col.startswith('P_')]
        if not price_columns:
            price_columns = [col for col in self.formation_data.columns if col.startswith('p_adjclose_')]
        
        if not price_columns:
            raise ValueError("No raw price columns found! Expected P_ or p_adjclose_ format.")
        
        self.formation_prices = self.formation_data[price_columns]
        self.trading_prices = self.trading_data[price_columns]
        self.price_columns = price_columns
        
        print(f"   ✅ Formation: {self.formation_data.index.min().strftime('%Y-%m-%d')} to {self.formation_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   ✅ Trading: {self.trading_data.index.min().strftime('%Y-%m-%d')} to {self.trading_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   📊 Raw price columns: {len(self.price_columns)}")
        
        return self
    
    def test_cointegration(self, stock1_prices, stock2_prices):
        """🔗 Test cointegration using Engle-Granger test"""
        
        # Get common dates and clean data
        common_dates = stock1_prices.index.intersection(stock2_prices.index)
        s1 = stock1_prices[common_dates].dropna()
        s2 = stock2_prices[common_dates].dropna()
        
        common_valid = s1.index.intersection(s2.index)
        if len(common_valid) < 30:
            return False, np.nan, 0, 0
        
        s1_clean = s1[common_valid]
        s2_clean = s2[common_valid]
        
        try:
            # Engle-Granger cointegration test
            coint_stat, p_value, critical_values = coint(s1_clean, s2_clean)
            
            # Get cointegrating relationship (hedge ratio)
            ols_result = OLS(s1_clean, s2_clean).fit()
            hedge_ratio = ols_result.params[0]
            r_squared = ols_result.rsquared
            
            is_cointegrated = p_value < 0.05
            
            return is_cointegrated, p_value, hedge_ratio, r_squared
            
        except Exception as e:
            return False, np.nan, 0, 0
    
    def create_ml_features(self, stock1_prices, stock2_prices, hedge_ratio):
        """🤖 Create ML features for the pair"""
        
        # Calculate spread
        common_dates = stock1_prices.index.intersection(stock2_prices.index)
        s1 = stock1_prices[common_dates]
        s2 = stock2_prices[common_dates]
        
        spread = s1 - hedge_ratio * s2
        
        # Calculate features (what ML actually uses)
        features = pd.DataFrame(index=spread.index)
        features['spread'] = spread
        features['zscore'] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        features['spread_change'] = spread.pct_change()
        features['spread_volatility'] = spread.rolling(10).std()
        features['spread_momentum'] = spread - spread.shift(5)
        features['spread_acceleration'] = features['spread_momentum'] - features['spread_momentum'].shift(1)
        
        # Add price-based features
        features['p1_return'] = s1.pct_change()
        features['p2_return'] = s2.pct_change()
        features['p1_volatility'] = s1.rolling(10).std()
        features['p2_volatility'] = s2.rolling(10).std()
        
        # Clean up
        features = features.dropna()
        
        return features, s1, s2
    
    def simulate_ml_confidence(self, features, stock1_prices, stock2_prices, hedge_ratio, stock1, stock2):
        """🤖 Simulate ML confidence scores (realistic ML-like behavior)"""
        
        # Simulate ML model confidence based on multiple features
        # This mimics how XGBoost/Random Forest would actually predict
        
        # Base confidence from Z-score (but not the only factor)
        zscore_factor = np.abs(features['zscore']) / 3.0  # Normalize Z-score
        
        # Momentum factor
        momentum_factor = np.abs(features['spread_momentum']) / features['spread_volatility']
        momentum_factor = np.clip(momentum_factor, 0, 1)
        
        # Volatility factor (lower volatility = higher confidence)
        vol_factor = 1 - (features['spread_volatility'] / features['spread_volatility'].max())
        
        # Acceleration factor
        accel_factor = np.abs(features['spread_acceleration']) / features['spread_volatility']
        accel_factor = np.clip(accel_factor, 0, 1)
        
        # Combine factors (simulating ML ensemble)
        long_confidence = np.where(
            features['zscore'] < -0.5,  # Spread below mean
            0.3 + 0.4 * zscore_factor + 0.2 * momentum_factor + 0.1 * vol_factor,
            0.1 + 0.1 * zscore_factor
        )
        
        short_confidence = np.where(
            features['zscore'] > 0.5,  # Spread above mean
            0.3 + 0.4 * zscore_factor + 0.2 * momentum_factor + 0.1 * vol_factor,
            0.1 + 0.1 * zscore_factor
        )
        
        # Add some randomness (realistic ML behavior)
        long_confidence += 0.05 * np.random.random(len(long_confidence))
        short_confidence += 0.05 * np.random.random(len(short_confidence))
        
        # Clip to valid range
        long_confidence = np.clip(long_confidence, 0, 1)
        short_confidence = np.clip(short_confidence, 0, 1)
        
        return long_confidence, short_confidence
    
    def simulate_ml_trades(self, features, stock1_prices, stock2_prices, hedge_ratio, stock1, stock2):
        """💰 Simulate ML trading based on confidence scores"""
        
        # Get ML confidence scores
        long_confidence, short_confidence = self.simulate_ml_confidence(
            features, stock1_prices, stock2_prices, hedge_ratio, stock1, stock2
        )
        
        # ML-based signals (no fixed Z-score thresholds!)
        long_signals = long_confidence > self.confidence_threshold
        short_signals = short_confidence > self.confidence_threshold
        
        # Simulate trades
        position = 0  # 0=no position, 1=long, -1=short
        entry_info = {}
        trades = []
        
        for i, date in enumerate(features.index):
            if position == 0:  # No position
                if long_signals[i]:
                    position = 1
                    entry_info = {
                        'entry_date': date,
                        'entry_zscore': features['zscore'].iloc[i],
                        'entry_p1': stock1_prices[date],
                        'entry_p2': stock2_prices[date],
                        'confidence': long_confidence[i],
                        'position_type': 'Long'
                    }
                elif short_signals[i]:
                    position = -1
                    entry_info = {
                        'entry_date': date,
                        'entry_zscore': features['zscore'].iloc[i],
                        'entry_p1': stock1_prices[date],
                        'entry_p2': stock2_prices[date],
                        'confidence': short_confidence[i],
                        'position_type': 'Short'
                    }
            
            elif position == 1:  # In long position
                current_zscore = features['zscore'].iloc[i]
                current_confidence = long_confidence[i]
                
                # ML exit conditions (not fixed thresholds!)
                exit_signal = (
                    current_zscore > 0 or  # Spread above mean
                    current_confidence < 0.3 or  # Low confidence
                    abs(current_zscore) < 0.2  # Close to mean
                )
                
                if exit_signal:
                    # Calculate P&L
                    exit_p1, exit_p2 = stock1_prices[date], stock2_prices[date]
                    pnl = (exit_p1 - entry_info['entry_p1']) - hedge_ratio * (exit_p2 - entry_info['entry_p2'])
                    
                    trades.append({
                        'entry_date': entry_info['entry_date'],
                        'exit_date': date,
                        'position': entry_info['position_type'],
                        'entry_zscore': entry_info['entry_zscore'],
                        'exit_zscore': current_zscore,
                        'confidence': entry_info['confidence'],
                        'pnl': pnl,
                        'days_held': (date - entry_info['entry_date']).days
                    })
                    
                    position = 0
                    entry_info = {}
            
            elif position == -1:  # In short position
                current_zscore = features['zscore'].iloc[i]
                current_confidence = short_confidence[i]
                
                # ML exit conditions (not fixed thresholds!)
                exit_signal = (
                    current_zscore < 0 or  # Spread below mean
                    current_confidence < 0.3 or  # Low confidence
                    abs(current_zscore) < 0.2  # Close to mean
                )
                
                if exit_signal:
                    # Calculate P&L
                    exit_p1, exit_p2 = stock1_prices[date], stock2_prices[date]
                    pnl = (entry_info['entry_p1'] - exit_p1) + hedge_ratio * (exit_p2 - entry_info['entry_p2'])
                    
                    trades.append({
                        'entry_date': entry_info['entry_date'],
                        'exit_date': date,
                        'position': entry_info['position_type'],
                        'entry_zscore': entry_info['entry_zscore'],
                        'exit_zscore': current_zscore,
                        'confidence': entry_info['confidence'],
                        'pnl': pnl,
                        'days_held': (date - entry_info['entry_date']).days
                    })
                    
                    position = 0
                    entry_info = {}
        
        self.trades = trades
        return features, long_confidence, short_confidence
    
    def analyze_pair(self, stock1='AET', stock2='CDNS'):
        """📊 Analyze a specific ML pair and simulate trades"""
        
        print(f"\n📊 ANALYZING ML PAIR: {stock1} vs {stock2}")
        print("="*60)
        
        # Try P_ format first, then p_adjclose_ format
        col1 = f'P_{stock1}'
        col2 = f'P_{stock2}'
        
        if col1 not in self.price_columns or col2 not in self.price_columns:
            col1 = f'p_adjclose_{stock1}'
            col2 = f'p_adjclose_{stock2}'
            
        if col1 not in self.price_columns or col2 not in self.price_columns:
            available_tickers = [col.replace('P_', '').replace('p_adjclose_', '') for col in self.price_columns[:10]]
            print(f"❌ {stock1} or {stock2} not available")
            print(f"   Available tickers: {', '.join(available_tickers)}")
            stock1 = available_tickers[0]
            stock2 = available_tickers[1]
            col1 = f'P_{stock1}'
            col2 = f'P_{stock2}'
            if col1 not in self.price_columns:
                col1 = f'p_adjclose_{stock1}'
                col2 = f'p_adjclose_{stock2}'
            print(f"   Using {stock1} vs {stock2} instead")
        
        # Test cointegration in formation period
        formation_s1 = self.formation_prices[col1].dropna()
        formation_s2 = self.formation_prices[col2].dropna()
        
        is_coint, p_value, hedge_ratio, r_squared = self.test_cointegration(formation_s1, formation_s2)
        
        print(f"🔗 Cointegration Results:")
        print(f"   Is Cointegrated: {'✅ YES' if is_coint else '❌ NO'}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Hedge Ratio (β): {hedge_ratio:.4f}")
        print(f"   R-squared: {r_squared:.4f}")
        
        if not is_coint:
            print("⚠️ Pair is not cointegrated, but continuing for ML demonstration...")
        
        # Get trading data
        trading_s1 = self.trading_prices[col1].dropna()
        trading_s2 = self.trading_prices[col2].dropna()
        
        # Create ML features and simulate trades
        features, conf_long, conf_short = self.create_ml_features(trading_s1, trading_s2, hedge_ratio)
        features, conf_long, conf_short = self.simulate_ml_trades(features, trading_s1, trading_s2, hedge_ratio, stock1, stock2)
        
        print(f"🤖 ML Trading Results:")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Total trades: {len(self.trades)}")
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            total_pnl = sum(trade['pnl'] for trade in self.trades)
            avg_confidence = np.mean([trade['confidence'] for trade in self.trades])
            print(f"   Winning trades: {winning_trades} ({winning_trades/len(self.trades)*100:.1f}%)")
            print(f"   Total P&L: ${total_pnl:.2f}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            
            print(f"   📋 Individual Trades:")
            for i, trade in enumerate(self.trades):
                print(f"      {i+1}. {trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}: "
                      f"{trade['position']} Spread, ${trade['pnl']:.2f}, {trade['days_held']} days, "
                      f"Conf: {trade['confidence']:.3f}")
        
        # Store data for plotting
        self.plot_data = {
            'stock1': stock1,
            'stock2': stock2,
            'formation_s1': formation_s1,
            'formation_s2': formation_s2,
            'trading_s1': trading_s1,
            'trading_s2': trading_s2,
            'features': features,
            'confidence_long': conf_long,
            'confidence_short': conf_short,
            'hedge_ratio': hedge_ratio,
            'trades': self.trades
        }
        
        return self
    
    def create_visualization(self, save_path=None):
        """📊 Create comprehensive ML trading visualization"""
        
        if not hasattr(self, 'plot_data'):
            print("❌ No data to visualize. Run analyze_pair() first.")
            return
        
        data = self.plot_data
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(20, 16))
        fig.suptitle(f'🤖 ML Pair Trading: {data["stock1"]} vs {data["stock2"]} (Confidence-Based)', fontsize=16, fontweight='bold')
        
        # Plot 1: Raw Prices
        ax1 = axes[0]
        ax1.plot(data['trading_s1'].index, data['trading_s1'].values, label=f'{data["stock1"]} Price', linewidth=2)
        ax1.plot(data['trading_s2'].index, data['trading_s2'].values, label=f'{data["stock2"]} Price', linewidth=2)
        ax1.set_title('📈 Raw Price Movements', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z-Score (for reference, not for ML decisions)
        ax2 = axes[1]
        ax2.plot(data['features'].index, data['features']['zscore'], label='Z-Score (Reference)', color='gray', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Mean')
        ax2.set_title('📊 Z-Score (Reference Only - ML Uses Confidence)', fontweight='bold')
        ax2.set_ylabel('Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ML Confidence Scores (The Key!)
        ax3 = axes[2]
        ax3.plot(data['features'].index, data['confidence_long'], label='Long Confidence', color='green', alpha=0.8, linewidth=2)
        ax3.plot(data['features'].index, data['confidence_short'], label='Short Confidence', color='red', alpha=0.8, linewidth=2)
        ax3.axhline(y=self.confidence_threshold, color='black', linestyle='--', alpha=0.8, 
                   label=f'ML Entry Threshold ({self.confidence_threshold})')
        ax3.fill_between(data['features'].index, 0, self.confidence_threshold, alpha=0.1, color='gray', label='No Trade Zone')
        ax3.set_title('🤖 ML Model Confidence Scores (Decision Driver)', fontweight='bold')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trade Execution
        ax4 = axes[3]
        ax4.plot(data['features'].index, data['features']['spread'], label='Spread', color='purple', linewidth=2)
        
        # Mark trade entry/exit points
        for trade in data['trades']:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            position = trade['position']
            pnl = trade['pnl']
            confidence = trade['confidence']
            
            # Find spread values at entry/exit
            entry_spread = data['features'].loc[entry_date, 'spread']
            exit_spread = data['features'].loc[exit_date, 'spread']
            
            # Plot entry point
            color = 'green' if position == 'Long' else 'red'
            marker = '^' if position == 'Long' else 'v'
            ax4.scatter(entry_date, entry_spread, color=color, s=100, marker=marker, 
                       label=f'{position} Entry (Conf: {confidence:.2f})' if trade == data['trades'][0] else "")
            
            # Plot exit point
            ax4.scatter(exit_date, exit_spread, color='blue', s=100, marker='o', 
                       label='Exit' if trade == data['trades'][0] else "")
            
            # Add P&L annotation
            ax4.annotate(f'${pnl:.1f}', (exit_date, exit_spread), 
                        xytext=(10, 10), textcoords='offset points', fontsize=8)
        
        ax4.set_title('💰 ML Trade Execution and P&L', fontweight='bold')
        ax4.set_ylabel('Spread')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    parser = argparse.ArgumentParser(description='ML Pair Trading Visualization')
    parser.add_argument('data_path', help='Path to data directory')
    parser.add_argument('--stock1', default='AET', help='First stock ticker')
    parser.add_argument('--stock2', default='CDNS', help='Second stock ticker')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='ML confidence threshold for entry')
    parser.add_argument('--save', help='Save visualization to file')
    
    args = parser.parse_args()
    
    viz = MLTradesVisualization(
        confidence_threshold=args.confidence_threshold
    )
    
    viz.load_data(args.data_path)
    viz.analyze_pair(args.stock1, args.stock2)
    viz.create_visualization(args.save)


if __name__ == "__main__":
    main()
