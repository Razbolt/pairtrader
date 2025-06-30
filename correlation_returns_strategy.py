#!/usr/bin/env python3
"""
📊 Correlation-Based Pair Trading Strategy

This strategy uses high correlation to identify trading pairs:
1. Find pairs with correlation > threshold
2. Calculate standardized spread (Z-score)
3. Trade on mean reversion signals
4. Analyze performance with statistical tests

Key Features:
- Clean visualizations without overlapping elements
- Statistical significance testing
- Performance metrics and trade analysis
- Easy comparison with cointegration approach

Usage:
    python correlation_strategy.py ../../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from scipy import stats
from scipy.stats import binomtest
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set clean plotting style
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CorrelationPairTradingStrategy:
    """
    📊 Correlation-Based Pair Trading Strategy
    
    Strategy Logic:
    1. Formation Period: Find high correlation pairs
    2. Trading Period: Trade on Z-score signals
    3. Entry: |Z-score| > entry_threshold
    4. Exit: |Z-score| < exit_threshold
    """
    
    def __init__(self, correlation_threshold=0.80, entry_zscore=2.0, exit_zscore=0.5):
        self.correlation_threshold = correlation_threshold
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.pairs = []
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
        
        print(f"📊 CORRELATION-BASED PAIR TRADING STRATEGY")
        print("="*60)
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
        
        # Get stock columns
        self.stock_columns = [col for col in self.formation_data.columns if col.startswith('R_')]
        
        print(f"   ✅ Formation: {self.formation_data.index.min().strftime('%Y-%m-%d')} to {self.formation_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   ✅ Trading: {self.trading_data.index.min().strftime('%Y-%m-%d')} to {self.trading_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   📊 Stocks: {len(self.stock_columns)}")
        
        return self
    
    def find_correlated_pairs(self, max_pairs=20):
        """🔍 Find highly correlated pairs"""
        print(f"\n🔍 Finding correlated pairs (threshold: {self.correlation_threshold})")
        
        # Calculate correlation matrix
        corr_matrix = self.formation_data[self.stock_columns].corr()
        
        # Find pairs above threshold
        pairs = []
        for i, stock1 in enumerate(self.stock_columns):
            for j, stock2 in enumerate(self.stock_columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.loc[stock1, stock2]
                    if abs(correlation) > self.correlation_threshold:
                        pairs.append({
                            'stock1': stock1,
                            'stock2': stock2,
                            'stock1_name': stock1.replace('R_', ''),
                            'stock2_name': stock2.replace('R_', ''),
                            'correlation': correlation
                        })
        
        # Sort by correlation strength
        pairs = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)
        
        # Limit pairs
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        
        self.pairs = pairs
        
        print(f"   📊 Found {len(pairs)} pairs above threshold")
        
        if pairs:
            print(f"\n   🏆 Top 10 correlated pairs:")
            for i, pair in enumerate(pairs[:10]):
                print(f"      {i+1:2d}. {pair['stock1_name']} - {pair['stock2_name']}: {pair['correlation']:.4f}")
        
        return self
    
    def backtest_strategy(self):
        """💰 Backtest the correlation strategy"""
        print(f"\n💰 Backtesting strategy...")
        print(f"   Entry Z-score: ±{self.entry_zscore}")
        print(f"   Exit Z-score: ±{self.exit_zscore}")
        
        all_trades = []
        
        for pair in self.pairs:
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            
            # Get formation data for this pair
            formation_s1 = self.formation_data[stock1].dropna()
            formation_s2 = self.formation_data[stock2].dropna()
            
            # Calculate formation spread
            common_formation = formation_s1.index.intersection(formation_s2.index)
            if len(common_formation) < 50:
                continue
                
            formation_spread = formation_s1[common_formation] - formation_s2[common_formation]
            spread_mean = formation_spread.mean()
            spread_std = formation_spread.std()
            
            if spread_std == 0:
                continue
            
            # Get trading data for this pair
            trading_s1 = self.trading_data[stock1].dropna()
            trading_s2 = self.trading_data[stock2].dropna()
            
            common_trading = trading_s1.index.intersection(trading_s2.index)
            if len(common_trading) < 10:
                continue
            
            trading_spread = trading_s1[common_trading] - trading_s2[common_trading]
            trading_zscore = (trading_spread - spread_mean) / spread_std
            
            # Execute trades
            pair_trades = self._execute_trades(trading_zscore, pair)
            all_trades.extend(pair_trades)
        
        self.trades = all_trades
        
        # Calculate performance metrics
        self._calculate_performance()
        
        return self
    
    def _execute_trades(self, zscore_series, pair):
        """🎯 Execute trades based on Z-score signals"""
        trades = []
        position = 0  # 0=no position, 1=long spread, -1=short spread
        entry_price = 0
        entry_date = None
        
        for date, zscore in zscore_series.items():
            if position == 0:  # No position
                if zscore > self.entry_zscore:  # Short spread (short stock1, long stock2)
                    position = -1
                    entry_price = zscore
                    entry_date = date
                elif zscore < -self.entry_zscore:  # Long spread (long stock1, short stock2)
                    position = 1
                    entry_price = zscore
                    entry_date = date
            
            else:  # In position
                exit_signal = False
                
                if position == 1 and zscore > -self.exit_zscore:  # Exit long
                    exit_signal = True
                elif position == -1 and zscore < self.exit_zscore:  # Exit short
                    exit_signal = True
                
                if exit_signal:
                    # Calculate PnL
                    pnl = position * (entry_price - zscore)  # Positive = profit
                    
                    trades.append({
                        'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_zscore': entry_price,
                        'exit_zscore': zscore,
                        'position': 'Long' if position == 1 else 'Short',
                        'pnl': pnl,
                        'days_held': (date - entry_date).days
                    })
                    
                    position = 0
        
        return trades
    
    def _calculate_performance(self):
        """📊 Calculate performance metrics"""
        if not self.trades:
            print("   ❌ No trades executed")
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_pnl = total_pnl / total_trades
        win_rate = winning_trades / total_trades
        
        pnls = [t['pnl'] for t in self.trades]
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
        
        # Statistical significance
        binom_result = binomtest(winning_trades, total_trades, 0.5)
        t_stat, t_p_value = stats.ttest_1samp(pnls, 0)
        
        print(f"\n📊 PERFORMANCE RESULTS")
        print("-" * 40)
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Losing Trades: {losing_trades}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total PnL: {total_pnl:.4f}")
        print(f"   Average PnL: {avg_pnl:.4f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        
        print(f"\n🧪 STATISTICAL SIGNIFICANCE")
        print("-" * 40)
        print(f"   Win Rate Test: p={binom_result.pvalue:.6f}")
        print(f"   PnL t-test: p={t_p_value:.6f}")
        print(f"   Significant: {'✅ Yes' if t_p_value < 0.05 else '❌ No'}")
        
        self.performance = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'binom_p_value': binom_result.pvalue,
            't_p_value': t_p_value
        }
    
    def plot_results(self, save_path=None):
        """📊 Create comprehensive results visualization"""
        if not self.trades:
            print("❌ No trades to plot")
            return
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                             hspace=0.3, wspace=0.3, top=0.93, bottom=0.08, left=0.08, right=0.95)
        
        fig.suptitle('📊 Correlation-Based Pair Trading Strategy Results', 
                     fontsize=16, fontweight='bold', y=0.97)
        
        # Plot 1: PnL Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        pnls = [t['pnl'] for t in self.trades]
        ax1.hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax1.set_title('PnL Distribution', fontweight='bold', pad=15)
        ax1.set_xlabel('PnL per Trade')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative PnL
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_pnl = np.cumsum(pnls)
        trade_numbers = range(1, len(pnls) + 1)
        ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color='green', marker='o', markersize=3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Cumulative PnL', fontweight='bold', pad=15)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative PnL')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win Rate by Pair
        ax3 = fig.add_subplot(gs[1, 0])
        pair_stats = {}
        for trade in self.trades:
            pair = trade['pair']
            if pair not in pair_stats:
                pair_stats[pair] = {'wins': 0, 'total': 0}
            pair_stats[pair]['total'] += 1
            if trade['pnl'] > 0:
                pair_stats[pair]['wins'] += 1
        
        pairs = list(pair_stats.keys())[:10]  # Top 10 pairs
        win_rates = [(pair_stats[pair]['wins'] / pair_stats[pair]['total']) * 100 for pair in pairs]
        
        bars = ax3.bar(range(len(pairs)), win_rates, color='lightcoral', alpha=0.7)
        ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% (Random)')
        ax3.set_title('Win Rate by Pair (Top 10)', fontweight='bold', pad=15)
        ax3.set_xlabel('Pair')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_xticks(range(len(pairs)))
        ax3.set_xticklabels([pair.replace('-', '\\n') for pair in pairs], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Holding Period Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        holding_periods = [t['days_held'] for t in self.trades]
        ax4.hist(holding_periods, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Holding Period Distribution', fontweight='bold', pad=15)
        ax4.set_xlabel('Days Held')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Performance Summary Table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create performance summary
        perf = self.performance
        summary_text = f"""
📊 CORRELATION STRATEGY PERFORMANCE SUMMARY

🎯 TRADING METRICS:
   • Total Trades: {perf['total_trades']}
   • Win Rate: {perf['win_rate']:.1%}
   • Total PnL: {perf['total_pnl']:.4f}
   • Average PnL per Trade: {perf['avg_pnl']:.4f}
   • Sharpe Ratio: {perf['sharpe_ratio']:.4f}
   • Average Holding Period: {np.mean(holding_periods):.1f} days

🧪 STATISTICAL SIGNIFICANCE:
   • Win Rate Test (vs 50%): p-value = {perf['binom_p_value']:.6f}
   • PnL Significance Test: p-value = {perf['t_p_value']:.6f}
   • Result: {'✅ Statistically Significant' if perf['t_p_value'] < 0.05 else '❌ Not Statistically Significant'}

📈 STRATEGY PARAMETERS:
   • Correlation Threshold: {self.correlation_threshold}
   • Entry Z-Score: ±{self.entry_zscore}
   • Exit Z-Score: ±{self.exit_zscore}
   • Pairs Analyzed: {len(self.pairs)}
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Plot saved to: {save_path}")
        
        plt.show()
        return self


def main():
    """🚀 Main function"""
    parser = argparse.ArgumentParser(
        description='Correlation-Based Pair Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📊 CORRELATION STRATEGY EXPLANATION:

🎯 STRATEGY LOGIC:
   1. Find pairs with high correlation (> threshold)
   2. Calculate standardized spread (Z-score)
   3. Enter trades when |Z-score| > entry_threshold
   4. Exit trades when |Z-score| < exit_threshold

📈 ASSUMPTIONS:
   • High correlation implies mean-reverting spread
   • Temporary deviations will revert to mean
   • Correlation is stable over time

🧪 STATISTICAL TESTS:
   • Win rate significance (binomial test)
   • PnL significance (t-test)
   • Performance vs random trading

💡 USAGE EXAMPLES:
   python correlation_strategy.py ../../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m
   python correlation_strategy.py ../../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m --correlation 0.85 --entry-zscore 1.5
        """
    )
    
    parser.add_argument('data_path', help='Path to pair trading data directory')
    parser.add_argument('--correlation', '-c', type=float, default=0.80,
                       help='Correlation threshold (default: 0.80)')
    parser.add_argument('--entry-zscore', '-e', type=float, default=2.0,
                       help='Entry Z-score threshold (default: 2.0)')
    parser.add_argument('--exit-zscore', '-x', type=float, default=0.5,
                       help='Exit Z-score threshold (default: 0.5)')
    parser.add_argument('--max-pairs', '-m', type=int, default=20,
                       help='Maximum pairs to analyze (default: 20)')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate result plots')
    
    args = parser.parse_args()
    
    try:
        strategy = CorrelationPairTradingStrategy(
            correlation_threshold=args.correlation,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore
        )
        
        strategy.load_data(args.data_path)
        strategy.find_correlated_pairs(max_pairs=args.max_pairs)
        strategy.backtest_strategy()
        
        if args.plot:
            plot_path = Path(args.data_path) / "correlation_strategy_results.png"
            strategy.plot_results(save_path=plot_path)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 