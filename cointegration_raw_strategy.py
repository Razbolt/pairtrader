#!/usr/bin/env python3
"""
🔗 Raw Prices Cointegration Strategy - Top 20 Pairs Analysis

This strategy uses actual raw prices to find and backtest cointegrated pairs:
1. Load actual raw price data (p_adjclose_ or P_ columns)
2. Find top 20 cointegrated pairs using Engle-Granger tests
3. Backtest strategy on all pairs simultaneously
4. Analyze performance with detailed statistics

Usage:
    python raw_prices_strategy.py data/pair_trading/sp500_20230101_20240705_prices_12m6m
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
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Set clean plotting style
plt.style.use('default')
sns.set_palette("Set1")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class RawPricesCointegrationStrategy:
    """
    🔗 Raw Prices Cointegration Strategy
    
    Uses actual price levels for true cointegration testing and trading.
    Finds multiple pairs and backtests them simultaneously.
    """
    
    def __init__(self, significance_level=0.05, entry_zscore=1.0, exit_zscore=0.0, transaction_cost=0.001):
        self.significance_level = significance_level
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.transaction_cost = transaction_cost
        self.cointegrated_pairs = []
        self.trades = []
        self.performance_metrics = {}
        
    def load_data(self, data_path):
        """📁 Load formation and trading data with raw prices"""
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
        
        print(f"🔗 RAW PRICES COINTEGRATION STRATEGY - TOP 20 PAIRS")
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
        
        # Find raw price columns (try both formats)
        price_columns = [col for col in self.formation_data.columns if col.startswith('p_adjclose_')]
        
        if not price_columns:
            price_columns = [col for col in self.formation_data.columns if col.startswith('P_')]
            if price_columns:
                print("✅ Using generated raw price columns (P_) from log returns conversion")
                self.price_col_format = 'P_'
            else:
                print("❌ No raw price columns found!")
                raise ValueError("No raw price columns found")
        else:
            print("✅ Using actual raw price columns (p_adjclose_)")
            self.price_col_format = 'p_adjclose_'
        
        # Use raw prices directly (no cumsum needed!)
        self.formation_prices = self.formation_data[price_columns]
        self.trading_prices = self.trading_data[price_columns]
        
        self.price_columns = price_columns
        
        print(f"   ✅ Formation: {self.formation_data.index.min().strftime('%Y-%m-%d')} to {self.formation_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   ✅ Trading: {self.trading_data.index.min().strftime('%Y-%m-%d')} to {self.trading_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   📊 Raw price columns: {len(self.price_columns)}")
        print(f"   💡 Using actual price levels for true cointegration testing")
        
        return self
    
    def test_cointegration(self, stock1_prices, stock2_prices):
        """🔗 Test cointegration using Engle-Granger test on raw prices"""
        
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
            
            is_cointegrated = p_value < self.significance_level
            
            return is_cointegrated, p_value, hedge_ratio, r_squared
            
        except Exception as e:
            return False, np.nan, 0, 0
    
    def find_cointegrated_pairs(self, max_pairs=20, min_stocks=50):
        """🔍 Find top cointegrated pairs using raw prices"""
        print(f"\n🔍 Finding top {max_pairs} cointegrated pairs (significance: {self.significance_level*100}%)")
        
        # Use a subset of stocks for efficiency (top by market cap/volume)
        available_stocks = self.price_columns[:min_stocks] if len(self.price_columns) > min_stocks else self.price_columns
        
        print(f"   Testing {len(available_stocks)} stocks...")
        print(f"   Total possible pairs: {len(available_stocks) * (len(available_stocks) - 1) // 2}")
        
        all_pairs = []
        tested_pairs = 0
        
        for i, stock1 in enumerate(available_stocks):
            for j, stock2 in enumerate(available_stocks):
                if i < j:  # Avoid duplicates
                    tested_pairs += 1
                    
                    if tested_pairs % 500 == 0:
                        print(f"   ... tested {tested_pairs} pairs so far")
                    
                    # Test cointegration
                    is_coint, p_value, hedge_ratio, r_squared = self.test_cointegration(
                        self.formation_prices[stock1], 
                        self.formation_prices[stock2]
                    )
                    
                    # Store all pairs (not just cointegrated ones)
                    stock1_name = stock1.replace(self.price_col_format, '')
                    stock2_name = stock2.replace(self.price_col_format, '')
                    
                    # Calculate correlation for comparison
                    s1 = self.formation_prices[stock1].dropna()
                    s2 = self.formation_prices[stock2].dropna()
                    common_dates = s1.index.intersection(s2.index)
                    correlation = s1[common_dates].corr(s2[common_dates]) if len(common_dates) > 10 else np.nan
                    
                    all_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'stock1_name': stock1_name,
                        'stock2_name': stock2_name,
                        'coint_p_value': p_value,
                        'hedge_ratio': hedge_ratio,
                        'r_squared': r_squared,
                        'correlation': correlation,
                        'is_cointegrated': is_coint,
                        'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    })
        
        # Sort by cointegration strength (lower p-value = stronger)
        all_pairs = sorted(all_pairs, key=lambda x: x['coint_p_value'] if not np.isnan(x['coint_p_value']) else 1.0)
        
        # Filter cointegrated pairs and take top ones
        cointegrated_pairs = [pair for pair in all_pairs if pair['is_cointegrated']]
        
        if len(cointegrated_pairs) > max_pairs:
            cointegrated_pairs = cointegrated_pairs[:max_pairs]
        
        self.cointegrated_pairs = cointegrated_pairs
        self.all_tested_pairs = all_pairs[:100]  # Keep top 100 for analysis
        
        print(f"\n   📊 Results Summary:")
        print(f"   • Total pairs tested: {tested_pairs}")
        print(f"   • Cointegrated pairs found: {len(cointegrated_pairs)}")
        print(f"   • Success rate: {len(cointegrated_pairs)/tested_pairs*100:.2f}%")
        
        if cointegrated_pairs:
            print(f"\n   🏆 Top {min(20, len(cointegrated_pairs))} cointegrated pairs:")
            for i, pair in enumerate(cointegrated_pairs[:20]):
                print(f"      {i+1:2d}. {pair['stock1_name']} - {pair['stock2_name']}: "
                      f"p={pair['coint_p_value']:.4f}{pair['significance']}, "
                      f"β={pair['hedge_ratio']:.3f}, R²={pair['r_squared']:.3f}, corr={pair['correlation']:.3f}")
        else:
            print(f"   ⚠️  No cointegrated pairs found at {self.significance_level*100}% significance!")
            print(f"   💡 Try increasing significance level (e.g., 0.10)")
            
            # Show best 10 pairs anyway
            print(f"\n   📈 Best 10 pairs by p-value (not significant):")
            for i, pair in enumerate(all_pairs[:10]):
                if not np.isnan(pair['coint_p_value']):
                    print(f"      {i+1:2d}. {pair['stock1_name']} - {pair['stock2_name']}: "
                          f"p={pair['coint_p_value']:.4f}, "
                          f"β={pair['hedge_ratio']:.3f}, R²={pair['r_squared']:.3f}")
        
        return self
    
    def backtest_strategy(self):
        """💰 Backtest the cointegration strategy on all pairs"""
        print(f"\n💰 Backtesting cointegration strategy...")
        print(f"   Entry Z-score: ±{self.entry_zscore}")
        print(f"   Exit Z-score: ±{self.exit_zscore}")
        print(f"   Transaction cost: {self.transaction_cost:.3%}")
        
        if not self.cointegrated_pairs:
            print("   ❌ No cointegrated pairs found for backtesting")
            return self
        
        all_trades = []
        pair_results = []
        
        print(f"   📊 Backtesting {len(self.cointegrated_pairs)} pairs...")
        
        for i, pair in enumerate(self.cointegrated_pairs):
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            hedge_ratio = pair['hedge_ratio']
            
            # Get formation data for spread calculation
            formation_s1 = self.formation_prices[stock1].dropna()
            formation_s2 = self.formation_prices[stock2].dropna()
            
            common_formation = formation_s1.index.intersection(formation_s2.index)
            if len(common_formation) < 50:
                continue
            
            # Calculate cointegrating spread in formation period
            formation_spread = formation_s1[common_formation] - hedge_ratio * formation_s2[common_formation]
            spread_mean = formation_spread.mean()
            spread_std = formation_spread.std()
            
            if spread_std == 0:
                continue
            
            # Get trading data
            trading_s1 = self.trading_prices[stock1].dropna()
            trading_s2 = self.trading_prices[stock2].dropna()
            
            common_trading = trading_s1.index.intersection(trading_s2.index)
            if len(common_trading) < 10:
                continue
            
            # Calculate trading spread and Z-scores
            trading_spread = trading_s1[common_trading] - hedge_ratio * trading_s2[common_trading]
            trading_zscore = (trading_spread - spread_mean) / spread_std
            
            # 🐛 FIX: Align price series with Z-score dates for trading execution
            trading_s1_aligned = trading_s1[common_trading]
            trading_s2_aligned = trading_s2[common_trading]
            
            # Add debug logging
            max_zscore = max(abs(trading_zscore.min()), abs(trading_zscore.max()))
            days_above_threshold = (abs(trading_zscore) > self.entry_zscore).sum()
            
            if max_zscore > self.entry_zscore:
                print(f"      📊 {pair['stock1_name']}-{pair['stock2_name']}: Max|Z|={max_zscore:.2f}, Days>{self.entry_zscore}={days_above_threshold}")
            
            # Execute trades for this pair (with aligned data)
            pair_trades = self._execute_trades(trading_zscore, pair, trading_s1_aligned, trading_s2_aligned)
            all_trades.extend(pair_trades)
            
            # Store pair results
            pair_pnl = sum(trade['net_pnl'] for trade in pair_trades)
            pair_results.append({
                'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                'trades': len(pair_trades),
                'pnl': pair_pnl,
                'coint_p_value': pair['coint_p_value'],
                'hedge_ratio': hedge_ratio
            })
        
        self.trades = all_trades
        self.pair_results = pair_results
        
        # Calculate performance metrics
        self._calculate_performance()
        
        print(f"   ✅ Backtest completed")
        print(f"   📊 Total trades across all pairs: {len(all_trades)}")
        if all_trades:
            total_pnl = sum(trade['net_pnl'] for trade in all_trades)
            winning_trades = sum(1 for trade in all_trades if trade['net_pnl'] > 0)
            print(f"   💰 Total P&L: ${total_pnl:.2f}")
            print(f"   📈 Win rate: {winning_trades/len(all_trades)*100:.1f}%")
        
        return self
    
    def _execute_trades(self, zscore_series, pair, price_s1, price_s2):
        """🎯 Execute trades based on cointegrating spread Z-scores"""
        trades = []
        position = 0  # 0=no position, 1=long spread, -1=short spread
        entry_info = {}
        
        hedge_ratio = pair['hedge_ratio']
        
        for date in zscore_series.index:
            if pd.isna(zscore_series[date]):
                continue
                
            zscore = zscore_series[date]
            
            # 🐛 FIX: Direct access since data is now properly aligned
            p1 = price_s1[date]
            p2 = price_s2[date]
            
            if pd.isna(p1) or pd.isna(p2):
                continue
            
            if position == 0:  # No position
                if zscore > self.entry_zscore:  # Short spread
                    position = -1
                    entry_info = {
                        'entry_date': date,
                        'entry_zscore': zscore,
                        'entry_p1': p1,
                        'entry_p2': p2,
                        'position_type': 'Short Spread'
                    }
                elif zscore < -self.entry_zscore:  # Long spread
                    position = 1
                    entry_info = {
                        'entry_date': date,
                        'entry_zscore': zscore,
                        'entry_p1': p1,
                        'entry_p2': p2,
                        'position_type': 'Long Spread'
                    }
            
            else:  # In position, check exit
                if abs(zscore) <= self.exit_zscore:
                    # Calculate P&L
                    if position == 1:  # Long spread position
                        # Long P1, Short hedge_ratio*P2
                        pnl_p1 = p1 - entry_info['entry_p1']
                        pnl_p2 = hedge_ratio * (entry_info['entry_p2'] - p2)
                        gross_pnl = pnl_p1 + pnl_p2
                    else:  # Short spread position
                        # Short P1, Long hedge_ratio*P2
                        pnl_p1 = entry_info['entry_p1'] - p1
                        pnl_p2 = hedge_ratio * (p2 - entry_info['entry_p2'])
                        gross_pnl = pnl_p1 + pnl_p2
                    
                    # Transaction costs
                    trade_value = entry_info['entry_p1'] + hedge_ratio * entry_info['entry_p2']
                    transaction_costs = 2 * self.transaction_cost * trade_value
                    net_pnl = gross_pnl - transaction_costs
                    
                    trades.append({
                        'pair': f"{pair['stock1_name']}-{pair['stock2_name']}",
                        'entry_date': entry_info['entry_date'],
                        'exit_date': date,
                        'entry_zscore': entry_info['entry_zscore'],
                        'exit_zscore': zscore,
                        'position_type': entry_info['position_type'],
                        'hedge_ratio': hedge_ratio,
                        'entry_p1': entry_info['entry_p1'],
                        'entry_p2': entry_info['entry_p2'],
                        'exit_p1': p1,
                        'exit_p2': p2,
                        'gross_pnl': gross_pnl,
                        'transaction_costs': transaction_costs,
                        'net_pnl': net_pnl,
                        'days_held': (date - entry_info['entry_date']).days,
                        'trade_value': trade_value
                    })
                    
                    position = 0
                    entry_info = {}
        
        return trades
    
    def _calculate_performance(self):
        """📊 Calculate comprehensive performance metrics"""
        if not self.trades:
            self.performance_metrics = {}
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['net_pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(trade['net_pnl'] for trade in self.trades)
        total_gross_pnl = sum(trade['gross_pnl'] for trade in self.trades)
        total_costs = sum(trade['transaction_costs'] for trade in self.trades)
        
        returns = [trade['net_pnl'] / trade['trade_value'] for trade in self.trades]
        avg_return = np.mean(returns) if returns else 0
        return_std = np.std(returns) if len(returns) > 1 else 0
        
        # Win/Loss analysis
        winning_pnls = [trade['net_pnl'] for trade in self.trades if trade['net_pnl'] > 0]
        losing_pnls = [trade['net_pnl'] for trade in self.trades if trade['net_pnl'] < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Risk metrics
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Store metrics
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_gross_pnl': total_gross_pnl,
            'total_costs': total_costs,
            'avg_return': avg_return,
            'return_std': return_std,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf
        }
    
    def print_results(self):
        """📊 Print comprehensive results"""
        print(f"\n" + "="*70)
        print(f"📊 COMPREHENSIVE RESULTS - TOP {len(self.cointegrated_pairs)} COINTEGRATED PAIRS")
        print("="*70)
        
        if not self.cointegrated_pairs:
            print("❌ No cointegrated pairs found")
            return
        
        # Pair analysis
        print(f"\n🔗 COINTEGRATION ANALYSIS:")
        print(f"   Pairs tested: {len(self.all_tested_pairs) if hasattr(self, 'all_tested_pairs') else 'N/A'}")
        print(f"   Cointegrated pairs: {len(self.cointegrated_pairs)}")
        print(f"   Significance level: {self.significance_level*100}%")
        
        # Trading results
        if self.performance_metrics:
            metrics = self.performance_metrics
            print(f"\n💰 TRADING PERFORMANCE:")
            print(f"   Total trades: {metrics['total_trades']}")
            print(f"   Winning trades: {metrics['winning_trades']} ({metrics['win_rate']*100:.1f}%)")
            print(f"   Losing trades: {metrics['losing_trades']}")
            print(f"   Total P&L: ${metrics['total_pnl']:.2f}")
            print(f"   Gross P&L: ${metrics['total_gross_pnl']:.2f}")
            print(f"   Transaction costs: ${metrics['total_costs']:.2f}")
            print(f"   Average return per trade: {metrics['avg_return']*100:.3f}%")
            print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Profit factor: {metrics['profit_factor']:.2f}")
            
            if metrics['winning_trades'] > 0:
                print(f"   Average winning trade: ${metrics['avg_win']:.2f}")
            if metrics['losing_trades'] > 0:
                print(f"   Average losing trade: ${metrics['avg_loss']:.2f}")
        
        # Pair-by-pair results
        if hasattr(self, 'pair_results') and self.pair_results:
            print(f"\n📈 PAIR-BY-PAIR RESULTS:")
            sorted_pairs = sorted(self.pair_results, key=lambda x: x['pnl'], reverse=True)
            for i, result in enumerate(sorted_pairs[:10]):
                print(f"   {i+1:2d}. {result['pair']}: {result['trades']} trades, ${result['pnl']:.2f} P&L")
        
        # Best pairs summary
        print(f"\n🏆 BEST PERFORMING PAIRS:")
        best_pairs = sorted(self.cointegrated_pairs, key=lambda x: x['coint_p_value'])[:5]
        for i, pair in enumerate(best_pairs):
            print(f"   {i+1}. {pair['stock1_name']}-{pair['stock2_name']}: p={pair['coint_p_value']:.4f}, β={pair['hedge_ratio']:.3f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Raw Prices Cointegration Strategy - Top 20 Pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('data_path', help='Path to pair trading data directory')
    parser.add_argument('--max-pairs', type=int, default=20, help='Maximum pairs to analyze (default: 20)')
    parser.add_argument('--min-stocks', type=int, default=100, help='Number of stocks to test (default: 100)')
    parser.add_argument('--entry-threshold', type=float, default=1.0, help='Entry Z-score threshold')
    parser.add_argument('--exit-threshold', type=float, default=0.0, help='Exit Z-score threshold')
    parser.add_argument('--significance', type=float, default=0.05, help='Statistical significance level')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost rate')
    
    args = parser.parse_args()
    
    try:
        strategy = RawPricesCointegrationStrategy(
            significance_level=args.significance,
            entry_zscore=args.entry_threshold,
            exit_zscore=args.exit_threshold,
            transaction_cost=args.transaction_cost
        )
        
        # Execute complete strategy
        strategy.load_data(args.data_path)
        strategy.find_cointegrated_pairs(max_pairs=args.max_pairs, min_stocks=args.min_stocks)
        strategy.backtest_strategy()
        strategy.print_results()
        
        print("\n🎯 Strategy analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 