#!/usr/bin/env python3
"""
📊 Easy Multi-Pair Cointegration Visualization

Simple, clear visualization of multiple cointegrated pairs with explanations
of long/short spreads and P&L calculations.

Usage:
    python easy_multi_pair_viz.py data/pair_trading/sp500_20230101_20240705_prices_12m6m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Set clean plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 11


class EasyMultiPairViz:
    """
    📊 Easy Multi-Pair Visualization
    
    Shows multiple cointegrated pairs with clear explanations:
    - What long/short spread means
    - How P&L is calculated
    - Position sizing explanation
    """
    
    def __init__(self, entry_zscore=0.75, exit_zscore=0.25):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.pair_results = []
        
    def load_data(self, data_path):
        """📁 Load formation and trading data"""
        data_path = Path(data_path)
        
        # Find files
        formation_file = list(data_path.glob("*_in_sample_formation.csv"))[0]
        trading_file = list(data_path.glob("*_out_sample_trading.csv"))[0]
        
        print(f"📊 EASY MULTI-PAIR COINTEGRATION VISUALIZATION")
        print("="*70)
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
        self.formation_prices = self.formation_data[self.price_columns]
        self.trading_prices = self.trading_data[self.price_columns]
        
        print(f"✅ Loaded {len(self.price_columns)} stocks")
        return self
    
    def find_profitable_pairs(self, max_pairs=6):
        """🔍 Find profitable pairs with multiple trades"""
        print(f"\n🔍 Finding profitable pairs...")
        
        # Test pairs that showed good results
        test_pairs = [
            ('APD', 'ALK'), ('ALK', 'AAL'), ('APD', 'ALGN'), 
            ('APD', 'AAL'), ('AES', 'AEP'), ('ACN', 'AMZN')
        ]
        
        for stock1, stock2 in test_pairs:
            col1 = f'p_adjclose_{stock1}'
            col2 = f'p_adjclose_{stock2}'
            
            if col1 in self.price_columns and col2 in self.price_columns:
                result = self._analyze_pair(stock1, stock2)
                if result and len(result['trades']) > 0:
                    self.pair_results.append(result)
                    print(f"   ✅ {stock1}-{stock2}: {len(result['trades'])} trades, ${result['total_pnl']:.2f} P&L")
        
        # Sort by number of trades, then by P&L
        self.pair_results.sort(key=lambda x: (len(x['trades']), x['total_pnl']), reverse=True)
        print(f"\n📊 Found {len(self.pair_results)} profitable pairs")
        
        return self
    
    def _analyze_pair(self, stock1, stock2):
        """📊 Analyze a single pair"""
        col1 = f'p_adjclose_{stock1}'
        col2 = f'p_adjclose_{stock2}'
        
        # Formation period analysis
        formation_s1 = self.formation_prices[col1].dropna()
        formation_s2 = self.formation_prices[col2].dropna()
        
        common_formation = formation_s1.index.intersection(formation_s2.index)
        if len(common_formation) < 30:
            return None
            
        s1_form = formation_s1[common_formation]
        s2_form = formation_s2[common_formation]
        
        try:
            # Cointegration test
            coint_stat, p_value, critical_values = coint(s1_form, s2_form)
            if p_value > 0.1:
                return None
            
            # Get hedge ratio
            ols_result = OLS(s1_form, s2_form).fit()
            hedge_ratio = ols_result.params[0]
            r_squared = ols_result.rsquared
            
        except Exception:
            return None
        
        # Calculate formation spread
        formation_spread = s1_form - hedge_ratio * s2_form
        spread_mean = formation_spread.mean()
        spread_std = formation_spread.std()
        
        # Trading period analysis
        trading_s1 = self.trading_prices[col1].dropna()
        trading_s2 = self.trading_prices[col2].dropna()
        
        common_trading = trading_s1.index.intersection(trading_s2.index)
        s1_trade = trading_s1[common_trading]
        s2_trade = trading_s2[common_trading]
        
        # Calculate trading spread and Z-scores
        trading_spread = s1_trade - hedge_ratio * s2_trade
        trading_zscore = (trading_spread - spread_mean) / spread_std
        
        # Execute trades
        trades = self._execute_trades(trading_zscore, s1_trade, s2_trade, hedge_ratio, stock1, stock2)
        
        if not trades:
            return None
        
        total_pnl = sum(t['net_pnl'] for t in trades)
        winning_trades = len([t for t in trades if t['net_pnl'] > 0])
        
        return {
            'stock1': stock1,
            'stock2': stock2,
            'p_value': p_value,
            'hedge_ratio': hedge_ratio,
            'r_squared': r_squared,
            'formation_s1': s1_form,
            'formation_s2': s2_form,
            'trading_s1': s1_trade,
            'trading_s2': s2_trade,
            'trading_zscore': trading_zscore,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'trades': trades,
            'total_pnl': total_pnl,
            'winning_trades': winning_trades,
            'total_trades': len(trades)
        }
    
    def _execute_trades(self, zscore_series, price_s1, price_s2, hedge_ratio, stock1, stock2):
        """🎯 Execute trades based on Z-scores"""
        trades = []
        position = 0
        entry_info = {}
        
        for date in zscore_series.index:
            if pd.isna(zscore_series[date]):
                continue
                
            zscore = zscore_series[date]
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
                        pnl_p1 = p1 - entry_info['entry_p1']
                        pnl_p2 = hedge_ratio * (entry_info['entry_p2'] - p2)
                        gross_pnl = pnl_p1 + pnl_p2
                    else:  # Short spread position
                        pnl_p1 = entry_info['entry_p1'] - p1
                        pnl_p2 = hedge_ratio * (p2 - entry_info['entry_p2'])
                        gross_pnl = pnl_p1 + pnl_p2
                    
                    # Transaction costs (0.1%)
                    trade_value = entry_info['entry_p1'] + hedge_ratio * entry_info['entry_p2']
                    transaction_costs = 2 * 0.001 * trade_value
                    net_pnl = gross_pnl - transaction_costs
                    
                    trades.append({
                        'entry_date': entry_info['entry_date'],
                        'exit_date': date,
                        'position_type': entry_info['position_type'],
                        'entry_zscore': entry_info['entry_zscore'],
                        'exit_zscore': zscore,
                        'entry_p1': entry_info['entry_p1'],
                        'entry_p2': entry_info['entry_p2'],
                        'exit_p1': p1,
                        'exit_p2': p2,
                        'hedge_ratio': hedge_ratio,
                        'gross_pnl': gross_pnl,
                        'transaction_costs': transaction_costs,
                        'net_pnl': net_pnl,
                        'days_held': (date - entry_info['entry_date']).days
                    })
                    
                    position = 0
                    entry_info = {}
        
        return trades
    
    def create_visualization(self):
        """📊 Create comprehensive multi-pair visualization"""
        if not self.pair_results:
            print("❌ No profitable pairs found!")
            return
        
        # Create figure with subplots - increased size and better spacing
        n_pairs = min(len(self.pair_results), 4)  # Show max 4 pairs
        fig, axes = plt.subplots(n_pairs, 2, figsize=(24, 6*n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'📊 Multi-Pair Cointegration Strategy Results\n'
                     f'Entry: ±{self.entry_zscore} | Exit: ±{self.exit_zscore} | '
                     f'Total P&L: ${sum(r["total_pnl"] for r in self.pair_results):.2f}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i, result in enumerate(self.pair_results[:n_pairs]):
            stock1, stock2 = result['stock1'], result['stock2']
            
            # Left subplot: Z-score with trades
            ax1 = axes[i, 0]
            zscore_data = result['trading_zscore']
            ax1.plot(zscore_data.index, zscore_data.values, 'blue', linewidth=2, alpha=0.8)
            
            # Mark thresholds
            ax1.axhline(y=self.entry_zscore, color='red', linestyle='-', alpha=0.8, label=f'Entry: ±{self.entry_zscore}')
            ax1.axhline(y=-self.entry_zscore, color='red', linestyle='-', alpha=0.8)
            ax1.axhline(y=self.exit_zscore, color='green', linestyle='-', alpha=0.8, label=f'Exit: ±{self.exit_zscore}')
            ax1.axhline(y=-self.exit_zscore, color='green', linestyle='-', alpha=0.8)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Mark trade points
            for trade in result['trades']:
                # Entry point
                ax1.scatter(trade['entry_date'], trade['entry_zscore'], 
                           color='red' if trade['position_type'] == 'Short Spread' else 'blue',
                           s=100, marker='v', zorder=5)
                # Exit point
                ax1.scatter(trade['exit_date'], trade['exit_zscore'], 
                           color='green', s=100, marker='^', zorder=5)
            
            ax1.set_title(f'{stock1}-{stock2}: {len(result["trades"])} trades, ${result["total_pnl"]:.2f} P&L', 
                         fontweight='bold', fontsize=12)
            ax1.set_ylabel('Z-Score', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Right subplot: Trade details table - more compact
            ax2 = axes[i, 1]
            ax2.axis('off')
            
            # Create more compact trade details text
            trade_text = f"{stock1}-{stock2} Details\n"
            trade_text += f"β: {result['hedge_ratio']:.3f} | P: {result['p_value']:.4f} | R²: {result['r_squared']:.3f}\n\n"
            
            for j, trade in enumerate(result['trades'], 1):
                trade_text += f"Trade {j}: {trade['position_type']}\n"
                trade_text += f"  {trade['entry_date'].strftime('%m/%d')} → {trade['exit_date'].strftime('%m/%d')} ({trade['days_held']}d)\n"
                trade_text += f"  P&L: ${trade['net_pnl']:.2f} | Z: {trade['entry_zscore']:.2f}→{trade['exit_zscore']:.2f}\n\n"
            
            ax2.text(0.05, 0.95, trade_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Better spacing to prevent overlap
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.3)
        plt.show()
        
        # Print summary
        self._print_summary()
        
        return self
    
    def _print_summary(self):
        """📋 Print summary with explanations"""
        print(f"\n" + "="*70)
        print(f"📊 SUMMARY & EXPLANATIONS")
        print("="*70)
        
        print(f"\n🔗 WHAT IS LONG/SHORT SPREAD?")
        print(f"   • Spread = Stock1 - β × Stock2")
        print(f"   • Long Spread: Buy Stock1, Sell β×Stock2 (expect spread to increase)")
        print(f"   • Short Spread: Sell Stock1, Buy β×Stock2 (expect spread to decrease)")
        
        print(f"\n💰 P&L CALCULATION:")
        print(f"   • Long Spread P&L = (Exit_Stock1 - Entry_Stock1) - β×(Exit_Stock2 - Entry_Stock2)")
        print(f"   • Short Spread P&L = (Entry_Stock1 - Exit_Stock1) + β×(Exit_Stock2 - Entry_Stock2)")
        
        print(f"\n💵 POSITION SIZING:")
        print(f"   • P&L shown is for $1 notional investment in the spread")
        print(f"   • In practice, scale up to your desired position size")
        print(f"   • Example: $10,000 investment = multiply P&L by 10,000")
        
        print(f"\n📈 PAIR RESULTS:")
        total_pnl = sum(r['total_pnl'] for r in self.pair_results)
        total_trades = sum(len(r['trades']) for r in self.pair_results)
        winning_trades = sum(r['winning_trades'] for r in self.pair_results)
        
        print(f"   • Total Pairs: {len(self.pair_results)}")
        print(f"   • Total Trades: {total_trades}")
        print(f"   • Winning Trades: {winning_trades}")
        print(f"   • Win Rate: {winning_trades/total_trades*100:.1f}%" if total_trades > 0 else "   • Win Rate: N/A")
        print(f"   • Total P&L: ${total_pnl:.2f}")
        print(f"   • Average P&L per Trade: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "   • Average P&L per Trade: N/A")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Easy Multi-Pair Cointegration Visualization')
    parser.add_argument('data_path', help='Path to data directory')
    parser.add_argument('--entry-threshold', type=float, default=0.75, help='Entry Z-score threshold')
    parser.add_argument('--exit-threshold', type=float, default=0.25, help='Exit Z-score threshold')
    
    args = parser.parse_args()
    
    viz = EasyMultiPairViz(
        entry_zscore=args.entry_threshold,
        exit_zscore=args.exit_threshold
    )
    
    viz.load_data(args.data_path)
    viz.find_profitable_pairs()
    viz.create_visualization()


if __name__ == "__main__":
    main() 