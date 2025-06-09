#!/usr/bin/env python3
"""
⚖️ Strategy Comparison: Correlation vs Cointegration

This script runs both strategies and provides side-by-side comparison:
1. Correlation-based pair trading strategy
2. Cointegration-based pair trading strategy
3. Comprehensive comparison analysis

Usage:
    python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add strategy directories to path
sys.path.append(str(Path(__file__).parent / "correlation_based"))
sys.path.append(str(Path(__file__).parent / "cointegration_based"))

from correlation_strategy import CorrelationPairTradingStrategy
from cointegration_strategy import CointegrationPairTradingStrategy

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class StrategyComparison:
    """⚖️ Compare correlation vs cointegration strategies"""
    
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.kwargs = kwargs
        
        # Initialize strategies
        self.correlation_strategy = CorrelationPairTradingStrategy(
            correlation_threshold=kwargs.get('correlation', 0.80),
            entry_zscore=kwargs.get('entry_zscore', 2.0),
            exit_zscore=kwargs.get('exit_zscore', 0.5)
        )
        
        self.cointegration_strategy = CointegrationPairTradingStrategy(
            significance_level=kwargs.get('significance', 0.05),
            entry_zscore=kwargs.get('entry_zscore', 2.0),
            exit_zscore=kwargs.get('exit_zscore', 0.5)
        )
        
    def run_comparison(self):
        """🚀 Run both strategies and compare results"""
        print("⚖️ STRATEGY COMPARISON: CORRELATION vs COINTEGRATION")
        print("="*80)
        
        # Run correlation strategy
        print("\n" + "="*30 + " CORRELATION STRATEGY " + "="*30)
        try:
            self.correlation_strategy.load_data(self.data_path)
            self.correlation_strategy.find_correlated_pairs(max_pairs=self.kwargs.get('max_pairs', 20))
            self.correlation_strategy.backtest_strategy()
            correlation_success = True
        except Exception as e:
            print(f"❌ Correlation strategy failed: {e}")
            correlation_success = False
        
        # Run cointegration strategy
        print("\n" + "="*30 + " COINTEGRATION STRATEGY " + "="*30)
        try:
            self.cointegration_strategy.load_data(self.data_path)
            focus_stocks = self.kwargs.get('focus_stocks')
            self.cointegration_strategy.find_cointegrated_pairs(
                max_pairs=self.kwargs.get('max_pairs', 50), 
                focus_stocks=focus_stocks
            )
            self.cointegration_strategy.backtest_strategy()
            cointegration_success = True
        except Exception as e:
            print(f"❌ Cointegration strategy failed: {e}")
            cointegration_success = False
        
        # Generate comparison
        if correlation_success and cointegration_success:
            self._compare_results()
        elif correlation_success:
            print("\n⚠️ Only correlation strategy succeeded")
        elif cointegration_success:
            print("\n⚠️ Only cointegration strategy succeeded")
        else:
            print("\n❌ Both strategies failed")
    
    def _compare_results(self):
        """📊 Compare and analyze results from both strategies"""
        print("\n" + "="*25 + " STRATEGY COMPARISON RESULTS " + "="*25)
        
        # Get performance metrics
        corr_perf = getattr(self.correlation_strategy, 'performance', None)
        coint_perf = getattr(self.cointegration_strategy, 'performance', None)
        
        if not corr_perf or not coint_perf:
            print("❌ Cannot compare - one or both strategies have no performance data")
            return
        
        # Create comparison table
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("-" * 80)
        print(f"{'Metric':<25} {'Correlation':<15} {'Cointegration':<15} {'Winner':<15}")
        print("-" * 80)
        
        # Compare metrics
        metrics_comparison = {
            'Total Trades': (corr_perf['total_trades'], coint_perf['total_trades']),
            'Win Rate': (corr_perf['win_rate'], coint_perf['win_rate']),
            'Total PnL': (corr_perf['total_pnl'], coint_perf['total_pnl']),
            'Average PnL': (corr_perf['avg_pnl'], coint_perf['avg_pnl']),
            'Sharpe Ratio': (corr_perf['sharpe_ratio'], coint_perf['sharpe_ratio']),
            'Significance (p-value)': (corr_perf['t_p_value'], coint_perf['t_p_value'])
        }
        
        for metric, (corr_val, coint_val) in metrics_comparison.items():
            if metric == 'Significance (p-value)':
                winner = 'Correlation' if corr_val < coint_val else 'Cointegration' if coint_val < corr_val else 'Tie'
                corr_str = f"{corr_val:.6f}"
                coint_str = f"{coint_val:.6f}"
            elif metric in ['Win Rate']:
                winner = 'Correlation' if corr_val > coint_val else 'Cointegration' if coint_val > corr_val else 'Tie'
                corr_str = f"{corr_val:.1%}"
                coint_str = f"{coint_val:.1%}"
            elif metric == 'Total Trades':
                winner = 'More trades' if corr_val > coint_val else 'More trades' if coint_val > corr_val else 'Tie'
                corr_str = f"{corr_val}"
                coint_str = f"{coint_val}"
            else:
                winner = 'Correlation' if corr_val > coint_val else 'Cointegration' if coint_val > corr_val else 'Tie'
                corr_str = f"{corr_val:.4f}"
                coint_str = f"{coint_val:.4f}"
            
            print(f"{metric:<25} {corr_str:<15} {coint_str:<15} {winner:<15}")
        
        # Strategy-specific comparison
        print(f"\n📈 STRATEGY-SPECIFIC METRICS")
        print("-" * 50)
        print(f"Correlation pairs found: {len(self.correlation_strategy.pairs)}")
        print(f"Cointegrated pairs found: {len(self.cointegration_strategy.cointegrated_pairs)}")
        
        if hasattr(coint_perf, 'avg_coint_p'):
            print(f"Average cointegration p-value: {coint_perf['avg_coint_p']:.6f}")
        
        # Overall winner
        self._determine_overall_winner(corr_perf, coint_perf)
    
    def _determine_overall_winner(self, corr_perf, coint_perf):
        """🏆 Determine overall winner based on multiple criteria"""
        print(f"\n🏆 OVERALL ASSESSMENT")
        print("-" * 40)
        
        # Scoring system
        corr_score = 0
        coint_score = 0
        
        # PnL (most important)
        if corr_perf['total_pnl'] > coint_perf['total_pnl']:
            corr_score += 3
        elif coint_perf['total_pnl'] > corr_perf['total_pnl']:
            coint_score += 3
        
        # Sharpe ratio
        if corr_perf['sharpe_ratio'] > coint_perf['sharpe_ratio']:
            corr_score += 2
        elif coint_perf['sharpe_ratio'] > corr_perf['sharpe_ratio']:
            coint_score += 2
        
        # Win rate
        if corr_perf['win_rate'] > coint_perf['win_rate']:
            corr_score += 2
        elif coint_perf['win_rate'] > corr_perf['win_rate']:
            coint_score += 2
        
        # Statistical significance
        if corr_perf['t_p_value'] < 0.05:
            corr_score += 1
        if coint_perf['t_p_value'] < 0.05:
            coint_score += 1
        
        # Determine winner
        if corr_score > coint_score:
            winner = "CORRELATION STRATEGY"
            margin = corr_score - coint_score
        elif coint_score > corr_score:
            winner = "COINTEGRATION STRATEGY"
            margin = coint_score - corr_score
        else:
            winner = "TIE"
            margin = 0
        
        print(f"🥇 Winner: {winner}")
        if margin > 0:
            print(f"   Margin: {margin} points")
        
        print(f"\n💡 INTERPRETATION:")
        if winner == "CORRELATION STRATEGY":
            print("   • High correlation worked well for this period")
            print("   • Simpler approach proved effective")
            print("   • May work in trending/momentum markets")
        elif winner == "COINTEGRATION STRATEGY":
            print("   • Long-term equilibrium relationships identified")
            print("   • More robust theoretical foundation")
            print("   • Better for mean-reverting markets")
        else:
            print("   • Both strategies performed similarly")
            print("   • Market regime may favor neither approach")
            print("   • Consider combining or alternative parameters")
    
    def plot_comparison(self, save_path=None):
        """📊 Create side-by-side comparison plots"""
        if not hasattr(self.correlation_strategy, 'trades') or not hasattr(self.cointegration_strategy, 'trades'):
            print("❌ Cannot plot - no trade data available")
            return
        
        if not self.correlation_strategy.trades and not self.cointegration_strategy.trades:
            print("❌ Cannot plot - no trades executed by either strategy")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('⚖️ Strategy Comparison: Correlation vs Cointegration', 
                     fontsize=16, fontweight='bold')
        
        # Get trade data
        corr_pnls = [t['pnl'] for t in self.correlation_strategy.trades] if self.correlation_strategy.trades else []
        coint_pnls = [t['pnl'] for t in self.cointegration_strategy.trades] if self.cointegration_strategy.trades else []
        
        # Plot 1: PnL Distribution Comparison
        ax1 = axes[0, 0]
        if corr_pnls:
            ax1.hist(corr_pnls, bins=15, alpha=0.7, label='Correlation', color='skyblue', density=True)
        if coint_pnls:
            ax1.hist(coint_pnls, bins=15, alpha=0.7, label='Cointegration', color='lightgreen', density=True)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('PnL Distribution Comparison')
        ax1.set_xlabel('PnL per Trade')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative PnL Comparison
        ax2 = axes[0, 1]
        if corr_pnls:
            corr_cumulative = np.cumsum(corr_pnls)
            ax2.plot(range(1, len(corr_cumulative) + 1), corr_cumulative, 
                    label='Correlation', linewidth=2, color='blue')
        if coint_pnls:
            coint_cumulative = np.cumsum(coint_pnls)
            ax2.plot(range(1, len(coint_cumulative) + 1), coint_cumulative, 
                    label='Cointegration', linewidth=2, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Cumulative PnL Comparison')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative PnL')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Metrics Bar Chart
        ax3 = axes[0, 2]
        if hasattr(self.correlation_strategy, 'performance') and hasattr(self.cointegration_strategy, 'performance'):
            metrics = ['Win Rate', 'Avg PnL', 'Sharpe Ratio']
            corr_values = [
                self.correlation_strategy.performance['win_rate'] * 100,
                self.correlation_strategy.performance['avg_pnl'] * 1000,  # Scale for visibility
                self.correlation_strategy.performance['sharpe_ratio']
            ]
            coint_values = [
                self.cointegration_strategy.performance['win_rate'] * 100,
                self.cointegration_strategy.performance['avg_pnl'] * 1000,  # Scale for visibility
                self.cointegration_strategy.performance['sharpe_ratio']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax3.bar(x - width/2, corr_values, width, label='Correlation', color='skyblue', alpha=0.8)
            ax3.bar(x + width/2, coint_values, width, label='Cointegration', color='lightgreen', alpha=0.8)
            
            ax3.set_title('Performance Metrics Comparison')
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Values')
            ax3.set_xticks(x)
            ax3.set_xticklabels(['Win Rate (%)', 'Avg PnL (×1000)', 'Sharpe Ratio'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trade Count Comparison
        ax4 = axes[1, 0]
        strategies = ['Correlation', 'Cointegration']
        trade_counts = [len(corr_pnls), len(coint_pnls)]
        colors = ['skyblue', 'lightgreen']
        
        bars = ax4.bar(strategies, trade_counts, color=colors, alpha=0.8)
        ax4.set_title('Number of Trades Comparison')
        ax4.set_ylabel('Number of Trades')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, trade_counts):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Pairs Found Comparison
        ax5 = axes[1, 1]
        pair_counts = [len(self.correlation_strategy.pairs), len(self.cointegration_strategy.cointegrated_pairs)]
        
        bars = ax5.bar(strategies, pair_counts, color=colors, alpha=0.8)
        ax5.set_title('Pairs Found Comparison')
        ax5.set_ylabel('Number of Pairs')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, pair_counts):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Summary Statistics Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        if hasattr(self.correlation_strategy, 'performance') and hasattr(self.cointegration_strategy, 'performance'):
            corr_perf = self.correlation_strategy.performance
            coint_perf = self.cointegration_strategy.performance
            
            summary_text = f"""
STRATEGY COMPARISON SUMMARY

📊 CORRELATION STRATEGY:
   • Pairs Found: {len(self.correlation_strategy.pairs)}
   • Trades: {corr_perf['total_trades']}
   • Win Rate: {corr_perf['win_rate']:.1%}
   • Total PnL: {corr_perf['total_pnl']:.4f}
   • Sharpe: {corr_perf['sharpe_ratio']:.4f}

🔗 COINTEGRATION STRATEGY:
   • Pairs Found: {len(self.cointegration_strategy.cointegrated_pairs)}
   • Trades: {coint_perf['total_trades']}
   • Win Rate: {coint_perf['win_rate']:.1%}
   • Total PnL: {coint_perf['total_pnl']:.4f}
   • Sharpe: {coint_perf['sharpe_ratio']:.4f}

🏆 WINNER:
   • PnL: {'Correlation' if corr_perf['total_pnl'] > coint_perf['total_pnl'] else 'Cointegration'}
   • Sharpe: {'Correlation' if corr_perf['sharpe_ratio'] > coint_perf['sharpe_ratio'] else 'Cointegration'}
   • Win Rate: {'Correlation' if corr_perf['win_rate'] > coint_perf['win_rate'] else 'Cointegration'}
            """
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Comparison plot saved to: {save_path}")
        
        plt.show()


def main():
    """🚀 Main function"""
    parser = argparse.ArgumentParser(
        description='Strategy Comparison: Correlation vs Cointegration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚖️ STRATEGY COMPARISON EXPLANATION:

🎯 PURPOSE:
   Compare correlation-based vs cointegration-based pair trading strategies
   to determine which approach works better for given market conditions.

📊 COMPARISON METRICS:
   • Total PnL and average PnL per trade
   • Win rate and Sharpe ratio
   • Statistical significance
   • Number of pairs found and trades executed

💡 INTERPRETATION:
   • Correlation strategy: Based on linear relationships
   • Cointegration strategy: Based on long-term equilibrium
   • Winner depends on market regime and time period

🔍 USAGE EXAMPLES:
   python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m
   python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m --correlation 0.85 --significance 0.01 --plot
        """
    )
    
    parser.add_argument('data_path', help='Path to pair trading data directory')
    parser.add_argument('--correlation', '-c', type=float, default=0.80,
                       help='Correlation threshold (default: 0.80)')
    parser.add_argument('--significance', '-s', type=float, default=0.05,
                       help='Cointegration significance level (default: 0.05)')
    parser.add_argument('--entry-zscore', '-e', type=float, default=2.0,
                       help='Entry Z-score threshold (default: 2.0)')
    parser.add_argument('--exit-zscore', '-x', type=float, default=0.5,
                       help='Exit Z-score threshold (default: 0.5)')
    parser.add_argument('--max-pairs', '-m', type=int, default=20,
                       help='Maximum pairs to analyze (default: 20)')
    parser.add_argument('--focus-stocks', nargs='+',
                       help='Focus on specific stocks for cointegration')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    try:
        comparison = StrategyComparison(
            data_path=args.data_path,
            correlation=args.correlation,
            significance=args.significance,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore,
            max_pairs=args.max_pairs,
            focus_stocks=args.focus_stocks
        )
        
        comparison.run_comparison()
        
        if args.plot:
            plot_path = Path(args.data_path) / "strategy_comparison.png"
            comparison.plot_comparison(save_path=plot_path)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 