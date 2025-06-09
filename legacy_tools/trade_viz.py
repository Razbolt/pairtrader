#!/usr/bin/env python3
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

class TradeAnalysisVisualization:
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.trades = []
        
    def load_data(self, data_path):
        data_path = Path(data_path)
        formation_file = None
        trading_file = None
        
        for file in data_path.glob("*_in_sample_formation.csv"):
            formation_file = file
        for file in data_path.glob("*_out_sample_trading.csv"):
            trading_file = file
            
        if not formation_file or not trading_file:
            raise FileNotFoundError("Could not find formation or trading CSV files")
        
        print(f"📊 TRADE ANALYSIS VISUALIZATION")
        print("="*60)
        print(f"📈 Formation data: {formation_file.name}")
        print(f"💰 Trading data: {trading_file.name}")
        
        formation_data = pd.read_csv(formation_file)
        trading_data = pd.read_csv(trading_file)
        
        for df in [formation_data, trading_data]:
            date_col = 'period' if 'period' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        stock_columns = [col for col in formation_data.columns if col.startswith('R_')]
        
        self.formation_prices = formation_data[stock_columns].cumsum()
        self.trading_prices = trading_data[stock_columns].cumsum()
        self.stock_columns = stock_columns
        
        print(f"   ✅ Formation: {self.formation_prices.index.min().strftime('%Y-%m-%d')} to {self.formation_prices.index.max().strftime('%Y-%m-%d')}")
        print(f"   ✅ Trading: {self.trading_prices.index.min().strftime('%Y-%m-%d')} to {self.trading_prices.index.max().strftime('%Y-%m-%d')}")
        print(f"   📊 Stocks: {len(self.stock_columns)}")
        
        return self
    
    def analyze_pair(self, stock1='AMZN', stock2='NKE'):
        print(f"\n📊 ANALYZING PAIR: {stock1} vs {stock2}")
        print("="*50)
        
        col1 = f'R_{stock1}'
        col2 = f'R_{stock2}'
        
        if col1 not in self.stock_columns or col2 not in self.stock_columns:
            available_tickers = [col.replace('R_', '') for col in self.stock_columns[:10]]
            print(f"❌ {stock1} or {stock2} not available")
            print(f"   Available tickers: {', '.join(available_tickers)}")
            stock1 = available_tickers[0]
            stock2 = available_tickers[1]
            col1 = f'R_{stock1}'
            col2 = f'R_{stock2}'
            print(f"   Using {stock1} vs {stock2} instead")
        
        formation_prices_1 = self.formation_prices[col1].dropna()
        formation_prices_2 = self.formation_prices[col2].dropna()
        
        common_formation = formation_prices_1.index.intersection(formation_prices_2.index)
        s1_form = formation_prices_1[common_formation]
        s2_form = formation_prices_2[common_formation]
        
        try:
            coint_stat, p_value, critical_values = coint(s1_form, s2_form)
            print(f"🔗 Cointegration test: p={p_value:.6f}")
            
            ols_result = OLS(s1_form, s2_form).fit()
            hedge_ratio = ols_result.params[0]
            r_squared = ols_result.rsquared
            
            print(f"   Hedge Ratio: {hedge_ratio:.4f}")
            print(f"   R-squared: {r_squared:.4f}")
            
        except Exception as e:
            print(f"❌ Cointegration test failed: {e}")
            return self
        
        formation_spread = s1_form - hedge_ratio * s2_form
        spread_mean = formation_spread.mean()
        spread_std = formation_spread.std()
        
        print(f"   Formation spread - Mean: {spread_mean:.6f}, Std: {spread_std:.6f}")
        
        trading_prices_1 = self.trading_prices[col1].dropna()
        trading_prices_2 = self.trading_prices[col2].dropna()
        
        common_trading = trading_prices_1.index.intersection(trading_prices_2.index)
        s1_trade = trading_prices_1[common_trading]
        s2_trade = trading_prices_2[common_trading]
        
        trading_spread = s1_trade - hedge_ratio * s2_trade
        trading_zscore = (trading_spread - spread_mean) / spread_std
        
        print(f"   Trading Z-score range: {trading_zscore.min():.3f} to {trading_zscore.max():.3f}")
        
        self._simulate_trades(trading_zscore, stock1, stock2)
        
        self.plot_data = {
            'stock1': stock1,
            'stock2': stock2,
            'formation_prices_1': s1_form,
            'formation_prices_2': s2_form,
            'trading_prices_1': s1_trade,
            'trading_prices_2': s2_trade,
            'trading_zscore': trading_zscore,
            'coint_p_value': p_value,
        }
        
        return self
    
    def _simulate_trades(self, zscore_series, stock1, stock2):
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        print(f"\n💰 SIMULATING TRADES")
        print(f"   Entry threshold: ±{self.entry_zscore}")
        print(f"   Exit threshold: ±{self.exit_zscore}")
        
        for date, zscore in zscore_series.items():
            if position == 0:
                if zscore > self.entry_zscore:
                    position = -1
                    entry_price = zscore
                    entry_date = date
                elif zscore < -self.entry_zscore:
                    position = 1
                    entry_price = zscore
                    entry_date = date
            else:
                exit_signal = False
                
                if position == 1 and zscore > -self.exit_zscore:
                    exit_signal = True
                elif position == -1 and zscore < self.exit_zscore:
                    exit_signal = True
                
                if exit_signal:
                    pnl = position * (entry_price - zscore)
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_zscore': entry_price,
                        'exit_zscore': zscore,
                        'position': 'Long' if position == 1 else 'Short',
                        'pnl': pnl,
                        'days_held': (date - entry_date).days
                    })
                    
                    position = 0
        
        self.trades = trades
        
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            print(f"   Total trades: {len(trades)}")
            print(f"   Winning trades: {winning_trades}")
            print(f"   Total P&L: {total_pnl:.4f}")
            print(f"   Average P&L: {total_pnl/len(trades):.4f}")
        else:
            print(f"   No trades executed!")
        
        return self
    
    def create_visualization(self, save_path=None):
        if not hasattr(self, 'plot_data'):
            print("❌ Please run analyze_pair() first!")
            return self
        
        data = self.plot_data
        stock1 = data['stock1']
        stock2 = data['stock2']
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 16), 
                                gridspec_kw={'height_ratios': [2, 2, 1, 1]},
                                sharex=True)
        
        fig.suptitle(f'📊 Trade Analysis: {stock1} vs {stock2} Pair Trading Strategy\n'
                     f'Why Do We Lose Money? (Cointegration p={data["coint_p_value"]:.4f})', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Stock Prices
        ax1 = axes[0]
        form_dates = data['formation_prices_1'].index
        ax1.plot(form_dates, data['formation_prices_1'], label=f'{stock1} (Formation)', 
                linewidth=2, alpha=0.8, color='blue')
        ax1.plot(form_dates, data['formation_prices_2'], label=f'{stock2} (Formation)', 
                linewidth=2, alpha=0.8, color='red')
        
        trade_dates = data['trading_prices_1'].index
        ax1.plot(trade_dates, data['trading_prices_1'], label=f'{stock1} (Trading)', 
                linewidth=2, alpha=1.0, color='darkblue')
        ax1.plot(trade_dates, data['trading_prices_2'], label=f'{stock2} (Trading)', 
                linewidth=2, alpha=1.0, color='darkred')
        
        if len(form_dates) > 0 and len(trade_dates) > 0:
            separator_date = trade_dates[0]
            ax1.axvline(x=separator_date, color='gray', linestyle='--', alpha=0.7, 
                       label='Formation→Trading')
        
        ax1.set_title('Stock Price Evolution (Log Prices)', fontweight='bold')
        ax1.set_ylabel('Log Price (Cumulative Log Returns)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z-Score with signals
        ax2 = axes[1]
        ax2.plot(trade_dates, data['trading_zscore'], label='Z-Score', 
                linewidth=2, color='purple', alpha=0.8)
        
        ax2.axhline(y=self.entry_zscore, color='red', linestyle='--', alpha=0.7, 
                   label=f'Entry Threshold (+{self.entry_zscore})')
        ax2.axhline(y=-self.entry_zscore, color='red', linestyle='--', alpha=0.7, 
                   label=f'Entry Threshold (-{self.entry_zscore})')
        ax2.axhline(y=self.exit_zscore, color='green', linestyle=':', alpha=0.7,
                   label=f'Exit Threshold (+{self.exit_zscore})')
        ax2.axhline(y=-self.exit_zscore, color='green', linestyle=':', alpha=0.7,
                   label=f'Exit Threshold (-{self.exit_zscore})')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        for trade in self.trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            entry_z = trade['entry_zscore']
            exit_z = trade['exit_zscore']
            
            ax2.scatter(entry_date, entry_z, color='red', s=100, marker='^', 
                       alpha=0.8, zorder=5)
            ax2.scatter(exit_date, exit_z, color='green', s=100, marker='v', 
                       alpha=0.8, zorder=5)
            
            pnl_color = 'green' if trade['pnl'] > 0 else 'red'
            ax2.annotate(f'P&L: {trade["pnl"]:.3f}', 
                        xy=(exit_date, exit_z), xytext=(10, 10),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=pnl_color, alpha=0.3))
        
        ax2.set_title('Z-Score Evolution and Trading Signals', fontweight='bold')
        ax2.set_ylabel('Z-Score')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position Holdings (like your image)
        ax3 = axes[2]
        position_series = pd.Series(0, index=trade_dates)
        
        for trade in self.trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            position_value = 1 if trade['position'] == 'Long' else -1
            
            mask = (position_series.index >= entry_date) & (position_series.index <= exit_date)
            position_series[mask] = position_value
        
        ax3.step(position_series.index, position_series.values, where='post', 
                linewidth=3, color='darkred', alpha=0.8)
        ax3.fill_between(position_series.index, 0, position_series.values, 
                        step='post', alpha=0.3, color='darkred')
        
        ax3.set_title('Position Holdings Over Time', fontweight='bold')
        ax3.set_ylabel('Position\n(+1=Long Spread, -1=Short Spread)')
        ax3.set_ylim(-1.2, 1.2)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Cumulative P&L
        ax4 = axes[3]
        
        if self.trades:
            pnl_series = []
            cumulative_pnl = 0
            pnl_dates = []
            
            for trade in self.trades:
                cumulative_pnl += trade['pnl']
                pnl_series.append(cumulative_pnl)
                pnl_dates.append(trade['exit_date'])
            
            ax4.plot(pnl_dates, pnl_series, marker='o', linewidth=2, 
                    color='red' if cumulative_pnl < 0 else 'green', 
                    markersize=6, alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax4.annotate(f'Final P&L: {cumulative_pnl:.3f}', 
                        xy=(pnl_dates[-1], cumulative_pnl), xytext=(10, 10),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='red' if cumulative_pnl < 0 else 'green', 
                                alpha=0.3))
        
        ax4.set_title('Cumulative P&L Over Time', fontweight='bold')
        ax4.set_ylabel('Cumulative P&L')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Visualization saved to: {save_path}")
        
        plt.show()
        
        if self.trades:
            print(f"\n📊 TRADE SUMMARY")
            print("="*50)
            for i, trade in enumerate(self.trades, 1):
                print(f"Trade {i}: {trade['position']} from {trade['entry_date'].strftime('%Y-%m-%d')} "
                      f"to {trade['exit_date'].strftime('%Y-%m-%d')}")
                print(f"   Entry Z: {trade['entry_zscore']:.3f}, Exit Z: {trade['exit_zscore']:.3f}")
                print(f"   P&L: {trade['pnl']:.4f}, Days held: {trade['days_held']}")
                print(f"   Result: {'✅ WIN' if trade['pnl'] > 0 else '❌ LOSS'}")
                print()
        
        return self


def main():
    parser = argparse.ArgumentParser(description='Trade Analysis Visualization')
    parser.add_argument('data_path', help='Path to pair trading data directory')
    parser.add_argument('--stock1', type=str, default='AMZN', help='First stock')
    parser.add_argument('--stock2', type=str, default='NKE', help='Second stock')
    parser.add_argument('--entry-zscore', type=float, default=2.0, help='Entry threshold')
    parser.add_argument('--exit-zscore', type=float, default=0.5, help='Exit threshold')
    
    args = parser.parse_args()
    
    try:
        analyzer = TradeAnalysisVisualization(
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore
        )
        
        analyzer.load_data(args.data_path)
        analyzer.analyze_pair(stock1=args.stock1, stock2=args.stock2)
        analyzer.create_visualization()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 