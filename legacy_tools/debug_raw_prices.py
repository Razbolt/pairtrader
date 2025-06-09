#!/usr/bin/env python3
"""
🐛 Debug Raw Prices Trading Issue

This script debugs why no trades are being executed in the raw prices strategy
despite having high Z-scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')


def debug_trading_logic(data_path):
    """🔍 Debug the trading logic step by step"""
    print("🐛 DEBUGGING TRADING LOGIC")
    print("="*60)
    
    data_path = Path(data_path)
    
    # Load data
    for file in data_path.glob("*_in_sample_formation.csv"):
        formation_file = file
    for file in data_path.glob("*_out_sample_trading.csv"):
        trading_file = file
    
    formation_data = pd.read_csv(formation_file)
    trading_data = pd.read_csv(trading_file)
    
    # Set date columns
    for df in [formation_data, trading_data]:
        date_col = 'period' if 'period' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    
    # Get price columns
    price_columns = [col for col in formation_data.columns if col.startswith('p_adjclose_')]
    formation_prices = formation_data[price_columns]
    trading_prices = trading_data[price_columns]
    
    # Test with first few stocks to find a cointegrated pair
    test_stocks = price_columns[:20]
    print(f"🔍 Testing {len(test_stocks)} stocks for cointegration...")
    
    pair_found = None
    for i, stock1 in enumerate(test_stocks):
        for j, stock2 in enumerate(test_stocks):
            if i < j:
                # Test cointegration quickly
                s1 = formation_prices[stock1].dropna()
                s2 = formation_prices[stock2].dropna()
                common_dates = s1.index.intersection(s2.index)
                
                if len(common_dates) < 30:
                    continue
                
                s1_clean = s1[common_dates]
                s2_clean = s2[common_dates]
                
                try:
                    coint_stat, p_value, critical_values = coint(s1_clean, s2_clean)
                    if p_value < 0.05:
                        ols_result = OLS(s1_clean, s2_clean).fit()
                        hedge_ratio = ols_result.params[0]
                        
                        pair_found = {
                            'stock1': stock1,
                            'stock2': stock2,
                            'stock1_name': stock1.replace('p_adjclose_', ''),
                            'stock2_name': stock2.replace('p_adjclose_', ''),
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value
                        }
                        print(f"✅ Found pair: {pair_found['stock1_name']}-{pair_found['stock2_name']}")
                        print(f"   p-value: {p_value:.4f}, hedge ratio: {hedge_ratio:.3f}")
                        break
                except:
                    continue
            
        if pair_found:
            break
    
    if not pair_found:
        print("❌ No cointegrated pair found in sample")
        return
    
    # Now debug the trading logic with this pair
    print(f"\n🔍 DEBUGGING TRADING LOGIC FOR {pair_found['stock1_name']}-{pair_found['stock2_name']}")
    print("="*60)
    
    stock1 = pair_found['stock1']
    stock2 = pair_found['stock2'] 
    hedge_ratio = pair_found['hedge_ratio']
    
    # Step 1: Formation spread calculation
    print("📊 Step 1: Formation period spread calculation")
    formation_s1 = formation_prices[stock1].dropna()
    formation_s2 = formation_prices[stock2].dropna()
    common_formation = formation_s1.index.intersection(formation_s2.index)
    
    print(f"   Formation {pair_found['stock1_name']}: {len(formation_s1)} days")
    print(f"   Formation {pair_found['stock2_name']}: {len(formation_s2)} days")
    print(f"   Common formation dates: {len(common_formation)} days")
    
    formation_spread = formation_s1[common_formation] - hedge_ratio * formation_s2[common_formation]
    spread_mean = formation_spread.mean()
    spread_std = formation_spread.std()
    
    print(f"   Spread mean: {spread_mean:.2f}")
    print(f"   Spread std: {spread_std:.2f}")
    
    # Step 2: Trading spread calculation - THIS IS WHERE THE BUG IS
    print(f"\n📊 Step 2: Trading period spread calculation")
    trading_s1 = trading_prices[stock1].dropna()
    trading_s2 = trading_prices[stock2].dropna()
    common_trading = trading_s1.index.intersection(trading_s2.index)
    
    print(f"   Trading {pair_found['stock1_name']}: {len(trading_s1)} days")
    print(f"   Trading {pair_found['stock2_name']}: {len(trading_s2)} days") 
    print(f"   Common trading dates: {len(common_trading)} days")
    
    # Calculate Z-scores using common dates
    trading_spread = trading_s1[common_trading] - hedge_ratio * trading_s2[common_trading]
    trading_zscore = (trading_spread - spread_mean) / spread_std
    
    print(f"   Z-score range: [{trading_zscore.min():.3f}, {trading_zscore.max():.3f}]")
    print(f"   Max |Z-score|: {max(abs(trading_zscore.min()), abs(trading_zscore.max())):.3f}")
    
    # Step 3: Debug the trading execution
    print(f"\n🎯 Step 3: Trading execution debug")
    entry_zscore = 0.5  # Low threshold
    exit_zscore = 0.0
    
    print(f"   Entry threshold: ±{entry_zscore}")
    print(f"   Exit threshold: ±{exit_zscore}")
    
    # THIS IS THE BUG: Using wrong price series!
    print(f"\n🐛 REPRODUCING THE BUG:")
    print(f"   Z-score index dates: {len(trading_zscore.index)} dates")
    print(f"   trading_s1 index dates: {len(trading_s1.index)} dates")
    print(f"   trading_s2 index dates: {len(trading_s2.index)} dates")
    
    # Check date alignment
    zscore_dates = set(trading_zscore.index)
    s1_dates = set(trading_s1.index)
    s2_dates = set(trading_s2.index)
    
    missing_s1 = zscore_dates - s1_dates
    missing_s2 = zscore_dates - s2_dates
    
    print(f"   Z-score dates missing from S1: {len(missing_s1)}")
    print(f"   Z-score dates missing from S2: {len(missing_s2)}")
    
    if missing_s1 or missing_s2:
        print(f"   🐛 BUG CONFIRMED: Date mismatch prevents trading!")
        print(f"      Z-scores calculated on common dates")
        print(f"      But trading logic checks individual stock dates")
        
        # Show first few mismatched dates
        if missing_s1:
            print(f"      Missing S1 dates: {list(missing_s1)[:3]}...")
        if missing_s2:
            print(f"      Missing S2 dates: {list(missing_s2)[:3]}...")
    
    # Now fix it and show trades
    print(f"\n✅ FIXING THE BUG:")
    print(f"   Using aligned price series for trading execution")
    
    # Fixed version: use common dates for prices too
    trading_s1_aligned = trading_s1[common_trading]
    trading_s2_aligned = trading_s2[common_trading]
    
    print(f"   Aligned S1 dates: {len(trading_s1_aligned)}")
    print(f"   Aligned S2 dates: {len(trading_s2_aligned)}")
    print(f"   Z-score dates: {len(trading_zscore)}")
    
    # Execute trades with fixed logic
    trades = []
    position = 0
    entry_info = {}
    
    for date in trading_zscore.index:
        zscore = trading_zscore[date]
        p1 = trading_s1_aligned[date]
        p2 = trading_s2_aligned[date]
        
        if position == 0:  # No position
            if zscore > entry_zscore:  # Short spread
                position = -1
                entry_info = {
                    'entry_date': date,
                    'entry_zscore': zscore,
                    'entry_p1': p1,
                    'entry_p2': p2,
                    'position_type': 'Short Spread'
                }
            elif zscore < -entry_zscore:  # Long spread
                position = 1
                entry_info = {
                    'entry_date': date,
                    'entry_zscore': zscore,
                    'entry_p1': p1,
                    'entry_p2': p2,
                    'position_type': 'Long Spread'
                }
        
        else:  # In position, check exit
            if abs(zscore) <= exit_zscore:
                # Calculate P&L
                if position == 1:  # Long spread position
                    pnl_p1 = p1 - entry_info['entry_p1']
                    pnl_p2 = hedge_ratio * (entry_info['entry_p2'] - p2)
                    gross_pnl = pnl_p1 + pnl_p2
                else:  # Short spread position
                    pnl_p1 = entry_info['entry_p1'] - p1
                    pnl_p2 = hedge_ratio * (p2 - entry_info['entry_p2'])
                    gross_pnl = pnl_p1 + pnl_p2
                
                trades.append({
                    'entry_date': entry_info['entry_date'],
                    'exit_date': date,
                    'entry_zscore': entry_info['entry_zscore'],
                    'exit_zscore': zscore,
                    'position_type': entry_info['position_type'],
                    'gross_pnl': gross_pnl,
                    'days_held': (date - entry_info['entry_date']).days
                })
                
                position = 0
                entry_info = {}
    
    print(f"\n🎯 TRADING RESULTS WITH FIX:")
    print(f"   Total trades executed: {len(trades)}")
    
    if trades:
        total_pnl = sum(trade['gross_pnl'] for trade in trades)
        winning_trades = sum(1 for trade in trades if trade['gross_pnl'] > 0)
        
        print(f"   Total gross P&L: ${total_pnl:.2f}")
        print(f"   Winning trades: {winning_trades} ({winning_trades/len(trades)*100:.1f}%)")
        
        print(f"\n   First 5 trades:")
        for i, trade in enumerate(trades[:5]):
            print(f"   {i+1}. {trade['position_type']} | "
                  f"Entry Z: {trade['entry_zscore']:.2f} | "
                  f"Exit Z: {trade['exit_zscore']:.2f} | "
                  f"P&L: ${trade['gross_pnl']:.2f} | "
                  f"Days: {trade['days_held']}")
    else:
        print(f"   ❌ Still no trades - pair might be too stable")


def main():
    parser = argparse.ArgumentParser(description='Debug raw prices trading logic')
    parser.add_argument('data_path', help='Path to data directory')
    
    args = parser.parse_args()
    debug_trading_logic(args.data_path)


if __name__ == "__main__":
    main() 