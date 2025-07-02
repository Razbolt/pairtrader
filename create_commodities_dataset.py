#!/usr/bin/env python3
"""
Create Commodities Dataset for Pair Trading
- Load commodities data from 2023-06 onwards
- Create 12-month in-sample formation period
- Create 6-month out-sample trading period
- Convert to log returns for pair trading analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta

def create_commodities_dataset(start_date='2018-01-01', formation_months=48, trading_months=12):
    """
    Main function to create commodities dataset
    """
    print("🛢️  COMMODITIES PAIR TRADING DATASET CREATION")
    print("="*60)

    # Convert string date to datetime object
    start_date_dt = pd.to_datetime(start_date)

    # Load and filter data
    print("📊 Loading commodities data...")
    df = pd.read_csv('data/converted_csv/commodities.csv', parse_dates=['date'])
    print(f"   Original data: {len(df)} rows ({df['date'].min()} to {df['date'].max()})")
    
    df_filtered = df[df['date'] >= start_date_dt].copy()
    print(f"   Filtered data: {len(df_filtered)} rows ({df_filtered['date'].min()} to {df_filtered['date'].max()})")
    
    # Clean and prepare price data
    print(f"\n🔧 Preparing price data...")
    
    # Convert object columns to numeric, handling missing values
    numeric_columns = ['oil', 'copper', 'gold', 'oil_brent', 'aluminium', 'wheat', 'cocoa', 'sugar', 'nickel', 'platinum']
    
    for col in numeric_columns:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Set date as index
    df_filtered.set_index('date', inplace=True)
    
    # Forward fill missing values (commodities often have gaps)
    df_filtered = df_filtered.fillna(method='ffill')
    
    # Remove any remaining rows with NaN values
    df_filtered = df_filtered.dropna()
    
    print(f"   Clean data: {len(df_filtered)} rows")
    print(f"   Available commodities: {list(df_filtered.columns)}")
    
    # Calculate log returns
    print(f"\n📈 Calculating log returns...")
    log_returns = np.log(df_filtered / df_filtered.shift(1))
    log_returns = log_returns.dropna()
    
    # Convert back to prices for cointegration analysis (using log returns as basis)
    # This ensures we have proper price levels for cointegration testing
    prices_from_returns = np.exp(log_returns.cumsum())
    
    # Split into formation and trading periods
    formation_end_date = start_date_dt + pd.DateOffset(months=formation_months)
    trading_end_date = formation_end_date + pd.DateOffset(months=trading_months)
    
    formation_data = prices_from_returns[prices_from_returns.index <= formation_end_date]
    trading_data = prices_from_returns[(prices_from_returns.index > formation_end_date) & 
                                     (prices_from_returns.index <= trading_end_date)]
    
    print(f"\n📅 Dataset periods:")
    print(f"   Formation: {formation_data.index.min()} to {formation_data.index.max()} ({len(formation_data)} days)")
    print(f"   Trading: {trading_data.index.min()} to {trading_data.index.max()} ({len(trading_data)} days)")
    
    # Create output directory
    output_dir_name = f"commodities_{start_date_dt.strftime('%Y%m%d')}_{formation_months}m{trading_months}m"
    output_dir = Path("data/pair_trading") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    formation_file_path = output_dir / f"{output_dir_name}_in_sample_formation.csv"
    trading_file_path = output_dir / f"{output_dir_name}_out_sample_trading.csv"
    
    # Reset index to turn the 'date' index into a column, matching other scripts' format
    formation_data.reset_index(inplace=True)
    trading_data.reset_index(inplace=True)

    formation_data.to_csv(formation_file_path, index=False)
    trading_data.to_csv(trading_file_path, index=False)

    print(f"\n💾 Saved formation data: {formation_file_path}")
    print(f"💾 Saved trading data: {trading_file_path}")
    
    # Create summary statistics
    print("\n📊 Commodities Summary Statistics:")
    
    # Select only numeric columns for statistics
    numeric_formation_data = formation_data.select_dtypes(include=np.number)
    numeric_trading_data = trading_data.select_dtypes(include=np.number)

    print("   Formation period statistics:")
    for col in numeric_formation_data.columns:
        mean_val = numeric_formation_data[col].mean()
        std_val = numeric_formation_data[col].std()
        print(f"     {col}: Mean={mean_val:.2f}, Std={std_val:.2f}")

    print("\n   Trading period statistics:")
    for col in numeric_trading_data.columns:
        mean_val = numeric_trading_data[col].mean()
        std_val = numeric_trading_data[col].std()
        print(f"     {col}: Mean={mean_val:.2f}, Std={std_val:.2f}")

    print("\n✅ Commodities dataset created successfully!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📈 Ready for pair trading analysis")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Create commodities dataset for pair trading')
    parser.add_argument('--start-date', default='2023-06-01', help='Start date for dataset')
    parser.add_argument('--formation-months', type=int, default=12, help='Formation period in months')
    parser.add_argument('--trading-months', type=int, default=6, help='Trading period in months')
    
    args = parser.parse_args()
    
    create_commodities_dataset(
        start_date=args.start_date,
        formation_months=args.formation_months,
        trading_months=args.trading_months
    )

if __name__ == "__main__":
    main() 