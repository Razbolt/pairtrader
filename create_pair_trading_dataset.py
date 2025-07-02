#!/usr/bin/env python3
"""
🔧 Enhanced Command-Line Data Cleaner for Pair Trading

Usage:
    python clean_data_enhanced.py sp500 2024-01-01 --in-sample 12 --out-sample 6
    python clean_data_enhanced.py Chinese_stocks 2024-01-01 --in-sample 8 --out-sample 4
    python clean_data_enhanced.py sp500 2023-01-01 --in-sample 6 --out-sample 3 --data-type simple_returns
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def clean_dataset(dataset_name: str, start_date: str, 
                 in_sample_months: int = 12, out_sample_months: int = 6,
                 data_type: str = 'auto') -> bool:
    """
    🚀 Clean dataset for pair trading with automatic end date calculation
    
    Args:
        dataset_name: Dataset name (sp500, Chinese_stocks, etc.)
        start_date: Start date (YYYY-MM-DD)
        in_sample_months: In-sample period for pair formation (default: 12)
        out_sample_months: Out-of-sample period for pair trading (default: 6)
        data_type: 'auto', 'log_returns', 'simple_returns', or 'prices'
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"🔧 CLEANING {dataset_name.upper()} FOR PAIR TRADING")
    print("="*70)
    print(f"📊 Dataset: {dataset_name}")
    print(f"📅 Start Date: {start_date}")
    print(f"📈 In-Sample (Pair Formation): {in_sample_months} months")
    print(f"💰 Out-Sample (Pair Trading): {out_sample_months} months")
    print(f"📊 Data Type: {data_type}")
    
    try:
        # 1. Load all data first
        print(f"\n📁 Loading data...")
        csv_path = Path(f"data/converted_csv/{dataset_name}.csv")
        if not csv_path.exists():
            print(f"❌ Error: Dataset '{dataset_name}' not found in data/converted_csv/")
            return False
        
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"   ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
        
        # Find date column
        date_col = None
        for col in ['date', 'period', 'timestamp']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            print(f"❌ Error: No date column found")
            return False
        
        # Convert dates
        df[date_col] = pd.to_datetime(df[date_col])
        start_dt = pd.to_datetime(start_date)
        
        # Calculate required trading days (approximately 21 per month)
        total_months = in_sample_months + out_sample_months
        required_trading_days = total_months * 21
        
        print(f"   📊 Total period needed: {total_months} months (~{required_trading_days} trading days)")
        
        # Filter data starting from start_date
        available_data = df[df[date_col] >= start_dt].copy().reset_index(drop=True)
        
        if len(available_data) == 0:
            print(f"❌ Error: No data found starting from {start_date}")
            return False
        
        print(f"   📅 Available from start date: {len(available_data):,} rows")
        print(f"   📅 Date range available: {available_data[date_col].min().strftime('%Y-%m-%d')} to {available_data[date_col].max().strftime('%Y-%m-%d')}")
        
        # Check if we have enough data
        if len(available_data) < required_trading_days:
            available_months = len(available_data) / 21
            print(f"   ⚠️  Warning: Only {len(available_data)} days available (~{available_months:.1f} months)")
            print(f"   💡 Requested: {total_months} months")
            max_in_sample = int(available_months * 0.67)
            max_out_sample = int(available_months * 0.33)
            print(f"   📝 Suggestion: Try --in-sample {max_in_sample} --out-sample {max_out_sample}")
            
            # Ask user if they want to continue with available data
            use_available = input("   🤔 Use all available data? (y/n): ").lower().startswith('y')
            if not use_available:
                return False
            
            # Use all available data
            filtered_df = available_data.copy()
        else:
            # Take exactly the required amount of data
            filtered_df = available_data.head(required_trading_days).copy()
        
        # Calculate actual end date
        actual_end_date = filtered_df[date_col].max()
        print(f"   ✅ Using data: {filtered_df[date_col].min().strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}")
        print(f"   📊 Actual period: {len(filtered_df)} trading days")
        
        # 2. Find and categorize trading columns
        print(f"\n🔍 Analyzing data structure...")

        # ENHANCED LOGIC FOR CRYPTO DATASETS
        if 'crypto' in dataset_name.lower():
            print("   🔬 Applying specialized crypto data logic...")
            all_cols = filtered_df.columns
            tickers = sorted(list(set([c.split('_')[1] for c in all_cols if '_' in c and c.split('_')[1] != ''])))
            
            price_cols = [f'close_{ticker}' for ticker in tickers if f'close_{ticker}' in all_cols]
            volume_cols = [f'volume_{ticker}' for ticker in tickers if f'volume_{ticker}' in all_cols]
            
            print(f"   🪙 Found {len(tickers)} unique tickers: {', '.join(tickers)}")
            
            # Select and rename columns
            analysis_df = filtered_df[[date_col] + price_cols + volume_cols].copy()
            
            rename_map = {}
            for p_col in price_cols:
                ticker = p_col.split('_')[1]
                rename_map[p_col] = f'p_adjclose_{ticker}' # Standardize to p_adjclose_
            for v_col in volume_cols:
                ticker = v_col.split('_')[1]
                rename_map[v_col] = f'v_{ticker}'
            
            analysis_df.rename(columns=rename_map, inplace=True)
            
            # Update internal column lists
            trading_cols = list(rename_map.values())
            prices = [col for col in trading_cols if col.startswith('p_adjclose_')]
            selected_columns = prices
            chosen_type = 'prices'
            log_returns, simple_returns, raw_price_columns = [], [], []

            print(f"   ✅ Selected {len(prices)} price columns and {len(volume_cols)} volume columns.")

        else: # Original logic for other datasets
            trading_cols = []
            for col in filtered_df.columns:
                if col != date_col and not any(x in col.lower() for x in ['volume', 'vol', 'timestamp', 'daten']):
                    if (col.startswith('R_') or col.startswith('r_') or 
                        'adjclose' in col or col.lower() in ['oil', 'gold', 'copper', 'wheat', 'nyse', 'nasdaq'] or
                        any(x in col for x in ['open_', 'high_', 'low_', 'close_'])):
                        trading_cols.append(col)
            
            # Categorize columns by type
            log_returns = [col for col in trading_cols if col.startswith('R_')]
            simple_returns = [col for col in trading_cols if col.startswith('r_') and not col.startswith('R_')]
            prices = [col for col in trading_cols if col not in log_returns + simple_returns]
            
            print(f"   📊 Total trading columns: {len(trading_cols)}")
            print(f"   📈 Log returns (R_): {len(log_returns)}")
            print(f"   📊 Simple returns (r_): {len(simple_returns)}")
            print(f"   💰 Prices (p_adjclose_): {len(prices)}")
            
            analysis_df = filtered_df[[date_col] + trading_cols].copy()

        # Clean and analyze coverage
        analysis_df[trading_cols] = analysis_df[trading_cols].replace('.', np.nan)
        
        for col in trading_cols:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # 3. Select data type for analysis and prepare raw prices if needed
        print(f"\n🎯 Selecting data for analysis...")
        
        if 'crypto' not in dataset_name.lower(): # Skip this section for crypto as it's already handled
            if data_type == 'auto':
                if log_returns:
                    selected_columns = log_returns
                    chosen_type = 'log_returns'
                    print(f"   ✅ Auto-selected: Log returns (best for correlation analysis)")
                elif simple_returns:
                    selected_columns = simple_returns
                    chosen_type = 'simple_returns'
                    print(f"   ✅ Auto-selected: Simple returns")
                else:
                    selected_columns = prices
                    chosen_type = 'prices'
                    print(f"   ✅ Auto-selected: Prices (will compute returns)")
            elif data_type == 'log_returns':
                if not log_returns:
                    print(f"❌ Error: No log returns columns found")
                    return False
                selected_columns = log_returns
                chosen_type = 'log_returns'
            elif data_type == 'simple_returns':
                if not simple_returns:
                    print(f"❌ Error: No simple returns columns found")
                    return False
                selected_columns = simple_returns
                chosen_type = 'simple_returns'
            elif data_type == 'prices':
                if not prices:
                    print(f"❌ Error: No price columns found")
                    return False
                selected_columns = prices
                chosen_type = 'prices'
            else:
                print(f"❌ Error: Invalid data_type. Use 'auto', 'log_returns', 'simple_returns', or 'prices'")
                return False
        
        # 📊 ENHANCEMENT: Handle raw prices for cointegration testing
        if 'crypto' not in dataset_name.lower(): # Skip this for crypto
            raw_price_columns = []
        
            if chosen_type == 'log_returns' and log_returns:
                print(f"   🔄 Converting log returns to raw prices for cointegration analysis...")
                
                # Convert log returns to cumulative price levels (starting from 100)
                for col in log_returns:
                    ticker = col.replace('R_', '')
                    raw_price_col = f'P_{ticker}'
                    raw_price_columns.append(raw_price_col)
                    
                    # Convert log returns to price levels: P_t = 100 * exp(cumsum(log_returns))
                    log_ret_series = analysis_df[col].fillna(0)  # Fill NaN with 0 for cumsum
                    analysis_df[raw_price_col] = 100 * np.exp(log_ret_series.cumsum())
                
                print(f"   ✅ Created {len(raw_price_columns)} raw price series (P_TICKER format)")
                print(f"   💡 Raw prices can be used for: Cointegration testing, stationarity analysis")
                print(f"   💡 Log returns remain for: Correlation analysis, mean reversion strategies")
            
                # Update selected_columns to use the new standardized names
                selected_columns = raw_price_columns
            
            elif chosen_type == 'prices' and prices:
                print(f"   🔄 Converting existing prices to standardized raw price format...")
                
                # Convert existing price columns to p_adjclose_TICKER format for consistency
                for col in prices:
                    # Extract ticker name from various formats
                    if 'adjclose' in col.lower():
                        ticker = col.replace('p_adjclose_', '').replace('adjclose_', '').replace('_adjclose', '')
                    elif col.lower() in ['oil', 'gold', 'copper', 'wheat', 'nyse', 'nasdaq']:
                        ticker = col.upper()
                    else:
                        # Try to extract ticker from column name
                        ticker = col.replace('close_', '').replace('_close', '').replace('open_', '').replace('high_', '').replace('low_', '')
                    
                    # Standardize to p_adjclose_TICKER format
                    raw_price_col = f'p_adjclose_{ticker}'
                    raw_price_columns.append(raw_price_col)
                    
                    # Copy the price data with standardized column name
                    analysis_df[raw_price_col] = analysis_df[col].copy()
                
                print(f"   ✅ Standardized {len(raw_price_columns)} raw price series (p_adjclose_TICKER format)")
                print(f"   💡 Raw prices ready for: Direct cointegration testing, true price level analysis")
                
                # Update selected_columns to use the new standardized names
                selected_columns = raw_price_columns
        
        # Filter for good quality columns (≥85% coverage)
        good_columns = []
        good_raw_price_columns = []
        total_rows = len(analysis_df)
        
        all_cols_to_check = selected_columns
        if 'crypto' in dataset_name.lower():
            # For crypto, we check all p_ and v_ columns
            all_cols_to_check = trading_cols

        for col in all_cols_to_check:
            valid_count = analysis_df[col].notna().sum()
            coverage_pct = (valid_count / total_rows) * 100
            if coverage_pct >= 85.0:
                good_columns.append(col)
        
        if 'crypto' in dataset_name.lower():
            selected_columns = [c for c in good_columns if c.startswith('p_adjclose_')]
            print(f"   🎯 High-quality price columns (≥85% coverage): {len(selected_columns)}")
            volume_cols = [c for c in good_columns if c.startswith('v_')]
            print(f"   📊 High-quality volume columns (≥85% coverage): {len(volume_cols)}")
            good_columns.sort() # for consistency
        else:
            # Handle raw price columns based on data type
            if raw_price_columns:
                if chosen_type == 'prices':
                    # When using prices, raw_price_columns ARE the main columns (no duplication needed)
                    good_raw_price_columns = []  # Don't duplicate - good_columns already contains prices
                else:
                    # When using log_returns, raw_price_columns are additional
                    for col in raw_price_columns:
                        valid_count = analysis_df[col].notna().sum()
                        coverage_pct = (valid_count / total_rows) * 100
                        if coverage_pct >= 85.0:
                            good_raw_price_columns.append(col)
            
            print(f"   📊 Available {chosen_type} columns: {len(selected_columns)}")
            print(f"   🎯 High-quality columns (≥85% coverage): {len(good_columns)}")
        
            if good_raw_price_columns:
                print(f"   💰 High-quality raw price columns: {len(good_raw_price_columns)}")
        
        if len(good_columns) < 5:
            print(f"   ⚠️  Warning: Only {len(good_columns)} good columns found")
        
        # 4. Create exact in-sample / out-of-sample periods
        print(f"\n📅 Creating exact periods...")
        
        total_available_days = len(analysis_df)
        in_sample_ratio = in_sample_months / (in_sample_months + out_sample_months)
        
        in_sample_actual = int(total_available_days * in_sample_ratio)
        out_sample_actual = total_available_days - in_sample_actual
        
        print(f"   📊 Total days: {total_available_days}")
        print(f"   📏 Allocated: {in_sample_actual} days in-sample + {out_sample_actual} days out-sample")
        
        # Split data SEQUENTIALLY (this is crucial for pair trading)
        columns_to_use = [date_col] + good_columns
        if good_raw_price_columns:
            columns_to_use.extend(good_raw_price_columns)
        
        clean_data = analysis_df[columns_to_use].copy()
        
        # IN-SAMPLE: First period for pair formation
        in_sample_data = clean_data.iloc[:in_sample_actual].copy()
        
        # OUT-OF-SAMPLE: Second period for pair trading (AFTER in-sample)
        out_sample_data = clean_data.iloc[in_sample_actual:in_sample_actual + out_sample_actual].copy()
        
        print(f"\n🔍 Period Details:")
        print(f"   📈 In-Sample Period: {in_sample_data[date_col].min().strftime('%Y-%m-%d')} to {in_sample_data[date_col].max().strftime('%Y-%m-%d')}")
        print(f"   💰 Out-Sample Period: {out_sample_data[date_col].min().strftime('%Y-%m-%d')} to {out_sample_data[date_col].max().strftime('%Y-%m-%d')}")
        print(f"   📊 Actual months: {len(in_sample_data)/21:.1f} in-sample + {len(out_sample_data)/21:.1f} out-sample")
        
        # Apply cleaning strategy
        all_cols_to_clean = good_columns
        if 'crypto' not in dataset_name.lower() and chosen_type in ['log_returns', 'simple_returns']:
             print(f"   🔧 Preserving original {chosen_type} (no interpolation)")
             if good_raw_price_columns:
                 print(f"   🔧 Interpolating {len(good_raw_price_columns)} raw price columns")
                 all_cols_to_clean = good_raw_price_columns
             else:
                 all_cols_to_clean = [] # Nothing to clean
        
        if all_cols_to_clean:
            print(f"   🔧 Forward filling and interpolating {len(all_cols_to_clean)} columns")
            for col in all_cols_to_clean:
                clean_data[col] = clean_data[col].fillna(method='ffill')
                clean_data[col] = clean_data[col].interpolate(method='linear')

        # Re-slice data after cleaning to ensure no SettingWithCopyWarning
        in_sample_data = clean_data.iloc[:in_sample_actual].copy()
        out_sample_data = clean_data.iloc[in_sample_actual:in_sample_actual + out_sample_actual].copy()

        # Calculate final statistics
        all_analysis_columns = good_columns + good_raw_price_columns
        in_sample_missing = in_sample_data[all_analysis_columns].isnull().sum().sum()
        out_sample_missing = out_sample_data[all_analysis_columns].isnull().sum().sum()
        total_missing = clean_data[all_analysis_columns].isnull().sum().sum()
        
        print(f"   📊 In-sample missing: {in_sample_missing:,}")
        print(f"   📊 Out-sample missing: {out_sample_missing:,}")
        print(f"   📊 Total missing: {total_missing:,}")
        
        # 5. Save results with clear naming
        print(f"\n💾 Saving results...")
        
        # Create output directory with auto-calculated end date
        start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_str = actual_end_date.strftime('%Y%m%d')
        output_name = f"{dataset_name}_{start_str}_{end_str}_{chosen_type}_{in_sample_months}m{out_sample_months}m"
        
        output_dir = Path("data/pair_trading") / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files with clear names
        full_path = output_dir / f"{output_name}_complete_dataset.csv"
        in_sample_path = output_dir / f"{output_name}_in_sample_formation.csv"
        out_sample_path = output_dir / f"{output_name}_out_sample_trading.csv"
        
        clean_data.to_csv(full_path, index=False)
        in_sample_data.to_csv(in_sample_path, index=False)
        out_sample_data.to_csv(out_sample_path, index=False)
        
        # Save detailed metadata
        metadata_path = output_dir / f"{output_name}_info.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"PAIR TRADING DATASET: {dataset_name.upper()}\n")
            f.write("="*60 + "\n\n")
            f.write(f"📊 DATASET INFORMATION\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Data Type: {chosen_type}\n")
            f.write(f"Start Date: {start_date}\n")
            f.write(f"Auto-calculated End Date: {actual_end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Requested Periods: {in_sample_months} months in-sample + {out_sample_months} months out-sample\n\n")
            
            f.write(f"📈 PERIOD STRUCTURE (Sequential)\n")
            f.write(f"In-Sample (Pair Formation): {in_sample_data[date_col].min().strftime('%Y-%m-%d')} to {in_sample_data[date_col].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Out-Sample (Pair Trading): {out_sample_data[date_col].min().strftime('%Y-%m-%d')} to {out_sample_data[date_col].max().strftime('%Y-%m-%d')}\n")
            f.write(f"In-Sample Days: {len(in_sample_data)}\n")
            f.write(f"Out-Sample Days: {len(out_sample_data)}\n")
            f.write(f"Actual Months: {len(in_sample_data)/21:.1f} in-sample + {len(out_sample_data)/21:.1f} out-sample\n\n")
            
            f.write(f"🎯 DATA QUALITY\n")
            f.write(f"Total Assets: {len(good_columns)}\n")
            if good_raw_price_columns:
                f.write(f"Raw Price Series: {len(good_raw_price_columns)}\n")
            f.write(f"Missing Values: {total_missing}\n")
            f.write(f"In-Sample Missing: {in_sample_missing}\n")
            f.write(f"Out-Sample Missing: {out_sample_missing}\n\n")
            
            f.write(f"📊 AVAILABLE DATA TYPES\n")
            f.write(f"Log Returns (R_): {len(log_returns)} columns\n")
            f.write(f"Simple Returns (r_): {len(simple_returns)} columns\n")
            f.write(f"Prices (p_adjclose_): {len(prices)} columns\n")
            if good_raw_price_columns:
                f.write(f"Raw Prices (P_): {len(good_raw_price_columns)} columns (converted from log returns)\n")
            f.write(f"\n")
            
            f.write(f"🔧 ANALYSIS GUIDANCE\n")
            f.write(f"✅ CORRELATION ANALYSIS: Use log returns (R_ columns)\n")
            f.write(f"   - More stationary (better statistical properties)\n")
            f.write(f"   - Symmetric around zero\n")
            f.write(f"   - Time-additive: ln(P_t/P_0) = Σ log_returns\n")
            f.write(f"   - Standard in academic literature\n\n")
            if good_raw_price_columns:
                f.write(f"✅ COINTEGRATION ANALYSIS: Use raw prices (P_ columns)\n")
                f.write(f"   - Test non-stationary price series for long-term equilibrium\n")
                f.write(f"   - Engle-Granger test requires actual price levels\n")
                f.write(f"   - Generated from: P_t = 100 * exp(cumsum(log_returns))\n\n")
            f.write(f"⚠️  Alternative: Simple returns also work but less preferred\n")
            f.write(f"❌ AVOID: Using raw prices for correlation (non-stationary)\n\n")
            
            f.write("📈 SELECTED ASSETS:\n")
            for i, col in enumerate(good_columns, 1):
                asset_name = col.replace('R_', '').replace('r_', '').replace('p_adjclose_', '')
                f.write(f"{i:3d}. {asset_name}\n")
        
        print(f"   📁 Saved to: {output_dir}")
        print(f"   📊 Complete dataset: {clean_data.shape}")
        print(f"   📈 In-sample: {in_sample_data.shape}")
        print(f"   💰 Out-sample: {out_sample_data.shape}")
        
        # Success message with guidance
        print(f"\n✅ SUCCESS!")
        if total_missing == 0:
            print(f"🏆 PERFECT DATA QUALITY - Ready for pair trading!")
        elif total_missing < 100:
            print(f"✅ EXCELLENT DATA QUALITY - Ready for pair trading!")
        else:
            print(f"⚠️  Some missing values - Check data quality")
        
        print(f"\n🎯 NEXT STEPS FOR PAIR TRADING:")
        print(f"   1️⃣  Use IN-SAMPLE data to find correlations & select pairs")
        print(f"   2️⃣  Use OUT-SAMPLE data to backtest/trade the pairs")
        print(f"   3️⃣  For correlation: Use LOG RETURNS (R_ columns)")
        if good_raw_price_columns:
            print(f"   4️⃣  For cointegration: Use RAW PRICES (P_ columns)")
            print(f"   💡 Now you can test both correlation AND cointegration approaches!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Enhanced command-line interface with automatic end date calculation"""
    parser = argparse.ArgumentParser(
        description='Enhanced data cleaner for pair trading with automatic period calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 AUTOMATIC PERIOD CALCULATION:
   Provide start date + desired periods → End date calculated automatically
   This ensures you get exactly the periods you request!
   
🔧 DATA TYPE RECOMMENDATIONS:
   log_returns:    Best for correlation analysis (recommended)
   simple_returns: Alternative for returns-based analysis
   prices:         Required for cointegration analysis (raw prices)

📊 Examples:
  python clean_data_enhanced.py sp500 2024-01-01 --in-sample 12 --out-sample 6
  python clean_data_enhanced.py Chinese_stocks 2024-01-01 --in-sample 8 --out-sample 4
  python clean_data_enhanced.py sp500 2023-01-01 --in-sample 6 --out-sample 3 --data-type log_returns
  python clean_data_enhanced.py FTSE100 2023-06-01 --in-sample 9 --out-sample 3
        """
    )
    
    parser.add_argument('dataset', help='Dataset name (sp500, FTSE100, Chinese_stocks, etc.)')
    parser.add_argument('start_date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--in-sample', '-i', type=int, default=12, 
                       help='In-sample period in months for pair formation (default: 12)')
    parser.add_argument('--out-sample', '-o', type=int, default=6, 
                       help='Out-of-sample period in months for pair trading (default: 6)')
    parser.add_argument('--data-type', '-d', choices=['auto', 'log_returns', 'simple_returns', 'prices'], 
                       default='auto', help='Data type for correlation analysis (default: auto)')
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print(f"\n🔍 Available datasets:")
        csv_dir = Path("data/converted_csv")
        if csv_dir.exists():
            datasets = [f.stem for f in csv_dir.glob("*.csv")]
            for dataset in sorted(datasets):
                print(f"   • {dataset}")
        else:
            print(f"   No datasets found in data/converted_csv/")
        return
    
    args = parser.parse_args()
    
    # Clean the dataset
    success = clean_dataset(
        dataset_name=args.dataset,
        start_date=args.start_date,
        in_sample_months=args.in_sample,
        out_sample_months=args.out_sample,
        data_type=args.data_type
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 