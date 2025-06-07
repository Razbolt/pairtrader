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
        
        # Clean and analyze coverage
        analysis_df = filtered_df[[date_col] + trading_cols].copy()
        analysis_df[trading_cols] = analysis_df[trading_cols].replace('.', np.nan)
        
        for col in trading_cols:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # 3. Select data type for correlation analysis
        print(f"\n🎯 Selecting data for correlation analysis...")
        
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
        
        # Filter for good quality columns (≥85% coverage)
        good_columns = []
        total_rows = len(analysis_df)
        
        for col in selected_columns:
            valid_count = analysis_df[col].notna().sum()
            coverage_pct = (valid_count / total_rows) * 100
            if coverage_pct >= 85.0:
                good_columns.append(col)
        
        print(f"   📊 Available {chosen_type} columns: {len(selected_columns)}")
        print(f"   🎯 High-quality columns (≥85% coverage): {len(good_columns)}")
        
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
        if chosen_type in ['log_returns', 'simple_returns']:
            print(f"   🔧 Preserving original {chosen_type} (no interpolation)")
        else:
            print(f"   🔧 Forward filling and interpolating prices")
            for data in [clean_data, in_sample_data, out_sample_data]:
                data[good_columns] = data[good_columns].fillna(method='ffill')
                data[good_columns] = data[good_columns].interpolate(method='linear')
        
        # Calculate final statistics
        in_sample_missing = in_sample_data[good_columns].isnull().sum().sum()
        out_sample_missing = out_sample_data[good_columns].isnull().sum().sum()
        total_missing = clean_data[good_columns].isnull().sum().sum()
        
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
            f.write(f"Missing Values: {total_missing}\n")
            f.write(f"In-Sample Missing: {in_sample_missing}\n")
            f.write(f"Out-Sample Missing: {out_sample_missing}\n\n")
            
            f.write(f"📊 AVAILABLE DATA TYPES\n")
            f.write(f"Log Returns (R_): {len(log_returns)} columns\n")
            f.write(f"Simple Returns (r_): {len(simple_returns)} columns\n")
            f.write(f"Prices (p_adjclose_): {len(prices)} columns\n\n")
            
            f.write(f"🔧 CORRELATION ANALYSIS GUIDANCE\n")
            f.write(f"✅ RECOMMENDED: Use log returns for correlation analysis\n")
            f.write(f"   - More stationary (better statistical properties)\n")
            f.write(f"   - Symmetric around zero\n")
            f.write(f"   - Time-additive: ln(P_t/P_0) = Σ log_returns\n")
            f.write(f"   - Standard in academic literature\n\n")
            f.write(f"⚠️  Alternative: Simple returns also work but less preferred\n")
            f.write(f"❌ NOT RECOMMENDED: Raw prices (non-stationary)\n\n")
            
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
        print(f"   3️⃣  For correlation: LOG RETURNS are recommended over simple returns")
        
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
   simple_returns: Alternative, less preferred
   prices:         Not recommended (non-stationary)

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