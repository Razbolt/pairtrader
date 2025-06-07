import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def list_available_datasets():
    """List all available converted datasets"""
    csv_dir = Path("data/converted_csv")
    
    if not csv_dir.exists():
        print("❌ No converted CSV files found. Run main_dta_reader.py first!")
        return []
    
    csv_files = list(csv_dir.glob("*.csv"))
    
    print("📂 AVAILABLE DATASETS:")
    print("=" * 50)
    
    for i, file in enumerate(csv_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {file.stem:<20} ({size_mb:6.1f} MB)")
    
    return [f.stem for f in csv_files]


def load_dataset(dataset_name):
    """Load a specific dataset"""
    csv_path = Path(f"data/converted_csv/{dataset_name}.csv")
    
    if not csv_path.exists():
        print(f"❌ Dataset '{dataset_name}' not found!")
        return None
    
    print(f"🔄 Loading {dataset_name}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {dataset_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None


def explore_dataset(df, dataset_name):
    """Explore a dataset interactively"""
    print(f"\n🔍 EXPLORING DATASET: {dataset_name.upper()}")
    print("=" * 60)
    
    print(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show column info
    print(f"\n📋 COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{i+1:3d}. {col:<30} | {dtype:<10} | {null_count:>6} nulls ({null_pct:4.1f}%)")
    
    # Basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📊 NUMERIC COLUMNS SUMMARY ({len(numeric_cols)} columns):")
        print(df[numeric_cols].describe().round(4))
    
    # Show first and last few rows
    print(f"\n📋 FIRST 5 ROWS:")
    print(df.head())
    
    print(f"\n📋 LAST 5 ROWS:")
    print(df.tail())


def analyze_correlations(df, dataset_name, top_n=20):
    """Analyze correlations in the dataset"""
    print(f"\n🔗 CORRELATION ANALYSIS: {dataset_name.upper()}")
    print("=" * 60)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for correlation analysis")
        return
    
    # Remove volume/id columns
    exclude_patterns = ['volume', 'vol', 'id', 'ID', 'period']
    price_cols = [col for col in numeric_cols 
                  if not any(pattern.lower() in col.lower() for pattern in exclude_patterns)]
    
    if len(price_cols) < 2:
        print("❌ Not enough price columns for correlation analysis")
        return
    
    # Calculate correlations
    corr_data = df[price_cols].dropna()
    corr_matrix = corr_data.corr()
    
    # Find top correlations
    corr_pairs = []
    for i, col1 in enumerate(price_cols):
        for j, col2 in enumerate(price_cols[i+1:], i+1):
            corr_value = corr_matrix.loc[col1, col2]
            if not np.isnan(corr_value):
                corr_pairs.append({
                    'Asset_1': col1,
                    'Asset_2': col2,
                    'Correlation': corr_value,
                    'Abs_Correlation': abs(corr_value)
                })
    
    if not corr_pairs:
        print("❌ No valid correlations found")
        return
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    print(f"📊 Analyzed {len(price_cols)} price columns with {len(corr_data)} data points")
    print(f"\n🏆 TOP {min(top_n, len(corr_df))} CORRELATIONS:")
    display_df = corr_df.head(top_n)[['Asset_1', 'Asset_2', 'Correlation']].copy()
    display_df['Correlation'] = display_df['Correlation'].round(4)
    print(display_df.to_string(index=False))
    
    return corr_df


def plot_time_series(df, dataset_name, columns=None, max_cols=10):
    """Plot time series for the dataset"""
    print(f"\n📈 TIME SERIES VISUALIZATION: {dataset_name.upper()}")
    print("=" * 60)
    
    # Find date column
    date_col = None
    date_candidates = ['date', 'Date', 'DATE', 'period', 'time', 'timestamp']
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("❌ No date column found for time series plot")
        return
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter out volume and id columns
    exclude_patterns = ['volume', 'vol', 'id', 'ID']
    plot_cols = [col for col in numeric_cols 
                 if not any(pattern.lower() in col.lower() for pattern in exclude_patterns)]
    
    if columns:
        plot_cols = [col for col in columns if col in plot_cols]
    
    plot_cols = plot_cols[:max_cols]  # Limit number of columns
    
    if len(plot_cols) == 0:
        print("❌ No suitable columns for plotting")
        return
    
    # Convert date column
    try:
        df_plot = df[[date_col] + plot_cols].copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        df_plot = df_plot.dropna()
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        for col in plot_cols:
            plt.plot(df_plot[date_col], df_plot[col], label=col, alpha=0.8, linewidth=1)
        
        plt.title(f'Time Series: {dataset_name.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Plotted {len(plot_cols)} time series")
        
    except Exception as e:
        print(f"❌ Error creating time series plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Browse and analyze converted datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name to analyze")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--correlations", action="store_true", help="Show correlation analysis")
    parser.add_argument("--plot", action="store_true", help="Create time series plots")
    parser.add_argument("--columns", nargs="+", help="Specific columns to analyze/plot")
    
    args = parser.parse_args()
    
    if args.list or not args.dataset:
        datasets = list_available_datasets()
        if not args.dataset:
            return
    
    # Load and analyze specific dataset
    df = load_dataset(args.dataset)
    if df is None:
        return
    
    # Basic exploration
    explore_dataset(df, args.dataset)
    
    # Correlation analysis
    if args.correlations:
        analyze_correlations(df, args.dataset)
    
    # Time series plotting
    if args.plot:
        plot_time_series(df, args.dataset, args.columns)


if __name__ == "__main__":
    main() 