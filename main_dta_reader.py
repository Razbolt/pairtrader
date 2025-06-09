import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from agents.ppo_agent import train_ppo
from agents.ddqn_agent import train_dqn
from agents.sac_agent import train_sac


def read_dta_files(data_folder="data/all-data"):
    """
    Read all .dta files from the specified folder and convert them to CSV
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"❌ Error: Folder '{data_folder}' does not exist!")
        return {}
    
    # Find all .dta files
    dta_files = list(data_path.glob("*.dta"))
    
    if not dta_files:
        print(f"❌ No .dta files found in '{data_folder}'")
        return {}
    
    print(f"🔍 Found {len(dta_files)} .dta files:")
    for file in dta_files:
        print(f"   📁 {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    dataframes = {}
    csv_output_dir = Path("data/converted_csv")
    csv_output_dir.mkdir(exist_ok=True)
    
    for dta_file in dta_files:
        try:
            print(f"\n🔄 Processing {dta_file.name}...")
            
            # Read .dta file
            df = pd.read_stata(dta_file)
            
            # Store in dictionary
            dataset_name = dta_file.stem  # filename without extension
            dataframes[dataset_name] = df
            
            # Convert to CSV
            csv_filename = csv_output_dir / f"{dataset_name}.csv"
            df.to_csv(csv_filename, index=False)
            
            print(f"✅ Successfully loaded {dta_file.name}")
            print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"   💾 Saved as: {csv_filename}")
            
            # Display first few rows and basic info
            print(f"   📈 Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
        except Exception as e:
            print(f"❌ Error reading {dta_file.name}: {e}")
            continue
    
    return dataframes


def display_dataframe_summaries(dataframes):
    """
    Display summary information for all dataframes
    """
    print("\n" + "="*80)
    print("📊 DATAFRAME SUMMARIES")
    print("="*80)
    
    for name, df in dataframes.items():
        print(f"\n🗂️  Dataset: {name.upper()}")
        print("-" * 50)
        print(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"📅 Date range: {get_date_range_info(df)}")
        print(f"🏷️  Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")
        
        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 Basic Statistics (first 5 numeric columns):")
            print(df[numeric_cols[:5]].describe().round(4))
        
        print("\n" + "-"*50)


def get_date_range_info(df):
    """
    Try to extract date range information from dataframe
    """
    # Common date column names
    date_columns = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp']
    
    for col in date_columns:
        if col in df.columns:
            try:
                date_series = pd.to_datetime(df[col])
                return f"{date_series.min().strftime('%Y-%m-%d')} to {date_series.max().strftime('%Y-%m-%d')}"
            except:
                continue
    
    # Check if index is datetime
    if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
        return f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    
    return "Date range not detected"


def analyze_dataset_for_pairs(df, dataset_name):
    """
    Analyze a dataset for potential pair trading opportunities
    """
    print(f"\n🔍 PAIR TRADING ANALYSIS: {dataset_name.upper()}")
    print("="*60)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for pair analysis")
        return None
    
    # Remove non-price columns (common names)
    exclude_cols = ['volume', 'Volume', 'VOLUME', 'vol', 'id', 'ID']
    price_cols = [col for col in numeric_cols if not any(excl.lower() in col.lower() for excl in exclude_cols)]
    
    if len(price_cols) < 2:
        print("❌ Not enough price columns for pair analysis")
        return None
    
    # Create correlation matrix
    price_data = df[price_cols].dropna()
    
    if len(price_data) < 50:  # Need minimum data points
        print("❌ Not enough data points for reliable analysis")
        return None
    
    corr_matrix = price_data.corr()
    
    print(f"📊 Correlation analysis on {len(price_cols)} price columns with {len(price_data)} data points")
    print(f"🏷️  Price columns: {', '.join(price_cols[:10])}{'...' if len(price_cols) > 10 else ''}")
    
    # Find top correlated pairs
    corr_pairs = []
    for i, col1 in enumerate(price_cols):
        for j, col2 in enumerate(price_cols[i+1:], i+1):
            corr_value = corr_matrix.loc[col1, col2]
            if not np.isnan(corr_value):
                corr_pairs.append({
                    'stock1': col1,
                    'stock2': col2,
                    'correlation': corr_value
                })
    
    if not corr_pairs:
        print("❌ No valid correlations found")
        return None
    
    # Sort by absolute correlation
    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['abs_corr'] = abs(corr_pairs_df['correlation'])
    corr_pairs_df = corr_pairs_df.sort_values('abs_corr', ascending=False)
    
    print(f"\n🏆 Top 10 Correlated Pairs:")
    print(corr_pairs_df.head(10)[['stock1', 'stock2', 'correlation']].round(4))
    
    return price_data, corr_pairs_df


def main_dta_analysis():
    """
    Main function to analyze .dta files
    """
    print("🚀 STARTING .DTA FILES ANALYSIS")
    print("="*80)
    
    # Read all .dta files
    dataframes = read_dta_files()
    
    if not dataframes:
        print("❌ No data loaded. Exiting...")
        return
    
    # Display summaries
    display_dataframe_summaries(dataframes)
    
    # Analyze each dataset for pair trading opportunities
    print("\n" + "="*80)
    print("🎯 PAIR TRADING ANALYSIS")
    print("="*80)
    
    trading_datasets = {}
    
    for name, df in dataframes.items():
        result = analyze_dataset_for_pairs(df, name)
        if result is not None:
            price_data, corr_pairs = result
            trading_datasets[name] = {
                'price_data': price_data,
                'correlations': corr_pairs,
                'original_df': df
            }
    
    # Interactive dataset selection
    if trading_datasets:
        print(f"\n🎯 Found {len(trading_datasets)} datasets suitable for pair trading:")
        for i, name in enumerate(trading_datasets.keys(), 1):
            print(f"   {i}. {name}")
        
        print("\n💡 You can now use these datasets for further analysis!")
        print("📁 All datasets have been converted to CSV files in 'data/converted_csv/'")
        
        return trading_datasets
    else:
        print("❌ No datasets suitable for pair trading analysis found")
        return None


def run_pair_trading_on_dataset(dataset_name, trading_datasets):
    """
    Run pair trading analysis on a specific dataset
    """
    if dataset_name not in trading_datasets:
        print(f"❌ Dataset '{dataset_name}' not found in trading datasets")
        return
    
    data = trading_datasets[dataset_name]
    price_data = data['price_data']
    corr_pairs = data['correlations']
    
    print(f"\n🚀 RUNNING PAIR TRADING ON: {dataset_name.upper()}")
    print("="*60)
    
    # Get best pair
    best_pair = corr_pairs.iloc[0]
    stock1, stock2 = best_pair['stock1'], best_pair['stock2']
    correlation = best_pair['correlation']
    
    print(f"🎯 Selected pair: {stock1} & {stock2}")
    print(f"📊 Correlation: {correlation:.4f}")
    
    # Prepare data for RL training
    pair_data = price_data[[stock1, stock2]].dropna()
    
    if len(pair_data) < 100:
        print("❌ Not enough data for RL training")
        return
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(pair_data),
        columns=pair_data.columns,
        index=pair_data.index
    )
    
    print(f"✅ Prepared {len(scaled_data)} data points for training")
    
    # Train RL agents
    pair_tuple = (stock1, stock2)
    
    try:
        print("\n🤖 Training PPO...")
        train_ppo(scaled_data, pair_tuple, timesteps=5000)
        
        print("🤖 Training DQN...")
        train_dqn(scaled_data, pair_tuple, timesteps=5000)
        
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")


if __name__ == "__main__":
    # Run the main analysis
    trading_datasets = main_dta_analysis()
    
    # Example: Run pair trading on the first suitable dataset
    if trading_datasets:
        first_dataset = list(trading_datasets.keys())[0]
        print(f"\n🎯 Running example pair trading on '{first_dataset}'...")
        run_pair_trading_on_dataset(first_dataset, trading_datasets) 