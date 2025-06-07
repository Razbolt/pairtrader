import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data.download_data import get_sp500_tickers, download_price_history
from agents.ppo_agent import train_ppo
from agents.ddqn_agent import train_dqn
from agents.sac_agent import train_sac
from pathlib import Path


def load_dta_dataset(dataset_name):
    """
    Load a specific .dta dataset from data/all-data/ or converted CSV from data/converted_csv/
    """
    # First try to load from converted CSV (faster)
    csv_path = Path(f"data/converted_csv/{dataset_name}.csv")
    if csv_path.exists():
        print(f"🔄 Loading converted CSV: {dataset_name}...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {dataset_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    # Fallback to reading .dta file directly
    dta_path = Path(f"data/all-data/{dataset_name}.dta")
    if dta_path.exists():
        print(f"🔄 Loading .dta file: {dataset_name}...")
        df = pd.read_stata(dta_path)
        print(f"✅ Loaded {dataset_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    print(f"❌ Dataset '{dataset_name}' not found in either CSV or .dta format!")
    return None


def prepare_dataset_for_trading(df, dataset_name):
    """
    Prepare a dataset for pair trading by extracting price columns
    """
    print(f"\n🔧 PREPARING {dataset_name.upper()} FOR PAIR TRADING")
    print("="*60)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove non-price columns
    exclude_patterns = ['volume', 'vol', 'id', 'ID', 'period', 'timestamp', 'r_', 'R_']
    price_cols = [col for col in numeric_cols 
                  if not any(pattern.lower() in col.lower() for pattern in exclude_patterns)]
    
    if len(price_cols) < 2:
        print(f"❌ Not enough price columns found in {dataset_name}")
        print(f"Available numeric columns: {list(numeric_cols)}")
        return None
    
    # Create price dataframe
    price_data = df[price_cols].dropna()
    
    if len(price_data) < 100:
        print(f"❌ Not enough data points ({len(price_data)}) for reliable analysis")
        return None
    
    print(f"✅ Extracted {len(price_cols)} price columns with {len(price_data)} data points")
    print(f"📊 Price columns: {', '.join(price_cols[:10])}{'...' if len(price_cols) > 10 else ''}")
    
    return price_data


def main(start: str = None, end: str = None, dataset: str = None):
    """
    Main function - can work with Yahoo Finance data OR .dta datasets
    """
    if dataset:
        # Use .dta dataset mode
        print(f"🎯 PAIR TRADING WITH DATASET: {dataset.upper()}")
        print("="*80)
        
        # Load dataset
        df = load_dta_dataset(dataset)
        if df is None:
            return
        
        # Prepare for pair trading
        prices = prepare_dataset_for_trading(df, dataset)
        if prices is None:
            return
            
    else:
        # Use Yahoo Finance mode (original functionality)
        if not start or not end:
            print("❌ Start and end dates required for Yahoo Finance mode!")
            return
            
        print(f"🎯 PAIR TRADING WITH YAHOO FINANCE DATA")
        print("="*80)
        
        tickers = get_sp500_tickers()
        prices = download_price_history(tickers, start, end)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        filename = data_dir / f"S&P-{start}--{end}.csv"
        prices.to_csv(filename)
        print(f"💾 Saved data to: {filename}")

    # Pair selection based on correlation
    print(f"\n📊 CORRELATION ANALYSIS")
    print("="*50)
    
    corr = prices.corr()
    print("Correlation matrix computed successfully!")
    
    # Convert correlation matrix to pairs format
    # Clear index names to avoid conflicts and then stack
    corr.index.name = None
    corr.columns.name = None
    corr_stacked = corr.stack()
    corr_pairs = corr_stacked.reset_index()
    corr_pairs.columns = ["stock1", "stock2", "corr"]
    
    # Remove self-correlations (diagonal elements)
    corr_pairs = corr_pairs[corr_pairs["stock1"] != corr_pairs["stock2"]]
    
    # Remove duplicate pairs (keep only upper triangle)
    corr_pairs = corr_pairs[corr_pairs["stock1"] < corr_pairs["stock2"]]
    
    # Sort by correlation (highest first)
    corr_pairs = corr_pairs.sort_values(by="corr", ascending=False)
    
    print("\n🏆 Top 10 correlated pairs:")
    print(corr_pairs.head(10))
    
    if len(corr_pairs) == 0:
        print("❌ No valid pairs found!")
        return
    
    pair = (corr_pairs.iloc[0]["stock1"], corr_pairs.iloc[0]["stock2"])
    correlation = corr_pairs.iloc[0]['corr']
    print(f"\n🎯 Selected pair: {pair} with correlation: {correlation:.4f}")

    # Scale spreads for environment
    print(f"\n🔧 PREPARING DATA FOR RL TRAINING")
    print("="*50)
    
    scaler = StandardScaler()
    pair_data = prices[list(pair)].dropna()
    
    if len(pair_data) < 100:
        print(f"❌ Not enough data points for training: {len(pair_data)}")
        return
    
    scaled_data = pd.DataFrame(
        scaler.fit_transform(pair_data),
        columns=pair_data.columns,
        index=pair_data.index
    )
    
    print(f"✅ Prepared {len(scaled_data)} data points for training")
    print(f"📈 Pair: {pair[0]} vs {pair[1]}")
    print(f"📊 Correlation: {correlation:.4f}")

    # Train RL agents
    print(f"\n🤖 TRAINING REINFORCEMENT LEARNING AGENTS")
    print("="*60)
    
    try:
        print("🔄 Training PPO...")
        train_ppo(scaled_data, pair, timesteps=5000)
        
        print("🔄 Training DQN...")
        train_dqn(scaled_data, pair, timesteps=5000)
        
        print("🔄 Training SAC...")
        train_sac(scaled_data, pair, timesteps=5000)
        
        print("✅ All training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")


def list_available_datasets():
    """List available .dta datasets"""
    dta_dir = Path("data/all-data")
    csv_dir = Path("data/converted_csv")
    
    print("📂 AVAILABLE DATASETS:")
    print("="*50)
    
    datasets = set()
    
    # Check .dta files
    if dta_dir.exists():
        for dta_file in dta_dir.glob("*.dta"):
            datasets.add(dta_file.stem)
    
    # Check converted CSV files
    if csv_dir.exists():
        for csv_file in csv_dir.glob("*.csv"):
            datasets.add(csv_file.stem)
    
    if not datasets:
        print("❌ No datasets found!")
        print("   • Place .dta files in data/all-data/")
        print("   • Or run main_dta_reader.py to convert existing .dta files")
        return
    
    for i, dataset in enumerate(sorted(datasets), 1):
        # Check if both formats exist
        has_dta = (dta_dir / f"{dataset}.dta").exists()
        has_csv = (csv_dir / f"{dataset}.csv").exists()
        
        format_info = []
        if has_csv:
            size = (csv_dir / f"{dataset}.csv").stat().st_size / (1024*1024)
            format_info.append(f"CSV({size:.1f}MB)")
        if has_dta:
            size = (dta_dir / f"{dataset}.dta").stat().st_size / (1024*1024)
            format_info.append(f"DTA({size:.1f}MB)")
        
        print(f"{i:2d}. {dataset:<20} [{', '.join(format_info)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pair Trading with RL - Yahoo Finance or .dta datasets")
    parser.add_argument("--start", type=str, help="Start date for Yahoo Finance data (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for Yahoo Finance data (YYYY-MM-DD)")
    parser.add_argument("--dataset", type=str, help="Name of .dta dataset to use (alternative to Yahoo Finance)")
    parser.add_argument("--list-datasets", action="store_true", help="List available .dta datasets")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_available_datasets()
    elif args.dataset:
        main(dataset=args.dataset)
    elif args.start and args.end:
        main(args.start, args.end)
    else:
        print("🔍 Usage Examples:")
        print("="*50)
        print("📊 Yahoo Finance mode:")
        print("   python main.py --start 2023-01-01 --end 2023-12-31")
        print()
        print("📁 Dataset mode:")
        print("   python main.py --dataset commodities")
        print("   python main.py --dataset sp500")
        print("   python main.py --dataset crypto_usd")
        print()
        print("📂 List available datasets:")
        print("   python main.py --list-datasets")
        print()
        print("💡 Available datasets:")
        list_available_datasets()

