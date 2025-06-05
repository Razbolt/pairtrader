import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.download_data import get_sp500_tickers, download_price_history
from agents.ppo_agent import train_ppo
from agents.ddqn_agent import train_dqn
from agents.sac_agent import train_sac
from pathlib import Path


def main(start: str, end: str):
    tickers = get_sp500_tickers()
    prices = download_price_history(tickers, start, end)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filename = data_dir / f"S&P-{start}--{end}.csv"
    prices.to_csv(filename)

    # Pair selection based on correlation
    corr = prices.corr()
    print("Correlation matrix:")
    print(corr)
    
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
    
    print("Top correlated pairs:")
    print(corr_pairs.head(10))
    
    pair = (corr_pairs.iloc[0]["stock1"], corr_pairs.iloc[0]["stock2"])
    print(f"Selected pair: {pair} with correlation: {corr_pairs.iloc[0]['corr']:.4f}")

    # Scale spreads for environment
    scaler = StandardScaler()
    prices[list(pair)] = scaler.fit_transform(prices[list(pair)])

    print("Training PPO...")
    train_ppo(prices, pair, timesteps=5000)
    print("Training DQN...")
    train_dqn(prices, pair, timesteps=5000)
    print("Training SAC...")
    train_sac(prices, pair, timesteps=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()
    main(args.start, args.end)

