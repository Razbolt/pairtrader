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
    corr_pairs = (
        corr.stack(future_stack=True)
        .reset_index(name="corr")
        .rename(columns={"level_0": "stock1", "level_1": "stock2"})
    )
    corr_pairs = corr_pairs[corr_pairs["stock1"] != corr_pairs["stock2"]]
    corr_pairs = corr_pairs.sort_values(by="corr", ascending=False)
    pair = (corr_pairs.iloc[0]["stock1"], corr_pairs.iloc[0]["stock2"])
    print(f"Selected pair: {pair}")

    # Scale spreads for environment
    scaler = StandardScaler()
    prices[pair] = scaler.fit_transform(prices[pair])

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

