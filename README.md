# Pair Trading with Reinforcement Learning

This repository contains a toy framework for experimenting with pair trading using reinforcement learning.  It provides helpers for fetching price data, a simple trading environment and training scripts for three different RL algorithms.  The goal is to showcase the basic workflow rather than provide a production-ready solution.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The main entry point is `main.py`. It downloads daily close prices (with dividends and splits auto-adjusted) for 20 S&P 500 tickers. The prices are saved under `data/` using a file name such as `S&P-2015-01-01--2020-01-01.csv`. The script prints the correlation matrix of the prices and then trains RL agents (PPO, DQN and SAC).

```bash
python main.py --start 2015-01-01 --end 2020-01-01
```

The code is for educational use. It is not production ready and should be extended for a full dissertation.

Data is downloaded using `yfinance`. If internet access is unavailable the script
falls back to a small set of hard coded tickers.

Only the adjusted close prices are used when computing the stock correlation matrix.

