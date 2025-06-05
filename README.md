# Pair Trading with Reinforcement Learning

This project demonstrates a simple research framework for creating a pair trading strategy using reinforcement learning. The repository contains utilities to download data, define a trading environment and train several RL agents.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The main entry point is `main.py`. It downloads historical data from Yahoo Finance for the S&P 500, prepares a pair trading environment and trains RL agents (PPO, DQN and SAC).

```bash
python main.py --start 2015-01-01 --end 2020-01-01
```

The code is for educational use. It is not production ready and should be extended for a full dissertation.

