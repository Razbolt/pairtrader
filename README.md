# Pair Trading Research Framework

This project provides a comprehensive research framework for pair trading strategies, with a new primary focus on **signal optimization using different methods** and direct comparison with the classic cointegration approach.

## 🚀 Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create a new dataset
python create_pair_trading_dataset.py sp500 2024-01-01 --in-sample 12 --out-sample 6

# Run cointegration strategy
python cointegration_strategy.py data/pair_trading/sp500_20240101_20250430_log_returns_12m6m --max-pairs 10

# Run signal optimization experiments (see below)
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_log_returns_12m6m --optimize-signals --compare-cointegration
```

## 🔬 Primary Research Focus: Signal Optimization

- **Goal:** Optimize entry/exit signals for pair trading using various methods (ML, statistical, hybrid, etc.)
- **Comparison:** All optimized signal methods are benchmarked against the classic cointegration strategy
- **Metrics:** Sharpe ratio, win rate, P&L, trade frequency, and robustness
- **Approaches:**
  - Machine learning classifiers/regressors (XGBoost, Random Forest, etc.)
  - Statistical threshold optimization
  - Hybrid and ensemble methods
  - Feature engineering for signal quality

## 📁 Main Components

### 🎯 Strategy Files
- **`cointegration_strategy.py`** - Cointegration-based pair trading using raw prices
- **`ml_pair_trading_strategy.py`** - Machine learning and signal optimization approaches
- **`correlation_returns_strategy.py`** - Correlation-based strategy using log returns

### 📊 Visualization Tools
- **`cointegration_trades_visualization.py`** - Visualize cointegration trades and signals
- **`multi_pair_trades_visualization.py`** - Multi-pair trading performance charts
- **`simple_multi_pair_visualization.py`** - Simplified multi-pair visualization

### 🛠️ Data Processing
- **`create_pair_trading_dataset.py`** - Create in-sample/out-sample datasets
- **`explore_dataset.py`** - Explore and analyze dataset contents
- **`stata_data_reader.py`** - Read Stata (.dta) files

## 🧪 Usage Examples

```bash
# Cointegration strategy
python cointegration_strategy.py data/pair_trading/dataset_name --max-pairs 20

# Signal optimization (ML, hybrid, etc.)
python ml_pair_trading_strategy.py data/pair_trading/dataset_name --optimize-signals --compare-cointegration --model xgboost

# Visualize results
python cointegration_trades_visualization.py data/pair_trading/dataset_name --stock1 AAPL --stock2 MSFT
```

## 🎓 Research Features

- **Signal Optimization** - Multiple methods for entry/exit signal generation
- **In-Sample/Out-Sample Analysis** - Proper train/test splits
- **Multiple Data Types** - Log returns, raw prices, simple returns
- **Statistical Testing** - Engle-Granger cointegration tests
- **Performance Metrics** - Sharpe ratio, win rate, profit factor
- **Visualization** - Trade signals, P&L charts, pair relationships

## ⚠️ Disclaimer

This code is for **educational and research purposes only**. It is not production-ready and should be extended and validated before any real trading applications.

## 📚 Documentation

- **`PRD.md`** - Product Requirements Document
- **`global_rules.md`** - Project guidelines and rules

