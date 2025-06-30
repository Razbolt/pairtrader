# Pair Trading Research Framework

This project provides a comprehensive research framework for pair trading strategies using cointegration analysis, correlation analysis, and machine learning approaches.

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
```

## 📁 Main Components

### 🎯 Strategy Files
- **`cointegration_strategy.py`** - Cointegration-based pair trading using raw prices
- **`correlation_returns_strategy.py`** - Correlation-based strategy using log returns  
- **`ml_pair_trading_strategy.py`** - Machine learning approach with XGBoost/Random Forest

### 📊 Visualization Tools
- **`cointegration_trades_visualization.py`** - Visualize cointegration trades and signals
- **`multi_pair_trades_visualization.py`** - Multi-pair trading performance charts
- **`simple_multi_pair_visualization.py`** - Simplified multi-pair visualization

### 🔧 Data Processing
- **`create_pair_trading_dataset.py`** - Create in-sample/out-sample datasets
- **`explore_dataset.py`** - Explore and analyze dataset contents
- **`stata_data_reader.py`** - Read Stata (.dta) files

## 📈 Usage Examples

```bash
# Cointegration strategy with custom parameters
python cointegration_strategy.py data/pair_trading/dataset_name \
    --max-pairs 20 \
    --min-stocks 50 \
    --entry-threshold 1.5 \
    --exit-threshold 0.5

# Visualize specific pair trades
python cointegration_trades_visualization.py data/pair_trading/dataset_name \
    --stock1 AAPL --stock2 MSFT \
    --entry-threshold 1.0 --exit-threshold 0.5

# Machine learning strategy
python ml_pair_trading_strategy.py data/pair_trading/dataset_name \
    --model xgboost --max-pairs 10
```

## 🎓 Research Features

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

