# Pair Trading Research Framework

A comprehensive research framework for pair trading strategies across multiple asset classes (stocks, commodities, crypto) with advanced signal optimization using machine learning, statistical methods, and unsupervised approaches.

## 🚀 Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create a new dataset
python create_pair_trading_dataset.py sp500 2024-01-01 --in-sample 12 --out-sample 6

# Run log prices cointegration strategy (enhanced)
python cointegration_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --max-pairs 20

# Run ML signal optimization
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --optimize-signals --model xgboost

# Run unsupervised pair trading
python unsupervised_pair_trading.py data/pair_trading/commodities_20210101_20231124_prices_24m12m --clustering-method kmeans
```

## 🔬 Research Features

### 🎯 Signal Optimization Methods
- **Machine Learning:** XGBoost, Random Forest, Logistic Regression, SVM
- **Statistical:** Threshold optimization, rolling statistics, z-score analysis
- **Hybrid:** Ensemble methods, feature engineering, market regime detection
- **Unsupervised:** Clustering-based pair selection (K-means, DBSCAN)

### 📊 Multi-Asset Support
- **Stocks:** S&P 500 with various timeframes (12m/6m, 24m/12m, 48m/12m)
- **Commodities:** Oil, gold, copper, wheat, cocoa, sugar, nickel, platinum
- **Crypto:** Bitcoin, Ethereum, and major altcoins
- **Flexible Parameters:** Asset-specific optimization and backtesting

### 📈 Advanced Analytics
- **Log Prices Cointegration:** More robust than raw prices for statistical testing
- **Performance Metrics:** Sharpe ratio, Sortino ratio, win rate, profit factor, max drawdown
- **Visualization:** Trade signals, P&L charts, pair relationships, performance comparisons
- **Backtesting:** Comprehensive in-sample/out-sample analysis

## 📁 Main Components

### 🎯 Strategy Files
- **`cointegration_strategy.py`** - Enhanced log prices cointegration strategy
- **`ml_pair_trading_strategy.py`** - Machine learning and signal optimization
- **`unsupervised_pair_trading.py`** - Clustering-based pair selection and trading

### 📊 Visualization Tools
- **`cointegration_trades_visualization.py`** - Cointegration strategy visualization
- **`ml_trades_visualization.py`** - ML strategy visualization and analysis
- **`multi_pair_trades_visualization.py`** - Multi-pair trading performance charts
- **`simple_multi_pair_visualization.py`** - Simplified multi-pair visualization

### 🛠️ Data Processing
- **`create_pair_trading_dataset.py`** - Multi-asset data preprocessing and split
- **`create_commodities_dataset.py`** - Commodities-specific data processing
- **`explore_dataset.py`** - Dataset exploration and analysis
- **`stata_data_reader.py`** - Stata (.dta) file reader

## 🧪 Usage Examples

### Basic Cointegration Strategy (Log Prices)
```bash
# Enhanced cointegration with log prices
python cointegration_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --max-pairs 20 --entry-threshold 1.5 --exit-threshold 0.5

# Commodities cointegration
python cointegration_strategy.py data/pair_trading/commodities_20210101_20231124_prices_24m12m --max-pairs 10 --significance 0.10
```

### Machine Learning Signal Optimization
```bash
# XGBoost regression for signal prediction
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --optimize-signals --model xgboost --regression

# Random Forest classification
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --optimize-signals --model random_forest --classification

# Compare with cointegration baseline
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --optimize-signals --compare-cointegration --model xgboost
```

### Unsupervised Pair Trading
```bash
# K-means clustering for pair selection
python unsupervised_pair_trading.py data/pair_trading/commodities_20210101_20231124_prices_24m12m --clustering-method kmeans --n-clusters 5

# DBSCAN clustering
python unsupervised_pair_trading.py data/pair_trading/crypto_usd_20240614_20250424_prices_10m5m --clustering-method dbscan --eps 0.3
```

### Visualization and Analysis
```bash
# Visualize ML strategy results
python ml_trades_visualization.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --strategy ml --model xgboost

# Compare multiple strategies
python multi_pair_trades_visualization.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --strategies cointegration,ml,unsupervised
```

## 🤖 XGBoost Regression for Pair Trading

### How It Works
XGBoost (eXtreme Gradient Boosting) is an ensemble learning method that combines multiple decision trees to predict trading signals:

1. **Feature Engineering:**
   - Price spreads between pairs
   - Rolling statistics (mean, std, z-scores)
   - Technical indicators (RSI, MACD, moving averages)
   - Market regime indicators

2. **Training Process:**
   - Uses gradient boosting to minimize prediction errors
   - Handles overfitting through regularization
   - Provides feature importance rankings

3. **Signal Generation:**
   - Model predicts expected returns for each time period
   - Thresholds determine entry/exit signals
   - Ensemble predictions improve robustness

### Key Advantages
- **Non-linear Relationships:** Captures complex market dynamics
- **Feature Importance:** Identifies most predictive factors
- **Regularization:** Prevents overfitting to historical data
- **Handles Missing Data:** Robust to data quality issues

### Example Usage
```bash
# Train XGBoost regression model
python ml_pair_trading_strategy.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --model xgboost --regression --features spread,rolling_mean,rolling_std,rsi

# Analyze feature importance
python ml_trades_visualization.py data/pair_trading/sp500_20240101_20250430_prices_12m6m --strategy ml --model xgboost --show-features
```

## 📊 Available Datasets

### Pre-built Datasets
- `sp500_20240101_20250430_prices_12m6m/` - S&P 500 stocks (12m formation, 6m trading)
- `sp500_20190101_20240103_prices_48m12m/` - S&P 500 stocks (48m formation, 12m trading)
- `commodities_20210101_20231124_prices_24m12m/` - Commodities (24m formation, 12m trading)
- `crypto_usd_20240614_20250424_prices_10m5m/` - Crypto pairs (10m formation, 5m trading)

### Create Custom Dataset
```bash
# Create S&P 500 dataset
python create_pair_trading_dataset.py sp500 2024-01-01 --in-sample 12 --out-sample 6

# Create commodities dataset
python create_commodities_dataset.py 2021-01-01 2023-11-24 --in-sample 24 --out-sample 12
```

## 🎓 Research Features

- **Signal Optimization** - Multiple methods for entry/exit signal generation
- **In-Sample/Out-Sample Analysis** - Proper train/test splits with statistical validation
- **Multi-Asset Support** - Stocks, commodities, crypto with asset-specific parameters
- **Statistical Testing** - Engle-Granger cointegration tests, ADF tests
- **Performance Metrics** - Sharpe ratio, Sortino ratio, win rate, profit factor, max drawdown
- **Advanced Visualization** - Trade signals, P&L charts, pair relationships, performance comparisons
- **Backtesting Results** - Comprehensive analysis stored in `backtest_results/`

## ⚠️ Disclaimer

This code is for **educational and research purposes only**. It is not production-ready and should be extended and validated before any real trading applications. The strategies implemented are for academic research and should not be used for actual trading without proper risk management and regulatory compliance.

## 📚 Documentation

- **`PRD.md`** - Product Requirements Document with complete project overview
- **`global_rules.md`** - Project guidelines and rules
- **`backtest_results/`** - Comprehensive backtesting results and analysis
- **`analysis/`** - Additional analysis and research notebooks

## 🔧 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- matplotlib, seaborn for visualization
- statsmodels for statistical testing
- See `requirements.txt` for complete list

## 📈 Performance Highlights

- **Log Prices Cointegration:** More robust statistical testing than raw prices
- **ML Signal Optimization:** XGBoost shows superior performance vs baseline
- **Multi-Asset Applicability:** Strategies work across different asset classes
- **Comprehensive Backtesting:** Rigorous in-sample/out-sample validation
- **Advanced Visualization:** Detailed analysis and performance tracking

