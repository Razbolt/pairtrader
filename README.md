# 🚀 Pair Trading Strategy Framework

A comprehensive statistical arbitrage system implementing cointegration-based pair trading strategies with clean data pipelines and flexible configuration options.

## 📊 Project Overview

This framework provides a complete implementation of **cointegration-based pair trading**, a market-neutral statistical arbitrage strategy that exploits temporary price divergences between statistically related assets.

### 🔗 **What is Pair Trading?**
Pair trading identifies two assets with a stable long-term statistical relationship. When their price relationship deviates from the historical norm, the strategy:
- **Goes long** the undervalued asset
- **Goes short** the overvalued asset  
- **Exits** when the relationship returns to normal

This creates a **market-neutral position** that profits from mean reversion rather than market direction.

## 🛠️ Key Components

### **1. Data Pipeline**
- **Raw Data Processing**: Convert financial data to clean CSV format
- **Quality Validation**: Ensure no missing values or data alignment issues
- **Period Separation**: Automatic formation/trading period splitting

### **2. Cointegration Analysis**
- **Statistical Testing**: Engle-Granger cointegration tests
- **Pair Selection**: Identify statistically significant relationships
- **Hedge Ratio Calculation**: Determine optimal position sizing

### **3. Trading Strategy**
- **Z-Score Signals**: Mean reversion trading based on spread normalization
- **Risk Management**: Transaction costs, position limits, exit rules
- **Performance Tracking**: Comprehensive trade and performance analysis

## 🔧 Quick Start

### Generate Clean Data
```bash
# Create clean price data for pair trading analysis
python clean_data_enhanced.py sp500 2023-01-01 --in-sample 12 --out-sample 6 --data-type prices
```

### Run Cointegration Strategy
```bash
# Basic cointegration pair trading
python cointegration_raw_strategy.py data/pair_trading/sp500_20230101_20240705_prices_12m6m

# Custom configuration
python cointegration_raw_strategy.py data/pair_trading/sp500_20230101_20240705_prices_12m6m \
  --entry-threshold 1.5 --exit-threshold 0.5 --max-pairs 30
```

## ⚙️ Configuration Options

### **Strategy Parameters**
- **Entry Threshold**: Z-score level to enter positions (default: 1.0)
- **Exit Threshold**: Z-score level to exit positions (default: 0.0)
- **Max Pairs**: Maximum number of pairs to trade (default: 20)
- **Min Stocks**: Number of stocks to test for pairs (default: 100)
- **Significance Level**: Statistical significance for cointegration (default: 0.05)
- **Transaction Cost**: Trading cost rate (default: 0.001)

### **Risk Profiles**
```bash
# Conservative: Fewer, higher-quality trades
--entry-threshold 2.0 --exit-threshold 0.5 --max-pairs 10

# Moderate: Balanced approach
--entry-threshold 1.0 --exit-threshold 0.5 --max-pairs 20

# Aggressive: More frequent trading
--entry-threshold 0.8 --exit-threshold 0.3 --max-pairs 50
```

## 📁 Project Structure

```
📦 Pair Trading Framework
├── 🎯 Core Strategy
│   ├── cointegration_raw_strategy.py    # Main trading strategy
│   └── clean_data_enhanced.py           # Data preprocessing
├── 📊 Data Management
│   ├── data/pair_trading/               # Processed datasets
│   ├── data/converted_csv/              # Raw market data
│   └── main_dta_reader.py              # Data format conversion
├── 📚 Documentation
│   ├── PRD.md                          # Project requirements
│   ├── README.md                       # This guide
│   └── global_rules.md                 # Development standards
└── 🔧 Legacy Tools
    └── main.py                         # Original analysis scripts
```

## 🎯 Features

### **Statistical Rigor**
- **Engle-Granger Testing**: Nobel Prize-winning cointegration methodology
- **Multiple Testing Correction**: Proper statistical significance handling
- **Robustness Checks**: Validation across different time periods

### **Data Quality**
- **Clean Pipeline**: Automated data preprocessing and validation
- **Multiple Formats**: Support for various financial data sources
- **Missing Value Handling**: Smart interpolation and forward-filling

### **Flexible Framework**
- **Configurable Parameters**: Adapt to different markets and risk profiles
- **Multiple Assets**: Support for stocks, commodities, crypto, forex
- **Scalable Architecture**: Handle large universes of potential pairs

### **Risk Management**
- **Transaction Costs**: Realistic cost modeling
- **Position Sizing**: Proper hedge ratio calculation
- **Exposure Limits**: Maximum position and concentration controls

## 📊 Supported Datasets

### **Equity Markets**
- **S&P 500** (`sp500`) - US large-cap stocks
- **FTSE 100** (`FTSE100`) - UK stocks
- **Chinese Stocks** (`Chinese_stocks`) - Chinese market

### **Alternative Assets**
- **Commodities** (`commodities`) - Commodity futures
- **Crypto EUR/USD** (`crypto_eur`, `crypto_usd`) - Cryptocurrency pairs
- **Market Indices** (`stock_market_indices`) - Global indices

## 🧠 Strategy Philosophy

### **Academic Foundation**
Based on established financial literature:
- **Engle & Granger (1987)**: Cointegration theory
- **Gatev, Goetzmann & Rouwenhorst (2006)**: Pairs trading methodology  
- **Avellaneda & Lee (2010)**: Statistical arbitrage framework

### **Practical Implementation**
- **Real-world Constraints**: Transaction costs, market impact, liquidity
- **Risk Management**: Position limits, stop-losses, exposure controls
- **Operational Efficiency**: Automated execution and monitoring

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
```

### **Basic Workflow**
1. **Data Preparation**: Clean and format financial data
2. **Pair Discovery**: Test assets for cointegration relationships
3. **Strategy Configuration**: Set risk parameters and thresholds
4. **Backtesting**: Validate strategy on historical data
5. **Analysis**: Review performance and risk metrics

### **Example Workflow**
```bash
# 1. Prepare data
python clean_data_enhanced.py sp500 2023-01-01 --data-type prices

# 2. Run strategy
python cointegration_raw_strategy.py data/pair_trading/sp500_20230101_20240705_prices_12m6m

# 3. Analyze results (automatically displayed)
```

## 📈 Use Cases

### **Quantitative Hedge Funds**
- Market-neutral strategies
- Statistical arbitrage portfolios
- Risk-controlled alpha generation

### **Institutional Investors**
- Portfolio diversification
- Alternative risk premia
- Systematic trading strategies

### **Academic Research**
- Cointegration analysis
- Market efficiency studies
- Strategy development and testing

### **Individual Traders**
- Systematic trading approaches
- Market-neutral positioning
- Statistical edge exploitation

## 📝 Development Notes

- **Time Periods**: Configurable formation and trading periods
- **Market Conditions**: Tested across different market regimes
- **Data Quality**: Comprehensive validation and cleaning
- **Best Practices**: Following academic and industry standards

## 🤝 Contributing

This project follows structured development principles:
1. **PRD-Driven**: All changes documented in PRD.md
2. **Modular Design**: Maximum 500 lines per file
3. **Clean Code**: Comprehensive error handling and logging
4. **Academic Rigor**: Proper statistical methodology

---

## 🎯 **Framework Benefits**

- ✅ **Academically Sound**: Proper statistical methodology
- ✅ **Production Ready**: Robust error handling and validation
- ✅ **Highly Configurable**: Adaptable to different markets and styles
- ✅ **Well Documented**: Comprehensive guides and examples
- ✅ **Proven Approach**: Based on established literature

---

*For detailed technical specifications and implementation notes, see `PRD.md`*

