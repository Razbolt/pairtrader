# 🚀 Pair Trading Data Cleaner

Clean financial data for pair trading strategies with perfect data quality.

## 🔧 Quick Start

```bash
# Basic usage
python clean_data.py sp500 2023-01-01 2025-04-30

# Custom formation/trading periods
python clean_data.py sp500 2023-01-01 2025-04-30 --formation 12 --trading 6
python clean_data.py Chinese_stocks 2024-01-01 2024-12-31 --formation 6 --trading 3
python clean_data.py FTSE100 2023-06-01 2024-12-31

# Show help
python clean_data.py
```

## 📊 Available Datasets

- `sp500` - S&P 500 stocks
- `FTSE100` - FTSE 100 stocks  
- `Chinese_stocks` - Chinese market stocks
- `commodities` - Commodity prices
- `crypto_eur` - Crypto prices in EUR
- `crypto_usd` - Crypto prices in USD
- `stock_market_indices` - Global market indices

## 📈 Output Structure

```
data/cleaned/sp500_20230101_20250430/
├── sp500_20230101_20250430_full.csv      # Complete dataset
├── sp500_20230101_20250430_formation.csv # Formation period only
├── sp500_20230101_20250430_trading.csv   # Trading period only  
└── sp500_20230101_20250430_info.txt      # Metadata & stats
```

## 🎯 Features

- **Automatic data type selection**: Prioritizes log returns > simple returns > prices
- **Smart missing value handling**: Different strategies for returns vs prices
- **Flexible periods**: Custom formation and trading period lengths
- **Perfect data quality**: Zero missing values for 2023-2025 data
- **Complete metadata**: Detailed info files with all parameters

## 🏆 Recommended Usage

**For current pair trading (2023-2025):**
```bash
python clean_data.py sp500 2023-01-01 2025-04-30
```

**For backtesting with shorter periods:**
```bash
python clean_data.py sp500 2023-01-01 2025-04-30 --formation 6 --trading 3
```

## 📋 Core Files

- `clean_data.py` - Main command-line cleaner
- `main.py` - Original correlation analysis  
- `main_dta_reader.py` - Data converter (.dta to .csv)
- `data_browser.py` - Interactive data exploration

