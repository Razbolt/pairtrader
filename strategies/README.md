# 📊 Pair Trading Strategies

This folder contains organized implementations of different pair trading approaches with clean structure and fixed visualizations.

## 📁 Folder Structure

```
strategies/
├── correlation_based/
│   ├── correlation_strategy.py          # Clean correlation-based strategy
│   ├── pair_trading_strategy.py         # Original correlation strategy
│   └── enhanced_pair_trading_strategy.py # Enhanced correlation strategy
├── cointegration_based/
│   ├── cointegration_strategy.py        # Clean cointegration-based strategy
│   ├── cointegration_analysis.py        # Full cointegration analysis
│   └── focused_cointegration.py         # Focused cointegration with examples
├── analysis_tools/
│   ├── clean_data_enhanced.py           # Data cleaning and preparation
│   └── data_browser.py                  # Interactive data exploration
├── strategy_comparison.py               # Side-by-side strategy comparison
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. **Correlation Strategy**
```bash
cd strategies/correlation_based
python correlation_strategy.py ../../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m --plot
```

### 2. **Cointegration Strategy**
```bash
cd strategies/cointegration_based
python cointegration_strategy.py ../../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m --plot
```

### 3. **Strategy Comparison** ⭐
```bash
cd strategies
python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m --plot
```

## 📊 Strategy Descriptions

### 🔗 **Correlation-Based Strategy**
- **Method**: High correlation pairs (>0.80)
- **Assumption**: Correlated stocks move together
- **Signal**: Z-score of simple spread (Stock A - Stock B)
- **Pros**: Simple, fast computation
- **Cons**: Correlation may not be stable

### 🔗 **Cointegration-Based Strategy**  
- **Method**: Statistical cointegration testing (Engle-Granger)
- **Assumption**: Long-term equilibrium relationship
- **Signal**: Z-score of cointegrating spread (Stock A - β×Stock B)
- **Pros**: Theoretically sound, stable relationships
- **Cons**: More complex, fewer pairs found

## 🎯 Key Features

### ✅ **Fixed Visualization Issues**
- **No overlapping elements** in plots
- **Proper spacing** between subplots
- **Clear titles and labels**
- **Professional appearance**

### 📈 **Comprehensive Analysis**
- **Statistical significance testing**
- **Performance metrics** (PnL, Sharpe ratio, win rate)
- **Trade-by-trade analysis**
- **Strategy comparison plots**

### 🔧 **Flexible Parameters**
- **Correlation threshold** (0.70-0.95)
- **Cointegration significance** (0.01-0.10)
- **Entry/exit Z-scores** (1.0-3.0)
- **Focus stocks** for targeted analysis

## 📊 Usage Examples

### **Basic Comparison**
```bash
python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m
```

### **Custom Parameters**
```bash
python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m \
  --correlation 0.85 \
  --significance 0.01 \
  --entry-zscore 1.5 \
  --exit-zscore 0.3 \
  --plot
```

### **Focus on Specific Stocks**
```bash
python strategy_comparison.py ../data/pair_trading/sp500_20180101_20190703_log_returns_12m6m \
  --focus-stocks AMZN GOOGL MSFT AAPL NVDA \
  --plot
```

## 📋 Output Files

Each strategy generates:
- **Performance metrics** in terminal
- **Statistical tests** results
- **Visualization plots** (if --plot flag used)
- **Saved PNG files** in data directory

## 🎯 Best Practices

1. **Start with comparison script** to see both approaches
2. **Use pre-COVID data** (2018-2019) for better mean reversion
3. **Adjust parameters** based on market regime
4. **Focus on fewer stocks** for cointegration analysis
5. **Check statistical significance** before trading

## 📊 Expected Results

### **Pre-COVID Period (2018-2019)**
- **Better cointegration relationships**
- **More mean-reverting spreads**
- **Higher success rates**

### **Recent Period (2023-2024)**
- **Trending markets** (challenging for both strategies)
- **Fewer profitable opportunities**
- **Strategy selection matters more**

## 🔍 Troubleshooting

### **No trades executed?**
- Lower Z-score thresholds (try 1.5/0.3)
- Increase correlation threshold to 0.70
- Use focus stocks for cointegration

### **Poor performance?**
- Check market regime (trending vs mean-reverting)
- Try different time periods
- Adjust formation/trading period split

### **Visualization issues?**
- All plots use fixed spacing to prevent overlaps
- PNG files saved automatically with --plot flag
- Check data directory for output files

---

💡 **Tip**: Start with `strategy_comparison.py` to get a complete overview of both approaches! 