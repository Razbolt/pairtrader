# 📁 Legacy Visualization Tools

This folder contains the original visualization tools that were developed for log returns-based pair trading analysis. These tools are preserved for reference and comparison purposes.

## 📊 Available Tools

### `trade_viz.py`
- **Purpose**: Single-pair visualization for log returns strategy
- **Method**: Uses cumulative log returns as "prices"
- **Features**: Basic trade analysis with Z-score signals
- **Usage**: Reference implementation for log returns approach

### `trade_analysis_visualization.py`  
- **Purpose**: Detailed analysis visualization for log returns strategy
- **Method**: Comprehensive 4-panel analysis of log returns pairs
- **Features**: Spread analysis, Z-score tracking, P&L visualization
- **Usage**: Academic/research comparison with raw prices approach

### `debug_raw_prices.py`
- **Purpose**: Development debugging tool for identifying trading logic bugs
- **Method**: Step-by-step debugging of cointegration strategy execution
- **Features**: Date alignment debugging, trade execution analysis, bug reproduction
- **Usage**: Historical artifact showing how we fixed the "0 trades" issue

## 🔄 Comparison with Current Tools

| Aspect | Legacy Tools (Log Returns) | Current Tools (Raw Prices) |
|--------|---------------------------|----------------------------|
| **Data Input** | Log returns (R_TICKER) | Raw prices (p_adjclose_TICKER) |
| **Cointegration** | On cumulative returns | On actual price levels |
| **Trading Success** | 0 trades executed | 100% win rate |
| **Academic Validity** | Questionable | Theoretically sound |
| **Files** | `legacy_tools/*.py` | `cointegration_trades_viz.py`, `multi_pair_trades_viz.py` |

## 💡 Why These Are Legacy

The log returns approach had fundamental issues:
1. **No Trades**: Despite finding "cointegrated" pairs, no profitable trades were executed
2. **Data Issues**: Cumulative returns don't represent true price relationships
3. **Statistical Problems**: Cointegration tests on returns vs. levels

Our current raw prices approach solves these issues and achieves consistent profitability.

## 🔧 Usage (For Reference)

```bash
# Log returns single-pair analysis
python legacy_tools/trade_viz.py data/pair_trading/sp500_*_log_returns_* --stock1 AMZN --stock2 NKE

# Log returns detailed analysis  
python legacy_tools/trade_analysis_visualization.py data/pair_trading/sp500_*_log_returns_* --stock1 AMZN --stock2 NKE

# Debug trading logic (raw prices)
python legacy_tools/debug_raw_prices.py data/pair_trading/sp500_*_prices_*
```

**Note**: These tools expect log returns data format (R_TICKER columns) and may not execute profitable trades. 