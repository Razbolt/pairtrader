import yfinance as yf
import pandas as pd
from typing import List


def get_sp500_tickers() -> List[str]:
    """Return a list of S&P 500 tickers using yfinance."""
    table = yf.Ticker("^GSPC").history(period="1d")  # ensures internet connectivity
    sp500 = yf.download("^GSPC", period="1d")
    # Placeholder: In practice, fetch component tickers via API or static file
    # Using a small subset for example purposes
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def download_price_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    data = data.dropna(axis=0, how="all")
    return data

