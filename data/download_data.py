import yfinance as yf
import pandas as pd
import requests
from typing import List


def get_sp500_tickers() -> List[str]:
    """Return a list of S&P 500 tickers.

    Attempts to fetch the component list from Wikipedia. If the request fails
    (e.g. due to lack of internet access), a fallback list of large cap stocks
    is returned.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(requests.get(url, timeout=10).text)
        tickers = tables[0]["Symbol"].tolist()
    except Exception:
        # Fallback list (20 tickers) if network access is not available
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "BRK-B", "JPM", "JNJ", "V",
            "PG", "XOM", "UNH", "HD", "MA",
            "CVX", "LLY", "MRK", "ABBV", "PEP",
        ]
    return tickers[:20]


def download_price_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        progress=False,
        threads=False,
    )["Adj Close"]
    data = data.dropna(axis=0, how="all")
    return data

