import yfinance as yf
import pandas as pd
import requests
from typing import List
import io


def get_sp500_tickers() -> List[str]:
    """Return a list of S&P 500 tickers.

    Attempts to fetch the component list from Wikipedia. If the request fails
    (e.g. due to lack of internet access), a fallback list of large cap stocks
    is returned.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = requests.get(url, timeout=10).text
        tables = pd.read_html(io.StringIO(html), flavor="bs4")
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
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
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )["Close"]
    except Exception as exc:
        raise RuntimeError("Failed to download price data") from exc

    data = data.dropna(axis=0, how="all")
    return data

