"""
FinBot — backend/market_data.py
================================
Live market data fetcher using yfinance.
Supports Indian stocks (NSE/BSE) and global stocks.

Usage:
    from backend.market_data import get_stock_data, get_index_data
"""

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf

from backend.schemas import MarketData, IndexData

logger = logging.getLogger(__name__)


# ============================================================
# STOCK DATA
# ============================================================

def get_stock_data(ticker: str) -> MarketData:
    """
    Fetch current stock price and metadata using yfinance.
    Automatically appends .NS for Indian NSE stocks if no exchange suffix.

    Args:
        ticker: Stock ticker symbol (e.g. "RELIANCE", "AAPL", "TCS.NS")

    Returns:
        MarketData pydantic object with price and metadata
    """
    try:
        # Add .NS suffix for Indian stocks if no suffix present
        original_ticker = ticker.upper()
        if "." not in original_ticker:
            yf_ticker = original_ticker + ".NS"
        else:
            yf_ticker = original_ticker

        logger.info(f"Fetching stock data for: {yf_ticker}")
        stock = yf.Ticker(yf_ticker)
        info  = stock.info

        # Get current price
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 0.0
        )

        # Get % change
        prev_close     = info.get("previousClose", current_price)
        change_percent = 0.0
        if prev_close and prev_close != 0:
            change_percent = round(
                ((current_price - prev_close) / prev_close) * 100, 2
            )

        # Format market cap
        market_cap_raw = info.get("marketCap")
        if market_cap_raw:
            if market_cap_raw >= 1_00_00_00_000:   # 1000 Cr+
                market_cap = f"₹{market_cap_raw / 1_00_00_00_000:.1f}T"
            elif market_cap_raw >= 1_00_00_000:    # 1 Cr+
                market_cap = f"₹{market_cap_raw / 1_00_00_000:.0f}Cr"
            else:
                market_cap = f"₹{market_cap_raw:,.0f}"
        else:
            market_cap = "N/A"

        return MarketData(
            ticker=original_ticker,
            current_price=round(float(current_price), 2),
            change_percent=change_percent,
            currency=info.get("currency", "INR"),
            market_cap=market_cap,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exchange=info.get("exchange", "NSE"),
        )

    except Exception as e:
        logger.error(f"Failed to fetch stock data for {ticker}: {str(e)}")
        raise ValueError(f"Could not fetch data for ticker '{ticker}': {str(e)}")


# ============================================================
# INDEX DATA
# ============================================================

def get_index_data() -> IndexData:
    """
    Fetch current Nifty 50 and Sensex index levels.

    Returns:
        IndexData pydantic object with both indices
    """
    try:
        logger.info("Fetching Nifty 50 and Sensex data...")

        nifty_ticker  = yf.Ticker("^NSEI")
        sensex_ticker = yf.Ticker("^BSESN")

        nifty_info  = nifty_ticker.info
        sensex_info = sensex_ticker.info

        def get_change(info: dict) -> float:
            current  = info.get("regularMarketPrice", 0)
            previous = info.get("previousClose", current)
            if previous and previous != 0:
                return round(((current - previous) / previous) * 100, 2)
            return 0.0

        return IndexData(
            nifty=round(
                float(nifty_info.get("regularMarketPrice", 0)), 2
            ),
            sensex=round(
                float(sensex_info.get("regularMarketPrice", 0)), 2
            ),
            nifty_change_percent=get_change(nifty_info),
            sensex_change_percent=get_change(sensex_info),
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as e:
        logger.error(f"Failed to fetch index data: {str(e)}")
        return IndexData(
            nifty=None,
            sensex=None,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


# ============================================================
# TICKER EXTRACTOR
# ============================================================

def extract_ticker_from_query(query: str) -> Optional[str]:
    """
    Extract stock ticker symbol from a natural language query.
    Handles common Indian and global stock names.

    Args:
        query: User's financial question

    Returns:
        Ticker string or None if not found
    """
    # Common Indian stock name → ticker mapping
    indian_stocks = {
        "reliance": "RELIANCE",
        "tcs": "TCS",
        "tata consultancy": "TCS",
        "infosys": "INFY",
        "hdfc bank": "HDFCBANK",
        "hdfc": "HDFCBANK",
        "icici bank": "ICICIBANK",
        "icici": "ICICIBANK",
        "sbi": "SBIN",
        "state bank": "SBIN",
        "wipro": "WIPRO",
        "hcl": "HCLTECH",
        "bajaj": "BAJFINANCE",
        "kotak": "KOTAKBANK",
        "axis bank": "AXISBANK",
        "axis": "AXISBANK",
        "maruti": "MARUTI",
        "asian paints": "ASIANPAINT",
        "itc": "ITC",
        "ultratech": "ULTRACEMCO",
        "nestle": "NESTLEIND",
        "titan": "TITAN",
        "adani": "ADANIENT",
    }

    query_lower = query.lower()

    # Check Indian stock names
    for name, ticker in indian_stocks.items():
        if name in query_lower:
            return ticker

    # Check for explicit ticker symbols (e.g. "RELIANCE", "AAPL")
    import re
    ticker_pattern = r'\b[A-Z]{2,10}(?:\.NS|\.BO)?\b'
    matches = re.findall(ticker_pattern, query)

    # Filter out common non-ticker words
    excluded = {
        "SIP", "EMI", "ITR", "NSE", "BSE", "RBI", "SEBI",
        "ULIP", "ELSS", "PPF", "EPF", "NPS", "FD", "ITD",
        "CIBIL", "PMAY", "HRA", "TDS", "GST", "PAN", "KYC"
    }
    for match in matches:
        clean = match.replace(".NS", "").replace(".BO", "")
        if clean not in excluded:
            return clean

    return None
