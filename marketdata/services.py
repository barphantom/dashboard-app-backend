import os
import datetime
from asyncio import timeout

import requests
from django.core.cache import cache
from dotenv import load_dotenv

load_dotenv()

def fetch_ohlcv_data(symbol: str):
    symbol = symbol.upper()
    cache_key = f"ohlcv_{symbol}"
    cached_data = cache.get(cache_key)

    if cached_data:
        print(f"✅ Using cached OHLCV for {symbol}")
        return cached_data

    print(f"🔄 Fetching OHLCV from Alpha Vantage for {symbol}")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
        "outputsize": "full",
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Alpha Vantage API: {e}")

    if "Note" in data:
        raise ValueError("Alpha Vantage API limit reached")
    if "Time Series (Daily)" not in data:
        raise ValueError(f"No data for symbol {symbol}")

    raw_data = data["Time Series (Daily)"]
    cache.set(cache_key, raw_data, timeout=60 * 60 * 24)  # 24h

    return raw_data


def filter_ohlcv_last_year(raw_data: dict, days: int = 365):
    one_year_ago = datetime.datetime.now() - datetime.timedelta(days=days)
    chart_data = []

    for date, value in raw_data.items():
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        if date_obj > one_year_ago:
            chart_data.append({
                "date": date,
                "open": float(value["1. open"]),
                "high": float(value["2. high"]),
                "low": float(value["3. low"]),
                "close": float(value["4. close"]),
                "volume": int(value["5. volume"]),
            })

    chart_data.sort(key=lambda x: datetime.datetime.strptime(x["date"], "%Y-%m-%d"))

    return chart_data


def get_company_name(symbol: str):
    symbol = symbol.upper()

    cached_name = cache.get(f"company_name_{symbol}")
    if cached_name:
        return cached_name

    print(f"🔄 Fetching company name from FinnHub for {symbol}")
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {
        "symbol": symbol,
        "token": os.getenv("FINNHUB_API_KEY"),
    }

    try:
        response = requests.get(url, params)
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Alpha Vantage API: {e}")

    name = data.get("name")
    if not name:
        print(f"⚠️ Nie udało się pobrać nazwy spółki dla {symbol}: {data}")
        return symbol

    cache.set(f"company_name_{symbol}", name, timeout=60 * 60 * 24)

    return name