import os
import yfinance as yf
import pandas as pd
import requests
from django.core.cache import cache
from dotenv import load_dotenv
from django.utils import timezone
from datetime import timedelta, datetime, time

load_dotenv()

def filter_ohlcv_last_year(raw_data: dict, days: int = 365):
    one_year_ago = datetime.now() - timedelta(days=days)
    chart_data = []

    for date, value in raw_data.items():
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        if date_obj > one_year_ago:
            chart_data.append({
                "date": date,
                "open": float(value["1. open"]),
                "high": float(value["2. high"]),
                "low": float(value["3. low"]),
                "close": float(value["4. close"]),
                "volume": int(value["5. volume"]),
            })

    chart_data.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))

    return chart_data

def fetch_ohlcv_data(symbol: str):
    symbol = symbol.upper()
    cache_key = f"ohlcv_yf_{symbol}"
    cached_df = cache.get(cache_key)

    if cached_df is not None and not cached_df.empty:
        print(f"✅ Using cached OHLCV for {symbol}")
        return cached_df

    print(f"🔄 Fetching history from yfinance for {symbol}")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", auto_adjust=True)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = df['Date'].dt.date

        cache.set(cache_key, df, timeout=get_seconds_until_midnight())

        return df

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        raise e

def get_seconds_until_midnight():
    now = timezone.now()
    tomorrow = now + timedelta(days=1)
    midnight = datetime.combine(tomorrow, time.min)

    if timezone.is_aware(now):
        current_tz = timezone.get_current_timezone()
        midnight = timezone.make_aware(midnight, current_tz)

    delta = midnight - now
    return int(delta.total_seconds())

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

def get_latest_prices(symbols: list):
    if not symbols:
        return {}

    symbols = [symbol.upper() for symbol in symbols]
    unique_symbols = list(set(symbols))

    tickers_str = " ".join(unique_symbols)
    print(f"🔄 Batch fetching prices for: {tickers_str}")

    try:
        data = yf.download(tickers_str, period="1d", group_by="ticker", progress=False, threads=True)
        prices = {}

        if len(unique_symbols) == 1:
            symbol = unique_symbols[0]

            if not data.empty:
                prices[symbol] = round(float(data["Close"].iloc[-1]), 2)
        else:
            for symbol in unique_symbols:
                try:
                    symbol_data = data[symbol]

                    if not symbol_data.empty:
                        prices[symbol] = round(float(symbol_data["Close"].iloc[-1]), 2)
                except KeyError:
                    print(f"⚠️ No data found for {symbol} in batch response")
                    continue

        return prices

    except Exception as e:
        print(f"❌ Error batch fetching prices: {e}")
        return {}
