import datetime, os, requests
from dotenv import load_dotenv
from django.core.cache import cache
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

load_dotenv()

class StockSearchView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        query = request.GET.get("query", "").strip()

        if not query:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

        api_key = os.getenv("FINNHUB_API_KEY")
        us_exchange = "US"
        finnhub_url = f"https://finnhub.io/api/v1/search?q={query}&token={api_key}&exchange={us_exchange}"

        response = requests.get(finnhub_url)
        if response.status_code != 200:
            return Response({"error": "External API error"}, status=status.HTTP_502_BAD_GATEWAY)

        data = response.json()
        if "result" not in data:
            return Response({"error": "Unexpected API format"}, status=status.HTTP_502_BAD_GATEWAY)

        results = []
        for item in data["result"]:
            if item.get("symbol") and item.get("description"):
                results.append({
                    "symbol": item["symbol"],
                    "name": item["description"],
                })

        return Response(results, status=status.HTTP_200_OK)


class OHLCVView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        symbol = request.query_params.get("symbol")
        if not symbol:
            return Response({"error": "Query parameter 'symbol' is required"}, status=status.HTTP_400_BAD_REQUEST)

        cache_key = f"ohlcv_{symbol.upper()}"
        cached_data = cache.get(cache_key)

        if cached_data:
            print(f"✅ Using cached OHLCV data for {symbol.upper()}")
            return Response({
                "symbol": symbol.upper(),
                "interval": "daily",
                "data": cached_data,
            })

        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "apikey": os.getenv("ALPHAVANTAGE_API_KEY"),
            "outputsize": "full",
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
        except requests.exceptions.RequestException as e:
            return Response({"error": "Failed to connect to Alpha Vantage: " + str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if "Note" in data:
            return Response(
                {"error": "API limit reached or temporary unavailability"},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        if "Time Series (Daily)" not in data:
            return Response({"error": "No data available for given symbol"}, status=status.HTTP_404_NOT_FOUND)

        time_series = data["Time Series (Daily)"]
        one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)

        chart_data = []
        for date, value in time_series.items():
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

        cache.set(cache_key, chart_data, timeout=60 * 60 * 24)

        return Response({
            "symbol": symbol.upper(),
            "interval": "daily",
            "data": chart_data,
        })


