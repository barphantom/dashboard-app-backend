import datetime, os, requests
from dotenv import load_dotenv
from django.core.cache import cache
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from marketdata.services import fetch_ohlcv_data, filter_ohlcv_last_year

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

        try:
            raw_data = fetch_ohlcv_data(symbol)
            one_year_chart_data = filter_ohlcv_last_year(raw_data=raw_data)
        except ConnectionError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except ValueError as e:
            msg = str(e)
            if "limit" in msg.lower():
                return Response({"error": msg}, status=status.HTTP_429_TOO_MANY_REQUESTS)
            return Response({"error": msg}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            "symbol": symbol.upper(),
            "interval": "daily",
            "data": one_year_chart_data,
        })
