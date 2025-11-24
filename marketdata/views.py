import datetime, os, requests
from django.utils import timezone
from dotenv import load_dotenv
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

        response = {
            "symbol": symbol.upper(),
            "interval": "daily",
            "data": [],
        }

        try:
            full_df = fetch_ohlcv_data(symbol)
            if full_df.empty:
                return Response(response, status=status.HTTP_200_OK)

            five_years_ago = timezone.now().date() - datetime.timedelta(days=365 * 5)
            mask = full_df["Date"] >= five_years_ago

            df_filtered = full_df.loc[mask].copy()
            if df_filtered.empty:
                return Response(response, status=status.HTTP_200_OK)

            df_filtered["date_str"] = df_filtered["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df_filtered = df_filtered.rename(columns={
                "date_str": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            final_cols = ["date", "open", "high", "low", "close", "volume"]
            data = df_filtered[final_cols].to_dict(orient="records")

            response["data"] = data

        except Exception as e:
            msg = f"❌ Error fetching ohlcv data for {symbol}: {e}"
            return Response({"error": msg}, status=status.HTTP_404_NOT_FOUND)

        return Response(response, status=status.HTTP_200_OK)
