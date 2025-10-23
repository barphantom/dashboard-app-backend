import os, requests, datetime
from dotenv import load_dotenv
from django.core.cache import cache
from rest_framework import status, generics, serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from portfolio.models import Portfolio, PortfolioStock
from portfolio.serializers import PortfolioSerializer, PortfolioStockSerializer

load_dotenv()

class PortfolioListCreateView(generics.ListCreateAPIView):
    serializer_class = PortfolioSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Portfolio.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class PortfolioDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = PortfolioSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Portfolio.objects.filter(owner=self.request.user)


class PortfolioStockListCreateView(generics.ListCreateAPIView):
    serializer_class = PortfolioStockSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        portfolio_id = self.kwargs['portfolio_id']
        return PortfolioStock.objects.filter(portfolio__id=portfolio_id, portfolio__owner=self.request.user)

    def perform_create(self, serializer):
        portfolio_id = self.kwargs['portfolio_id']
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=self.request.user)
        except Portfolio.DoesNotExist:
            raise serializers.ValidationError({
                "error": f"Portfolio with id {portfolio_id} does not exist or you don't have permission to access it."
            })

        serializer.save(portfolio=portfolio)


class PortfolioStockDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = PortfolioStockSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return PortfolioStock.objects.filter(portfolio__owner=self.request.user)


class PortfolioStatsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

        stocks = portfolio.stocks.all()
        if not stocks:
            return Response({
                "total_value": 0,
                "profit_value": 0,
                "profit_percent": 0,
                "weekly_change_percent": 0,
            })

        total_value = 0
        total_cost = 0
        total_value_week_ago = 0

        for stock in stocks:
            symbol = stock.symbol
            quantity = float(stock.quantity)

            cache_key = f"ohlcv_{symbol.upper()}"
            cached_data = cache.get(cache_key)

            if cached_data:
                print(f"✅ Using cached OHLCV for {symbol}")
                chart_data = cached_data
            else:
                print(f"🔄 Fetching data from Alpha Vantage for {symbol}")
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
                    return Response({"error": f"Failed to connect to API: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                if "Note" in data:
                    return Response({"error": "Alpha Vantage API limit reached"}, status=status.HTTP_429_TOO_MANY_REQUESTS)
                if "Time Series (Daily)" not in data:
                    return Response({"error": f"No data for symbol {symbol}"}, status=status.HTTP_404_NOT_FOUND)

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

            if not chart_data or len(chart_data) < 6:
                print(f"Skipping {symbol}!!!")
                continue

            latest_data = chart_data[-1]
            current_price = float(latest_data["close"])

            week_ago_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            price_week_ago = current_price
            for date in reversed(chart_data[:-1]):
                if date["date"] <= week_ago_date:
                    price_week_ago = date["close"]
                    break

            if not price_week_ago:
                price_week_ago = chart_data[-6]["close"]

            total_value += current_price * quantity
            total_cost += float(stock.purchase_price) * quantity
            total_value_week_ago += price_week_ago * quantity

        profit_value = total_value - total_cost
        profit_percent = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        weekly_change_percent = ((total_value - total_value_week_ago) / total_value_week_ago) * 100 if total_value_week_ago > 0 else 0

        return Response({
            "total_value": round(total_value, 2),
            "profit_value": round(profit_value, 2),
            "profit_percent": round(profit_percent, 2),
            "weekly_change_percent": round(weekly_change_percent, 2),
        })


class PortfolioChartView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

        stocks = portfolio.stocks.all()
        if not stocks:
            return Response({"error": "Portfolio jest puste"}, status=status.HTTP_400_BAD_REQUEST)

        chart_values = dict()

        for stock in stocks:
            symbol = stock.symbol
            quantity = float(stock.quantity)

            cache_key = f"ohlcv_{symbol.upper()}"
            cached_data = cache.get(cache_key)

            if cached_data:
                print(f"✅ Using cached OHLCV for {symbol}")
                chart_data = cached_data
            else:
                print(f"🔄 Fetching data from Alpha Vantage for {symbol}")
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
                    return Response({"error": f"Failed to connect to API: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                if "Note" in data:
                    return Response({"error": "Alpha Vantage API limit reached"}, status=status.HTTP_429_TOO_MANY_REQUESTS)
                if "Time Series (Daily)" not in data:
                    return Response({"error": f"No data for symbol {symbol}"}, status=status.HTTP_404_NOT_FOUND)

                chart_data = data["Time Series (Daily)"]
                cache.set(cache_key, chart_data, timeout=60 * 60 * 24)

            for date, value in chart_data.items():
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                buy_date = stock.purchase_date
                if date_obj >= buy_date:
                    if date in chart_values:
                        chart_values[date] += float(value["4. close"]) * quantity
                    else:
                        chart_values[date] = float(value["4. close"]) * quantity

        sorted_dates = sorted(chart_values.keys())
        chart_data_response = [
            {
                "time": date_str,
                "value": round(chart_values[date_str], 2),
            }
            for date_str in sorted_dates
        ]

        return Response(chart_data_response)
