import os, requests, datetime
from dotenv import load_dotenv
from django.core.cache import cache
from rest_framework import status, generics, serializers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import NotFound
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from decimal import Decimal
import pandas as pd

from portfolio.models import Portfolio, PortfolioStock
from portfolio.serializers import PortfolioSerializer, PortfolioStockSerializer
from marketdata.services import fetch_ohlcv_data, filter_ohlcv_last_year, get_company_name

load_dotenv()

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_active_portfolio(request):
    portfolio = Portfolio.objects.filter(owner=request.user).first()
    if not portfolio:
        return Response({"detail": "No portfolio was found!"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"id": portfolio.id})


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

        symbol = serializer.validated_data.get("symbol")
        company_name = get_company_name(symbol)

        serializer.save(portfolio=portfolio, name=company_name)


class PortfolioStockDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API endpoint for retrieving, updating, and deleting a specific portfolio stock position.

    Endpoints:
        GET    /portfolios/<portfolio_id>/stocks/<position_id>/  - Retrieve position details
        PUT    /portfolios/<portfolio_id>/stocks/<position_id>/  - Update position
        PATCH  /portfolios/<portfolio_id>/stocks/<position_id>/  - Partial update
        DELETE /portfolios/<portfolio_id>/stocks/<position_id>/  - Delete position
    """
    serializer_class = PortfolioStockSerializer
    permission_classes = [IsAuthenticated]
    lookup_url_kwarg = "position_id"

    def get_queryset(self):
        return PortfolioStock.objects.filter(portfolio__owner=self.request.user)

    def get_object(self):
        portfolio_id = self.kwargs['portfolio_id']
        position_id = self.kwargs['position_id']

        portfolio = get_object_or_404(Portfolio, id=portfolio_id, owner=self.request.user)

        try:
            position = PortfolioStock.objects.get(
                id=position_id,
                portfolio=portfolio,
            )
            return position
        except PortfolioStock.DoesNotExist:
            raise NotFound(f"Position with id {position_id} not found in portfolio {portfolio_id}")

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)

        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response({
            "message": "Position updated successfully",
            "data": serializer.data
        }, status=status.HTTP_200_OK)

    def destroy(self, request, *args, **kwargs):
        """
        Handle DELETE requests to remove a position from portfolio.
        """
        instance = self.get_object()
        symbol = instance.symbol
        instance.delete()

        return Response({
            "message": f"Position {symbol} deleted successfully"
        }, status=status.HTTP_200_OK)

    def retrieve(self, request, *args, **kwargs):
        """
        Handle GET requests to retrieve position details.
        """
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data, status=status.HTTP_200_OK)


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

            try:
                raw_data = fetch_ohlcv_data(symbol)
                chart_data = filter_ohlcv_last_year(raw_data)
            except ConnectionError as e:
                print(f"⚠️ Connection error for {symbol}: {e}")
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except ValueError as e:
                msg = str(e)
                if "limit" in msg.lower():
                    return Response({"error": msg}, status=status.HTTP_429_TOO_MANY_REQUESTS)
                print(f"⚠️ Skipping {symbol} - {msg}")
                continue  # nie przerywaj, tylko pomiń ten ticker

            if not chart_data or len(chart_data) < 6:
                print(f"Skipping {symbol} - insufficient data")
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

        if total_cost == 0:
            return Response({"error": "Portfolio has no valid stocks with data"}, status=status.HTTP_400_BAD_REQUEST)

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

        dfs = []
        errors = []

        for stock in stocks:
            symbol = stock.symbol.upper()
            quantity = Decimal(stock.quantity)
            buy_date = stock.purchase_date

            try:
                raw_data = fetch_ohlcv_data(symbol)
            except ConnectionError as e:
                errors.append(f"Connection error for {symbol}: {e}")
                continue
            except ValueError as e:
                msg = str(e)
                if "limit" in msg.lower():
                    return Response({"error": msg}, status=status.HTTP_429_TOO_MANY_REQUESTS)
                errors.append(f"Skipping {symbol} - {msg}")
                continue
            except Exception as e:
                errors.append(f"Unexpected error for {symbol}: {e}")
                continue

            # Filtrujemy dane tylko od daty zakupu
            data = []
            for date, value in raw_data.items():
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                if date_obj >= buy_date:
                    data.append({
                        "date": date_obj,
                        "value": float(value["4. close"]) * float(quantity),
                    })

            if not data:
                continue

            df = pd.DataFrame(data)
            df = df.sort_values("date").set_index("date")
            df.rename(columns={"value": symbol}, inplace=True)
            dfs.append(df)

        if not dfs:
            return Response({"error": "Brak danych dla wykresu portfela"}, status=status.HTTP_400_BAD_REQUEST)

        # Połącz dane po dacie i wypełnij brakujące dni ostatnią znaną wartością
        combined = pd.concat(dfs, axis=1)
        combined = combined.asfreq("D", method="ffill")

        # Usuń wiersze, gdzie wszystko jest NaN (przed datą zakupu pierwszej spółki)
        combined = combined.dropna(how="all")

        # Oblicz całkowitą wartość portfela dla każdego dnia
        combined["total"] = combined.sum(axis=1)

        # Przygotuj dane do zwrotu
        chart_data_response = [
            {
                "time": date.strftime("%Y-%m-%d"),
                "value": round(float(value), 2),
            }
            for date, value in combined["total"].items()
        ]

        response = {"chart": chart_data_response}
        if errors:
            response["warnings"] = errors

        return Response(response, status=status.HTTP_200_OK)


class PortfolioCompositionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

        stocks = portfolio.stocks.all()
        if not stocks:
            return Response([], status=status.HTTP_200_OK)

        total_value = Decimal(0)
        data = []
        errors = []

        for stock in stocks:
            symbol = stock.symbol
            quantity = Decimal(stock.quantity)

            try:
                chart_data = fetch_ohlcv_data(symbol)
                first_key = next(iter(chart_data))
                current_price = Decimal(chart_data[first_key]["4. close"])
            except ConnectionError as e:
                errors.append(f"Connection error for {symbol}: {e}")
                continue
            except ValueError as e:
                msg = str(e)
                if "limit" in msg.lower():
                    return Response({"error": msg}, status=status.HTTP_429_TOO_MANY_REQUESTS)
                errors.append(f"Skipping {symbol} - {msg}")
                continue
            except Exception as e:
                errors.append(f"Unexpected error for {symbol}: {e}")
                continue

            value = quantity * current_price
            total_value += value
            data.append({
                "id": symbol,
                "label": stock.name or symbol,
                "value": round(float(value), 2),
            })

        if total_value > 0:
            for item in data:
                item["percentage"] = round(float(Decimal(item["value"]) / total_value * 100), 2)

        response = {"data": data}
        if errors:
            response["warnings"] = errors

        return Response(response, status=status.HTTP_200_OK)
