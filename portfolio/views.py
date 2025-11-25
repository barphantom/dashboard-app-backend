import datetime
from django.utils import timezone
from dotenv import load_dotenv
from rest_framework import status, generics, serializers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import NotFound
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
import pandas as pd

from portfolio.models import Portfolio, PortfolioStock
from portfolio.serializers import PortfolioSerializer, PortfolioStockSerializer
from marketdata.services import fetch_ohlcv_data, get_company_name, get_latest_prices

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
        # company_name = get_company_profile(symbol)["name"]

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

        total_current_value = 0.0
        total_cost_basis = 0.0
        total_value_week_ago = 0.0

        week_ago_date = timezone.now().date() - datetime.timedelta(days=7)

        for stock in stocks:
            symbol = stock.symbol
            quantity = float(stock.quantity)
            purchase_price = float(stock.purchase_price)

            try:
                df = fetch_ohlcv_data(symbol)
                if df.empty:
                    print(f"⚠️ Skipping {symbol} - no data returned")
                    continue

                current_price = float(df.iloc[-1]["Close"])
                past_data = df.loc[df["Date"] <= week_ago_date]

                if not past_data.empty:
                    price_week_ago = float(past_data.iloc[-1]["Close"])
                else:
                    price_week_ago = float(df.iloc[0]["Close"])

                total_current_value += current_price * quantity
                total_cost_basis += purchase_price * quantity
                total_value_week_ago += price_week_ago * quantity

            except Exception as e:
                print(f"❌ Error calculating stats for {symbol}: {e}")
                continue

        if total_cost_basis > 0:
            profit_value = total_current_value - total_cost_basis
            profit_percent = (profit_value / total_cost_basis) * 100
        else:
            profit_value = 0
            profit_percent = 0

        if total_value_week_ago > 0:
            weekly_change_percent = ((total_current_value - total_value_week_ago) / total_value_week_ago) * 100
        else:
            weekly_change_percent = 0

        return Response({
            "total_value": round(total_current_value, 2),
            "profit_value": round(profit_value, 2),
            "profit_percent": round(profit_percent, 2),
            "weekly_change_percent": round(weekly_change_percent, 2),
        })


class PortfolioChartView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        response = {
            "chart": [],
            "warnings": [],
            "error": None,
        }

        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            response["error"] = "Portfolio not found"
            return Response(response, status=status.HTTP_404_NOT_FOUND)

        stocks = portfolio.stocks.all()
        if not stocks:
            return Response(response, status=status.HTTP_200_OK)

        dfs = []
        warnings = []

        earliest_buy_date = min(stock.purchase_date for stock in stocks)

        for stock in stocks:
            symbol = stock.symbol.upper()
            quantity = float(stock.quantity)
            buy_date = stock.purchase_date

            try:
                full_df = fetch_ohlcv_data(symbol)
                if full_df.empty:
                    warnings.append(f"No data for {symbol}")
                    continue

                mask = full_df['Date'] >= buy_date
                df_filtered = full_df.loc[mask].copy()

                df_filtered[symbol] = df_filtered['Close'] * quantity
                df_filtered = df_filtered.set_index('Date')
                dfs.append(df_filtered[[symbol]])

            except Exception as e:
                warnings.append(f"Error processing {symbol}: {str(e)}")
                continue

        if not dfs:
            response["warnings"] = warnings
            return Response(response, status=status.HTTP_200_OK)

        combined = pd.concat(dfs, axis=1)
        all_days = pd.date_range(start=earliest_buy_date, end=datetime.date.today(), freq='D').date
        combined = combined.reindex(all_days)

        combined = combined.ffill()
        combined = combined.fillna(0)

        combined["total"] = combined.sum(axis=1)

        chart = [
            {"time": date.strftime("%Y-%m-%d"), "value": round(val, 2)}
            for date, val in combined["total"].items()
            if val > 0
        ]

        response["chart"] = chart
        response["warnings"] = warnings

        return Response(response, status=status.HTTP_200_OK)


class PortfolioCompositionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        response = {
            "data": [],
            "warnings": [],
            "error": None,
        }

        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

        data = []
        warnings = []

        stocks = portfolio.stocks.all()
        if not stocks:
            response["warnings"] = ["No stocks found"]
            return Response(response, status=status.HTTP_200_OK)

        total_value = 0

        for stock in stocks:
            symbol = stock.symbol.upper()
            quantity = float(stock.quantity)

            try:
                full_df = fetch_ohlcv_data(symbol)
                if full_df.empty:
                    warnings.append(f"No data for {symbol}")
                    continue

                current_price = float(full_df.iloc[-1]["Close"])
                total_current_value = current_price * quantity
                total_value += total_current_value

                data.append({
                    "id": symbol,
                    "label": stock.name or symbol,
                    "value": round(total_current_value, 2),
                })

            except Exception as e:
                warnings.append(f"Error processing {symbol}: {str(e)}")
                continue

        if total_value > 0:
            data.sort(key=lambda x: x["value"], reverse=True)
            for stock in data:
                stock["percentage"] = round((stock["value"] / total_value) * 100, 2)
        else:
            warnings.append("Total portfolio value is zero - cannot calculate percentages")

        response["data"] = data
        response["warnings"] = warnings

        return Response(response, status=status.HTTP_200_OK)


class PortfolioQuotesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id):
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, owner=request.user)
        except Portfolio.DoesNotExist:
            return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

        stocks = portfolio.stocks.all()
        symbols = [stock.symbol for stock in stocks]

        if not symbols:
            return Response({}, status=status.HTTP_200_OK)

        current_prices = get_latest_prices(symbols)

        return Response(current_prices, status=status.HTTP_200_OK)
