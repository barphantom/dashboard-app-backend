from django.urls import path
from .views import PortfolioListCreateView, PortfolioDetailView, PortfolioStockListCreateView, PortfolioStockDetailView, \
    PortfolioStatsView, PortfolioChartView

urlpatterns = [
    path('portfolios/', PortfolioListCreateView.as_view(), name='portfolio-list-create'),
    path('portfolios/<int:pk>', PortfolioDetailView.as_view(), name='portfolio-detail'),

    path('portfolios/<int:portfolio_id>/stocks/', PortfolioStockListCreateView.as_view(), name='portfolio-stock-list-create'),
    path('stocks/<int:pk>', PortfolioStockDetailView.as_view(), name='portfolio-stock-detail'),
    path('portfolio/<int:portfolio_id>/stats/', PortfolioStatsView.as_view(), name='portfolio-stats'),
    path('portfolio/<int:portfolio_id>/chart/', PortfolioChartView.as_view(), name='portfolio-chart')
]
