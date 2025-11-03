from django.urls import path
from .views import PortfolioListCreateView, PortfolioDetailView, PortfolioStockListCreateView, PortfolioStockDetailView, \
    PortfolioStatsView, PortfolioChartView, PortfolioCompositionView, get_active_portfolio

urlpatterns = [
    path('portfolios/', PortfolioListCreateView.as_view(), name='portfolio-list-create'),
    path('portfolios/<int:pk>', PortfolioDetailView.as_view(), name='portfolio-detail'),

    # Pobieranie portfolio id
    path('portfolio/active/', get_active_portfolio, name='portfolio-active'),

    path('portfolios/<int:portfolio_id>/stocks/', PortfolioStockListCreateView.as_view(), name='portfolio-stock-list-create'),
    # Edycja portfolio stock
    path('portfolios/<int:portfolio_id>/stocks/<int:position_id>/', PortfolioStockDetailView.as_view(), name='portfolio-stock-detail'),

    path('portfolio/<int:portfolio_id>/stats/', PortfolioStatsView.as_view(), name='portfolio-stats'),
    path('portfolio/<int:portfolio_id>/chart/', PortfolioChartView.as_view(), name='portfolio-chart'),
    path('portfolio/<int:portfolio_id>/composition/', PortfolioCompositionView.as_view(), name='portfolio-composition')
]
