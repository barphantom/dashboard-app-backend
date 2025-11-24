from django.urls import path

from marketdata.views import StockSearchView, OHLCVView

urlpatterns = [
    path('search/', StockSearchView.as_view(), name='stock-search'),
    path('ohlcv/', OHLCVView.as_view(), name='ohlcv'),
]
