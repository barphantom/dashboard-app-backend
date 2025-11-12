from django.urls import path
from .views import recommendation_view

urlpatterns = [
    path('recommendations/', recommendation_view, name='recommendations'),
]
