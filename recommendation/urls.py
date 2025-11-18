from django.urls import path
from .views import recommendation_view, ga_recommendation_view

urlpatterns = [
    path('recommendations/', recommendation_view, name='recommendations'),
    path('recommendations/ga/', ga_recommendation_view, name='recommendations-ga'),
]
