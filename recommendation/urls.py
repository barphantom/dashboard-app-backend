from django.urls import path
from .views import recommendation_view, ga_recommendation_view, GARecommendationView2

urlpatterns = [
    path('recommendations/', recommendation_view, name='recommendations'),
    path('recommendations/ga/', ga_recommendation_view, name='recommendations-ga'),
    path('recommendations/ga2/', GARecommendationView2.as_view(), name='recommendations-ga2'),
]
