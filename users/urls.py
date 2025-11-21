from django.urls import path
from .views import RegisterView, LoginView, UserProfileView
from rest_framework_simplejwt.views import (
    TokenVerifyView,
    TokenRefreshView
)

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', UserProfileView.as_view(), name='user_profile'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
]
