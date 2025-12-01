from django.contrib import admin
from .models import Portfolio, PortfolioStock

# Register your models here.
admin.site.register(Portfolio)
admin.site.register(PortfolioStock)