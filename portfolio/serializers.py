from datetime import date
from decimal import Decimal

from rest_framework import serializers
from portfolio.models import Portfolio, PortfolioStock


class PortfolioStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioStock
        fields = ['id', 'symbol', 'name', 'quantity', 'purchase_price', 'purchase_date', 'created_at', 'updated_at']

    def validate_symbol(self, value):
        if not value:
            raise serializers.ValidationError("Symbol cannot be empty")

        value = value.upper().strip()

        if len(value) > 10:
            raise serializers.ValidationError("Symbol too long (max 10 characters)")

        if not value.replace('.', '').isalnum():
            raise serializers.ValidationError("Symbol can only contain letters, numbers, and dots")

        return value

    def validate_quantity(self, value):
        if value <= 0:
            raise serializers.ValidationError("Quantity must be greater than 0")

        if value > Decimal('999999999.99'):
            raise serializers.ValidationError("Quantity too large")

        return value

    def validate_purchase_price(self, value):
        if value <= 0:
            raise serializers.ValidationError("Purchase price must be greater than 0")

        if value > Decimal('999999999.99'):
            raise serializers.ValidationError("Purchase price too large")

        return value

    def validate_purchase_date(self, value):
        if value > date.today():
            raise serializers.ValidationError("Purchase date cannot be in the future")

        if value.year < 1970:
            raise serializers.ValidationError("Purchase date too old")

        return value


class PortfolioSerializer(serializers.ModelSerializer):
    stocks = PortfolioStockSerializer(many=True, read_only=True)

    class Meta:
        model = Portfolio
        fields = ['id', 'name', 'description', 'created_at', 'stocks']
        read_only_fields = ['created_at']

