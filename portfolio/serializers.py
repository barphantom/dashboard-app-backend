from rest_framework import serializers
# from portfolio.models import Contact
from portfolio.models import Portfolio, PortfolioStock

# class PortfolioHelloSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Contact
#         fields = ['id', 'name', 'email', 'age']
#         read_only_fields = ['id']
#
#     def validate_name(self, value):
#         if len(value.strip()) < 2:
#             raise serializers.ValidationError('Name must be at least 2 characters')
#         return value.strip()
#
#     def validate(self, data):
#         name = data['name']
#         email = data.get('email')
#         age = data['age']
#
#         errors = {}
#
#         if age and age < 18 and not email:
#             errors["email"] = "Email opiekuna jest wymagany dla osób poniżej 18 lat"
#
#         if name and any(char.isdigit() for char in name):
#             errors["name"] = "Imię nie może zawierać cyfr"
#
#         if errors:
#             raise serializers.ValidationError(errors)
#
#         return data

class PortfolioStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioStock
        fields = ['id', 'symbol', 'name', 'quantity', 'purchase_price', 'purchase_date', 'created_at', 'updated_at']


class PortfolioSerializer(serializers.ModelSerializer):
    stocks = PortfolioStockSerializer(many=True, read_only=True)

    class Meta:
        model = Portfolio
        fields = ['id', 'name', 'description', 'created_at', 'stocks']
        read_only_fields = ['created_at']

