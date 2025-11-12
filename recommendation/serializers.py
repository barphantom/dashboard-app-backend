from rest_framework import serializers

class RecommendationRequestSerializer(serializers.Serializer):
    sector_allocations = serializers.DictField(
        child=serializers.FloatField(min_value=0.0),
        help_text="Udziały sektorów, np. {'Technology': 50, 'Healthcare': 30, 'Financial Services': 20}"
    )
    global_params = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Parametry globalne, np. {'beta': 1.0, 'pe_ratio': 20, 'market_cap': 2e11, 'dividend_yield': 1.5}"
    )

    def validate(self, data):
        sectors = data.get("sector_allocations", {})
        total = sum(sectors.values())

        if total != 1:
            raise serializers.ValidationError(f"Suma udziałów sektorów powinna wynosić 100%, a jest {total}%")

        params = data.get("global_params", {})
        for key in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
            if key not in params:
                raise serializers.ValidationError(f"Brak parametru {key}")
            if params[key] < 0:
                raise serializers.ValidationError(f"Wartość {key} musi być dodatnia")

        return data
