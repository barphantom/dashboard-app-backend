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


class GARecommendationRequestSerializer(serializers.Serializer):
    # Nowy serializer dla endpointu GA (użytkownik podaje proste preferencje)
    portfolio_size = serializers.IntegerField(min_value=5, max_value=25, default=10)
    risk = serializers.IntegerField(min_value=1, max_value=5, default=3,
                                    help_text="Profil ryzyka 1 (konserwatywny) - 5 (agresywny)")
    investment_goal = serializers.ChoiceField(choices=["growth", "income", "balanced"], default="balanced",
                                              help_text="Wzrost / Dochód / Zbalansowany")
    time_horizon = serializers.ChoiceField(choices=["short", "medium", "long"], default="medium",
                                           help_text="Horyzont inwestycyjny")
    concentration = serializers.IntegerField(min_value=1, max_value=5, default=3,
                                             help_text="Poziom koncentracji 1-5")
    sector_allocations = serializers.DictField(
        child=serializers.FloatField(min_value=0.0), required=False,
        help_text="Opcjonalnie: udział sektorów (sumuje się do 1.0)"
    )
    exclude_sectors = serializers.ListField(child=serializers.CharField(), required=False)
    exclude_tickers = serializers.ListField(child=serializers.CharField(), required=False)

    # hiperparametry GA (opcjonalne)
    population_size = serializers.IntegerField(min_value=10, max_value=2000, default=200)
    generations = serializers.IntegerField(min_value=1, max_value=2000, default=150)
    mutation_rate = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.08)

    def validate(self, data):
        # jeśli podano sektor_allocations, sprawdź sumę
        sector_allocs = data.get("sector_allocations")
        if sector_allocs:
            total = sum(sector_allocs.values())
            if not (abs(total - 1.0) < 1e-8):
                raise serializers.ValidationError(f"Suma udziałów sektorów powinna wynosić 1.0, a jest {total}")
        return data
