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


# class GARecommendationRequestSerializer(serializers.Serializer):
#     # Nowy serializer dla endpointu GA (użytkownik podaje proste preferencje)
#     portfolio_size = serializers.IntegerField(min_value=5, max_value=25, default=10)
#     risk = serializers.IntegerField(min_value=1, max_value=5, default=3,
#                                     help_text="Profil ryzyka 1 (konserwatywny) - 5 (agresywny)")
#     investment_goal = serializers.ChoiceField(choices=["growth", "income", "balanced"], default="balanced",
#                                               help_text="Wzrost / Dochód / Zbalansowany")
#     time_horizon = serializers.ChoiceField(choices=["short", "medium", "long"], default="medium",
#                                            help_text="Horyzont inwestycyjny")
#     concentration = serializers.IntegerField(min_value=1, max_value=5, default=3,
#                                              help_text="Poziom koncentracji 1-5")
#     sector_allocations = serializers.DictField(
#         child=serializers.FloatField(min_value=0.0), required=False,
#         help_text="Opcjonalnie: udział sektorów (sumuje się do 1.0)"
#     )
#     exclude_sectors = serializers.ListField(child=serializers.CharField(), required=False)
#     exclude_tickers = serializers.ListField(child=serializers.CharField(), required=False)
#
#     # hiperparametry GA (opcjonalne)
#     population_size = serializers.IntegerField(min_value=10, max_value=2000, default=200)
#     generations = serializers.IntegerField(min_value=1, max_value=2000, default=150)
#     mutation_rate = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.08)
#
#     def validate(self, data):
#         # jeśli podano sektor_allocations, sprawdź sumę
#         sector_allocs = data.get("sector_allocations")
#         if sector_allocs:
#             total = sum(sector_allocs.values())
#             if not (abs(total - 1.0) < 1e-8):
#                 raise serializers.ValidationError(f"Suma udziałów sektorów powinna wynosić 1.0, a jest {total}")
#         return data



class PortfolioSegmentSerializer(serializers.Serializer):
    """
    Definiuje parametry dla pojedynczej części portfela (np. 50% Agresywne).
    """
    ratio = serializers.FloatField(min_value=0.01, max_value=1.0, required=True,
                                   help_text="Jaka część portfela (0.0 - 1.0) ma mieć te parametry")
    risk = serializers.IntegerField(min_value=1, max_value=5, default=3)
    investment_goal = serializers.ChoiceField(choices=["growth", "income", "balanced"], default="balanced")
    time_horizon = serializers.ChoiceField(choices=["short", "medium", "long"], default="medium")


class GARecommendationRequestSerializer(serializers.Serializer):
    # --- Ustawienia Globalne ---
    portfolio_size = serializers.IntegerField(min_value=5, max_value=50, default=10)

    # Koncentracja (1-5) steruje karą za brak dywersyfikacji w CAŁYM portfelu
    concentration = serializers.IntegerField(min_value=1, max_value=5, default=3,
                                             help_text="1=Wysoka dywersyfikacja, 5=Skupienie")

    # Filtrowanie twarde
    exclude_sectors = serializers.ListField(child=serializers.CharField(), required=False, default=[])
    exclude_tickers = serializers.ListField(child=serializers.CharField(), required=False, default=[])

    # --- Ustawienia Segmentów (Nowość) ---
    segments = PortfolioSegmentSerializer(many=True, required=False,
                                          help_text="Lista segmentów portfela. Suma ratio musi wynosić 1.0")

    # --- Pola Legacy (Opcjonalne - dla wstecznej kompatybilności) ---
    # Jeśli użytkownik nie poda 'segments', użyjemy tych pól, aby stworzyć jeden segment 100%
    risk = serializers.IntegerField(min_value=1, max_value=5, required=False, default=3)
    investment_goal = serializers.ChoiceField(choices=["growth", "income", "balanced"], required=False,
                                              default="balanced")
    time_horizon = serializers.ChoiceField(choices=["short", "medium", "long"], required=False, default="medium")

    # --- Hiperparametry GA ---
    population_size = serializers.IntegerField(min_value=10, max_value=2000, default=200)
    generations = serializers.IntegerField(min_value=1, max_value=2000, default=120)
    mutation_rate = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.08)

    def validate(self, data):
        """
        Walidacja i normalizacja danych.
        Jeśli brak 'segments', tworzy segment na podstawie pól legacy.
        Sprawdza sumę ratio.
        """
        segments = data.get("segments")

        # 1. Fallback: Jeśli nie podano segmentów, zbuduj jeden segment 100% ze starych pól
        if not segments:
            # Tworzymy sztuczny segment 100%
            data["segments"] = [{
                "ratio": 1.0,
                "risk": data.get("risk", 3),
                "investment_goal": data.get("investment_goal", "balanced"),
                "time_horizon": data.get("time_horizon", "medium")
            }]

        # 2. Walidacja sumy ratio (jeśli segmenty są podane lub właśnie utworzone)
        current_segments = data["segments"]
        total_ratio = sum([seg["ratio"] for seg in current_segments])

        # Tolerancja dla float (np. 0.33 + 0.33 + 0.33 = 0.99)
        if not (0.99 <= total_ratio <= 1.01):
            raise serializers.ValidationError(
                f"Suma udziałów segmentów (ratio) musi wynosić 1.0. Obecnie wynosi: {total_ratio:.2f}"
            )

        return data


class SegmentSerializer(serializers.Serializer):
    ratio = serializers.FloatField(min_value=0.0, max_value=1.0)
    risk = serializers.IntegerField(min_value=1, max_value=5)
    investment_goal = serializers.ChoiceField(choices=["growth", "income", "balanced"])
    time_horizon = serializers.ChoiceField(choices=["short", "medium", "long"])


class GAPortfolioRequestSerializer(serializers.Serializer):
    portfolio_size = serializers.IntegerField(min_value=1, max_value=100, default=10)
    concentration = serializers.IntegerField(min_value=1, max_value=5, default=1)
    segments = SegmentSerializer(many=True)

    def validate_segments(self, value):
        if not value:
            raise serializers.ValidationError("Lista segmentów nie może być pusta.")

        total_ratio = sum(item['ratio'] for item in value)

        if not (0.95 <= total_ratio <= 1.05):
            raise serializers.ValidationError(f"Suma ratio segmentów musi wynosić 1.0 (obecnie: {total_ratio})")
        return value
