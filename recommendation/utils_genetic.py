# import random
# import math
# from collections import Counter, defaultdict
# from copy import deepcopy
#
# import pandas as pd
# import numpy as np
#
# # =============================================
# # Prosty, czytelny moduł implementujący GA
# # =============================================
# # Założenia:
# # - Dane spółek w CSV w recommendation/sp500_companies_final.csv
# # - Kolumny csv: symbol,shortName,sector,beta,pe_ratio,market_cap,dividend_yield
# # - Ten moduł NIE używa cosine-similarity; używa cech fundamentalnych wyłącznie.
# # =============================================
#
# CSV_PATH = "recommendation/sp500_companies_final.csv"
#
# # ---- Wczytanie danych i przygotowanie statystyk ----
# _df = pd.read_csv(CSV_PATH)
# _df.columns = [c.strip() for c in _df.columns]
#
# # oczyść nazwy i wartości
# _df["shortName"] = _df["shortName"].astype(str).str.replace('"', '', regex=False)
# _df["sector"] = _df["sector"].astype(str).str.replace('"', '', regex=False)
#
# for col in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
#     _df[col] = pd.to_numeric(_df[col], errors="coerce")
#
# # uzupełnij brakujące medianą (proste i bezpieczne)
# for col in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
#     if _df[col].isna().any():
#         _df[col].fillna(_df[col].median(), inplace=True)
#
# # Oblicz statystyki pomocnicze do normalizacji
# _STATS = {
#     "min_beta": float(_df["beta"].min()),
#     "max_beta": float(_df["beta"].max()),
#     "max_pe": float(max(1.0, _df["pe_ratio"].max())),  # zabezpieczenie, aby nie dzielić przez 0
#     "max_dividend": float(max(0.0001, _df["dividend_yield"].max())),
#     "min_cap": float(max(1.0, _df["market_cap"].min())),
#     "max_cap": float(max(1.0, _df["market_cap"].max())),
#     "median_cap": float(_df["market_cap"].median()),
#     "p10_cap": float(_df["market_cap"].quantile(0.1)),
#     "p90_cap": float(_df["market_cap"].quantile(0.9)),
# }
#
# # Słownik dostępnych spółek
# _STOCK_POOL = _df["symbol"].tolist()
# _STOCK_BY_SYMBOL = {row["symbol"]: row for _, row in _df.iterrows()}
#
#
# # -----------------------
# # Mapping user preferences -> targets & weights
# # -----------------------
# def map_user_preferences(user_input):
#     """
#     user_input: dict (pochodzący z serializer)
#     Zwraca mapping zawierający:
#       - target_beta
#       - target_size (market cap)
#       - w_pe, w_div (dla celu inwestycyjnego)
#       - concentration rules
#       - weights: w_risk, w_goal, w_size, w_sector
#     """
#     # target_beta
#     target_beta_map = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
#     target_beta = target_beta_map.get(user_input["risk"], 1.0)
#
#     # investment goal -> weights for P/E vs dividend when computing goal_score
#     goal = user_input.get("investment_goal", "balanced")
#     if goal == "growth":
#         w_pe, w_div = 0.7, 0.3
#     elif goal == "income":
#         w_pe, w_div = 0.3, 0.7
#     else:
#         w_pe, w_div = 0.5, 0.5
#
#     # time_horizon -> target_size (use percentiles)
#     horizon = user_input.get("time_horizon", "medium")
#     if horizon == "short":
#         target_size = _STATS["p10_cap"]
#     elif horizon == "long":
#         target_size = _STATS["p90_cap"]
#     else:
#         target_size = _STATS["median_cap"]
#
#     # concentration -> rules
#     concentration = int(user_input.get("concentration", 3))
#     if concentration <= 1:
#         conc_rules = {"max_sector_share": 0.15, "min_sectors": 3}
#     elif concentration == 3:
#         conc_rules = {"max_sector_share": 0.30, "min_sectors": 3}
#     else:  # 4..5
#         conc_rules = {"max_sector_share": 0.50, "min_sectors": 1}
#
#     # derive weights for fitness from user preferences:
#     # base weights
#     w_risk = 0.25
#     w_goal = 0.25
#     w_size = 0.25
#     w_sector = 0.25
#
#     # adjust by explicit preferences
#     # risk: stronger weight if user risk far from neutral (3)
#     w_risk += (user_input["risk"] - 3) * 0.05  # small adjustment
#
#     # investment goal affects w_goal (income/growth emphasize goal)
#     if goal == "income":
#         w_goal += 0.10
#     elif goal == "growth":
#         w_goal += 0.05
#
#     # horizon: longer horizon increases importance of size (stability)
#     if horizon == "long":
#         w_size += 0.05
#     elif horizon == "short":
#         w_size -= 0.03
#
#     # concentration: if user wants high diversification, increase weight on sector
#     if concentration <= 2:
#         w_sector += 0.08
#     elif concentration >= 4:
#         w_sector -= 0.05
#
#     # normalize weights to sum 1
#     total = max(1e-9, (w_risk + w_goal + w_size + w_sector))
#     weights = {
#         "w_risk": w_risk / total,
#         "w_goal": w_goal / total,
#         "w_size": w_size / total,
#         "w_sector": w_sector / total,
#         "w_pe": w_pe,
#         "w_div": w_div
#     }
#
#     mapping = {
#         "target_beta": float(target_beta),
#         "target_size": float(target_size),
#         "concentration_rules": conc_rules,
#         "weights": weights,
#     }
#     return mapping
#
#
# # -----------------------
# # Scoring pojedynczej spółki
# # -----------------------
# def score_stock(symbol, mapping):
#     """
#     Zwraca słownik score'ów (wszystko w [0,1]) dla danej spółki.
#     """
#     row = _STOCK_BY_SYMBOL[symbol]
#     # risk_score (beta)
#     max_beta_diff = max(1e-6, _STATS["max_beta"] - _STATS["min_beta"])
#     risk_score = 1.0 - abs(float(row["beta"]) - mapping["target_beta"]) / max_beta_diff
#     risk_score = float(max(0.0, min(1.0, risk_score)))
#
#     # valuation_score (P/E) - mniejsze P/E lepsze
#     valuation_score = 1.0 - (float(row["pe_ratio"]) / _STATS["max_pe"])
#     valuation_score = float(max(0.0, min(1.0, valuation_score)))
#
#     # dividend_score - wyższa dywidenda lepsza
#     dividend_score = float(row["dividend_yield"]) / _STATS["max_dividend"]
#     dividend_score = float(max(0.0, min(1.0, dividend_score)))
#
#     # goal_score - kombinacja pe & dividend
#     w_pe = mapping["weights"]["w_pe"]
#     w_div = mapping["weights"]["w_div"]
#     goal_score = w_pe * valuation_score + w_div * dividend_score
#
#     # size_score - log-normalized market cap
#     # zabezpieczenie przed log(0) - użyj min 1
#     cap = max(1.0, float(row["market_cap"]))
#     min_cap = max(1.0, _STATS["min_cap"])
#     max_cap = max(1.0, _STATS["max_cap"])
#     # odległość logarytmiczna
#     denom = math.log(max_cap / min_cap) if max_cap > min_cap else 1.0
#     size_score = 1.0 - abs(math.log(cap) - math.log(mapping["target_size"])) / (denom if denom > 0 else 1.0)
#     size_score = float(max(0.0, min(1.0, size_score)))
#
#     return {
#         "symbol": symbol,
#         "risk_score": risk_score,
#         "valuation_score": valuation_score,
#         "dividend_score": dividend_score,
#         "goal_score": goal_score,
#         "size_score": size_score,
#         "sector": row["sector"],
#         "name": row.get("shortName", symbol),
#     }
#
#
# # -----------------------
# # Population initialization
# # -----------------------
# def initialize_population(stock_pool, stock_scores, portfolio_size, population_size, bias=True):
#     """
#     Tworzy populację list tickerów. Jeśli bias=True, preferuje lepsze spółki
#     przy inicjalizacji (szybsza konwergencja).
#     """
#     population = []
#     # przygotuj ranking po goal_score + risk_score (prosty agregat) do biasowania
#     aggregated = []
#     for s in stock_pool:
#         sc = stock_scores[s]
#         aggregated_score = 0.6 * sc["goal_score"] + 0.4 * sc["risk_score"]
#         aggregated.append((s, aggregated_score))
#     aggregated.sort(key=lambda x: x[1], reverse=True)
#     top_list = [s for s, _ in aggregated[:max(50, len(aggregated)//4)]]
#
#     for _ in range(population_size):
#         if bias:
#             # mieszamy: część z top_list, część losowa
#             pick = []
#             # weź 60% z top_list (jeśli dostępne)
#             n_top = int(portfolio_size * 0.6)
#             n_random = portfolio_size - n_top
#             pick.extend(random.sample(top_list, min(n_top, len(top_list))))
#             # uzupełnij losowymi (bez powtórzeń)
#             remaining = list(set(stock_pool) - set(pick))
#             if len(remaining) >= n_random:
#                 pick.extend(random.sample(remaining, n_random))
#             else:
#                 pick.extend(random.sample(stock_pool, portfolio_size - len(pick)))
#             random.shuffle(pick)
#             individual = pick[:portfolio_size]
#         else:
#             individual = random.sample(stock_pool, portfolio_size)
#         population.append(individual)
#     return population
#
#
# # -----------------------
# # Fitness evaluation for a portfolio (individual)
# # -----------------------
# def evaluate_portfolio(individual, stock_scores, mapping):
#     """
#     individual: list of tickers
#     stock_scores: dict symbol -> score dict
#     mapping: mapping from map_user_preferences
#     Zwraca (fitness, meta) gdzie meta zawiera rozbicie składowych
#     """
#     n = len(individual)
#     if n == 0:
#         return 0.0, {}
#
#     # mean scores
#     risk_mean = np.mean([stock_scores[s]["risk_score"] for s in individual])
#     goal_mean = np.mean([stock_scores[s]["goal_score"] for s in individual])
#     size_mean = np.mean([stock_scores[s]["size_score"] for s in individual])
#
#     # sector score: policz udziały liczebne (ilość spółek z sektora / n)
#     sector_counts = Counter([stock_scores[s]["sector"] for s in individual])
#     sector_shares = {s: c / n for s, c in sector_counts.items()}
#     max_share = max(sector_shares.values()) if sector_shares else 0.0
#
#     allowed_max = mapping["concentration_rules"]["max_sector_share"]
#     # penalty = nadmiar nad allowed_max (znormalizowany do [0,1])
#     penalty = max(0.0, (max_share - allowed_max) / 1.0)  # surowa normalizacja; allowed_max w (0..1)
#     # jeżeli jest za mało sektorów (min_sectors), kara również
#     min_sectors = mapping["concentration_rules"]["min_sectors"]
#     unique_sectors = len(sector_counts)
#     if unique_sectors < min_sectors:
#         # dopal dodatkową karę proporcjonalną do brakujących sektorów
#         penalty += (min_sectors - unique_sectors) / max(1.0, min_sectors)
#
#     sector_score = float(max(0.0, min(1.0, 1.0 - penalty)))
#
#     # agregacja z wagami
#     w = mapping["weights"]
#     fitness = (
#         w["w_risk"] * risk_mean
#         + w["w_goal"] * goal_mean
#         + w["w_size"] * size_mean
#         + w["w_sector"] * sector_score
#     )
#
#     meta = {
#         "risk_mean": float(risk_mean),
#         "goal_mean": float(goal_mean),
#         "size_mean": float(size_mean),
#         "sector_score": float(sector_score),
#         "unique_sectors": unique_sectors,
#         "max_sector_share": float(max_share)
#     }
#     return float(fitness), meta
#
#
# # -----------------------
# # Genetic operators: selection, crossover, mutation, repair
# # -----------------------
# def tournament_selection(population, fitnesses, k=3):
#     """Zwraca kopię wybranego osobnika (tournament selection)."""
#     candidates = random.sample(range(len(population)), min(k, len(population)))
#     best = max(candidates, key=lambda i: fitnesses[i])
#     return deepcopy(population[best])
#
#
# def crossover(parent_a, parent_b):
#     """
#     One-point crossover na listach tickerów, z repair (bez duplikatów).
#     """
#     size = len(parent_a)
#     if size < 2:
#         return deepcopy(parent_a), deepcopy(parent_b)
#
#     point = random.randint(1, size - 1)
#     child1 = parent_a[:point] + [g for g in parent_b if g not in parent_a[:point]]
#     child2 = parent_b[:point] + [g for g in parent_a if g not in parent_b[:point]]
#
#     # repair do właściwego rozmiaru
#     child1 = repair_individual(child1, _STOCK_POOL, size)
#     child2 = repair_individual(child2, _STOCK_POOL, size)
#     return child1, child2
#
#
# def mutate(individual, mutation_rate):
#     """
#     Prosta mutacja: z pewnym prawdopodobieństwem zamień gen (ticker) na inny losowy.
#     """
#     for i in range(len(individual)):
#         if random.random() < mutation_rate:
#             candidates = list(set(_STOCK_POOL) - set(individual))
#             if candidates:
#                 individual[i] = random.choice(candidates)
#     return individual
#
#
# def repair_individual(individual, stock_universe, size):
#     """
#     Usuń duplikaty i uzupełnij losowymi tickerami, aby uzyskać długość size.
#     """
#     unique = []
#     for x in individual:
#         if x not in unique:
#             unique.append(x)
#     while len(unique) < size:
#         candidate = random.choice(stock_universe)
#         if candidate not in unique:
#             unique.append(candidate)
#     return unique[:size]
#
#
# def select_top_k(population, fitnesses, k):
#     """Zwraca listę kopii top-k osobników."""
#     idx = np.argsort(fitnesses)[-k:][::-1]  # top k indices
#     return [deepcopy(population[i]) for i in idx]
#
#
# # -----------------------
# # Główna pętla GA
# # -----------------------
# def run_ga(stock_pool, stock_scores, mapping, portfolio_size,
#            population_size=200, generations=150, mutation_rate=0.08, elite_size=5):
#     """
#     Uruchamia GA i zwraca najlepszy portfel oraz historię konwergencji.
#     """
#     population = initialize_population(stock_pool, stock_scores, portfolio_size, population_size, bias=True)
#     best_history = []
#
#     for gen in range(generations):
#         fitnesses = []
#         metas = []
#         for ind in population:
#             f, meta = evaluate_portfolio(ind, stock_scores, mapping)
#             fitnesses.append(f)
#             metas.append(meta)
#
#         fitnesses = list(map(float, fitnesses))
#         best_idx = int(np.argmax(fitnesses))
#         best_history.append(float(fitnesses[best_idx]))
#
#         # create next generation
#         next_pop = []
#         # elitism - zachowaj top elite_size
#         elites = select_top_k(population, fitnesses, elite_size)
#         next_pop.extend(elites)
#
#         # generate children until reach population_size
#         while len(next_pop) < population_size:
#             parent_a = tournament_selection(population, fitnesses, k=3)
#             parent_b = tournament_selection(population, fitnesses, k=3)
#             child1, child2 = crossover(parent_a, parent_b)
#             child1 = mutate(child1, mutation_rate)
#             child2 = mutate(child2, mutation_rate)
#             next_pop.append(child1)
#             if len(next_pop) < population_size:
#                 next_pop.append(child2)
#
#         population = next_pop
#
#     # final evaluation
#     final_fitnesses = []
#     final_metas = []
#     for ind in population:
#         f, meta = evaluate_portfolio(ind, stock_scores, mapping)
#         final_fitnesses.append(f)
#         final_metas.append(meta)
#
#     best_idx = int(np.argmax(final_fitnesses))
#     best_individual = population[best_idx]
#     best_score = float(final_fitnesses[best_idx])
#     best_meta = final_metas[best_idx]
#
#     return {
#         "portfolio": best_individual,
#         "fitness": best_score,
#         "meta": best_meta,
#         "convergence": best_history
#     }
#
#
# # -----------------------
# # Główny wrapper: get_ga_recommendations(params)
# # -----------------------
# def get_ga_recommendations(params: dict):
#     """
#     params: dict z serializer GARecommendationRequestSerializer
#     Zwraca strukturalny dict gotowy do serializacji JSON.
#     """
#     # 1) Mapowanie preferencji użytkownika
#     mapping = map_user_preferences(params)
#
#     # 2) Odfiltrowanie puli spółek (exclude lists + ewentualne ograniczenia)
#     pool = list(_STOCK_POOL)
#     # exclude tickers/sectors
#     exclude_tickers = [t.upper() for t in params.get("exclude_tickers", [])]
#     exclude_sectors = [s for s in params.get("exclude_sectors", [])]
#     if exclude_tickers:
#         pool = [s for s in pool if s.upper() not in exclude_tickers]
#     if exclude_sectors:
#         pool = [s for s in pool if _STOCK_BY_SYMBOL[s]["sector"] not in exclude_sectors]
#
#     # 3) Compute stock_scores dict
#     stock_scores = {}
#     for s in pool:
#         stock_scores[s] = score_stock(s, mapping)
#
#     # optionally limit pool to top-K by combined score to speed up GA
#     # tutaj bierzemy top 300 lub full pool jeśli mniejszy
#     combined = []
#     for s, sc in stock_scores.items():
#         combined_score = 0.6 * sc["goal_score"] + 0.4 * sc["risk_score"]
#         combined.append((s, combined_score))
#     combined.sort(key=lambda x: x[1], reverse=True)
#     top_k = min(len(combined), 300)
#     top_pool = [s for s, _ in combined[:top_k]]
#
#     # 4) Run GA
#     population_size = int(params.get("population_size", 200))
#     generations = int(params.get("generations", 150))
#     mutation_rate = float(params.get("mutation_rate", 0.08))
#     portfolio_size = int(params.get("portfolio_size", 10))
#
#     ga_res = run_ga(
#         stock_pool=top_pool,
#         stock_scores=stock_scores,
#         mapping=mapping,
#         portfolio_size=portfolio_size,
#         population_size=population_size,
#         generations=generations,
#         mutation_rate=mutation_rate,
#         elite_size=max(1, int(population_size * 0.02))
#     )
#
#     # 5) Przygotuj czytelny wynik: rozbij portfolio na obiekty z metadanymi
#     result_portfolio = []
#     for sym in ga_res["portfolio"]:
#         sc = stock_scores[sym]
#         row = _STOCK_BY_SYMBOL[sym]
#         result_portfolio.append({
#             "symbol": sym,
#             "name": row.get("shortName", sym),
#             "sector": row.get("sector"),
#             "scores": {
#                 "risk_score": sc["risk_score"],
#                 "goal_score": sc["goal_score"],
#                 "size_score": sc["size_score"]
#             }
#         })
#
#     # Proponowane wagi - prosta strategia: równe udziały
#     suggested_weight = round(1.0 / max(1, len(result_portfolio)), 4)
#     for p in result_portfolio:
#         p["suggested_weight"] = suggested_weight
#
#     response = {
#         "portfolio": result_portfolio,
#         "fitness_score": ga_res["fitness"],
#         "meta": ga_res["meta"],
#         "convergence": ga_res["convergence"],
#         "parameters_used": {
#             "mapping": mapping,
#             "population_size": population_size,
#             "generations": generations,
#             "mutation_rate": mutation_rate
#         }
#     }
#     return response

import random
import math
from collections import Counter, defaultdict
from copy import deepcopy
import pandas as pd
import numpy as np

# =============================================
# KONFIGURACJA I DANE (Bez zmian)
# =============================================
CSV_PATH = "recommendation/sp500_companies_final.csv"

_df = pd.read_csv(CSV_PATH)
_df.columns = [c.strip() for c in _df.columns]
_df["shortName"] = _df["shortName"].astype(str).str.replace('"', '', regex=False)
_df["sector"] = _df["sector"].astype(str).str.replace('"', '', regex=False)

for col in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
    _df[col] = pd.to_numeric(_df[col], errors="coerce")

for col in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
    if _df[col].isna().any():
        _df[col].fillna(_df[col].median(), inplace=True)

_STATS = {
    "min_beta": float(_df["beta"].min()),
    "max_beta": float(_df["beta"].max()),
    "max_pe": float(max(1.0, _df["pe_ratio"].max())),
    "max_dividend": float(max(0.0001, _df["dividend_yield"].max())),
    "min_cap": float(max(1.0, _df["market_cap"].min())),
    "max_cap": float(max(1.0, _df["market_cap"].max())),
    "median_cap": float(_df["market_cap"].median()),
    "p10_cap": float(_df["market_cap"].quantile(0.1)),
    "p90_cap": float(_df["market_cap"].quantile(0.9)),
}

_STOCK_POOL = _df["symbol"].tolist()
_STOCK_BY_SYMBOL = {row["symbol"]: row for _, row in _df.iterrows()}


# =============================================
# [NOWOŚĆ] 1. Logika podziału portfela (Integer Partitioning)
# =============================================
def get_partition_sizes(total_size, ratios):
    """
    Dzieli liczbę całkowitą (total_size) na części wg podanych proporcji (ratios),
    tak aby suma części była idealnie równa total_size.
    Rozwiązuje problem zaokrągleń (Largest Remainder Method).
    """
    # 1. Surowe wielkości
    raw_sizes = [total_size * r for r in ratios]

    # 2. Części całkowite (podłoga)
    int_sizes = [int(math.floor(s)) for s in raw_sizes]

    # 3. Reszta do rozdysponowania
    remainder = total_size - sum(int_sizes)

    # 4. Sprawdź, kto ma największą część ułamkową
    diffs = [(r - i, idx) for idx, (r, i) in enumerate(zip(raw_sizes, int_sizes))]
    # Sortuj malejąco po części ułamkowej
    diffs.sort(key=lambda x: x[0], reverse=True)

    # 5. Rozdaj resztę
    for i in range(remainder):
        idx_to_increment = diffs[i][1]
        int_sizes[idx_to_increment] += 1

    return int_sizes


# =============================================
# [ZMIANA] 2. Mapowanie preferencji per SEGMENT
# =============================================
def map_single_segment_preferences(segment_input):
    """
    Mapuje parametry pojedynczego segmentu (np. 'agresywna część portfela').
    """
    # target_beta
    target_beta_map = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
    # Domyślnie 3 (neutral) jeśli brak
    risk_val = segment_input.get("risk", 3)
    target_beta = target_beta_map.get(risk_val, 1.0)

    # investment goal
    goal = segment_input.get("investment_goal", "balanced")
    if goal == "growth":
        w_pe, w_div = 0.8, 0.2  # Growth bardziej agresywnie w P/E (lub w momentum, tu uproszczone)
    elif goal == "income":
        w_pe, w_div = 0.2, 0.8
    else:
        w_pe, w_div = 0.5, 0.5

    # time_horizon -> target_size
    horizon = segment_input.get("time_horizon", "medium")
    if horizon == "short":
        target_size = _STATS["p10_cap"]  # Preferuje duże, stabilne
    elif horizon == "long":
        target_size = _STATS["median_cap"]  # Większa tolerancja
    else:
        target_size = _STATS["median_cap"]

    # Wagi dla segmentu (uproszczone względem oryginału, skupione na cechach)
    mapping = {
        "target_beta": float(target_beta),
        "target_size": float(target_size),
        "weights": {
            "w_risk": 0.35,  # Ryzyko jest kluczowe dla definicji segmentu
            "w_goal": 0.35,  # Cel (Div/Growth) też
            "w_size": 0.30,
            "w_pe": w_pe,
            "w_div": w_div
        }
    }
    return mapping


def map_global_constraints(user_input):
    """
    Osobna funkcja dla globalnych zasad (koncentracja sektorowa).
    """
    concentration = int(user_input.get("concentration", 3))
    if concentration <= 1:
        conc_rules = {"max_sector_share": 0.15, "min_sectors": 3}
    elif concentration == 3:
        conc_rules = {"max_sector_share": 0.30, "min_sectors": 3}
    else:  # 4..5
        conc_rules = {"max_sector_share": 0.50, "min_sectors": 1}
    return conc_rules


# =============================================
# [ZMIANA] 3. Scoring kontekstowy (Per Segment)
# =============================================
def calculate_score_for_segment(symbol, mapping):
    """
    Oblicza przydatność spółki dla konkretnego segmentu.
    Zwraca jedną liczbę (fitness) + szczegóły.
    """
    row = _STOCK_BY_SYMBOL[symbol]

    # 1. Risk Score (im bliżej target_beta tym lepiej)
    max_beta_diff = max(1e-6, _STATS["max_beta"] - _STATS["min_beta"])
    risk_dist = abs(float(row["beta"]) - mapping["target_beta"])
    risk_score = 1.0 - (risk_dist / max_beta_diff)
    risk_score = max(0.0, min(1.0, risk_score))

    # 2. Valuation & Dividend (Goal)
    val_score = 1.0 - (float(row["pe_ratio"]) / _STATS["max_pe"])
    val_score = max(0.0, min(1.0, val_score))

    div_score = float(row["dividend_yield"]) / _STATS["max_dividend"]
    div_score = max(0.0, min(1.0, div_score))

    goal_score = (mapping["weights"]["w_pe"] * val_score) + \
                 (mapping["weights"]["w_div"] * div_score)

    # 3. Size Score
    cap = max(1.0, float(row["market_cap"]))
    min_cap = max(1.0, _STATS["min_cap"])
    max_cap = max(1.0, _STATS["max_cap"])
    denom = math.log(max_cap / min_cap) if max_cap > min_cap else 1.0

    dist_size = abs(math.log(cap) - math.log(mapping["target_size"]))
    size_score = 1.0 - (dist_size / denom)
    size_score = max(0.0, min(1.0, size_score))

    # Weighted Sum
    w = mapping["weights"]
    final_score = (w["w_risk"] * risk_score) + \
                  (w["w_goal"] * goal_score) + \
                  (w["w_size"] * size_score)

    return float(final_score), {
        "risk_score": risk_score,
        "goal_score": goal_score,
        "size_score": size_score
    }


def precalculate_all_segment_scores(pool, segments_mappings):
    """
    Tworzy macierz wyników: scores[segment_index][symbol] = fitness_value.
    Optymalizacja, żeby nie liczyć tego w pętli GA.
    """
    matrix = []
    meta_matrix = []  # Do debugowania/wyników

    for seg_map in segments_mappings:
        seg_scores = {}
        seg_metas = {}
        for sym in pool:
            score, meta = calculate_score_for_segment(sym, seg_map)
            seg_scores[sym] = score
            seg_metas[sym] = meta
        matrix.append(seg_scores)
        meta_matrix.append(seg_metas)

    return matrix, meta_matrix


# =============================================
# [ZMIANA] 4. Fitness Function (Segmentowana)
# =============================================
def evaluate_segmented_portfolio(individual, score_matrix, partition_sizes, global_rules):
    """
    individual: lista symboli (len = total_portfolio_size)
    score_matrix: preobliczone wyniki [seg_idx][symbol]
    partition_sizes: lista intów, np. [5, 3, 2]
    global_rules: słownik z zasadami koncentracji sektorów
    """
    total_fitness = 0.0
    current_idx = 0
    total_slots = len(individual)

    # 1. Suma fitnessów cząstkowych z segmentów
    for i, size in enumerate(partition_sizes):
        if size == 0: continue

        segment_genes = individual[current_idx: current_idx + size]
        scores_for_seg = score_matrix[i]

        # Średnia ocena spółek w tym segmencie wg kryteriów tego segmentu
        segment_sum = sum(scores_for_seg[sym] for sym in segment_genes)
        segment_avg = segment_sum / size

        # Waga segmentu w ocenie całkowitej (proporcjonalna do jego wielkości)
        weight_in_portfolio = size / total_slots
        total_fitness += segment_avg * weight_in_portfolio

        current_idx += size

    # 2. Global Sector Penalty (Dywersyfikacja całego portfela)
    all_sectors = [_STOCK_BY_SYMBOL[s]["sector"] for s in individual]
    n = len(individual)
    sector_counts = Counter(all_sectors)
    max_share = max(sector_counts.values()) / n if n > 0 else 0
    unique_sectors = len(sector_counts)

    penalty = 0.0
    allowed_max = global_rules["max_sector_share"]

    # Kara za nadmierną koncentrację w jednym sektorze
    if max_share > allowed_max:
        penalty += (max_share - allowed_max) * 2.0  # Silna kara

    # Kara za zbyt mało sektorów
    if unique_sectors < global_rules["min_sectors"]:
        penalty += 0.1 * (global_rules["min_sectors"] - unique_sectors)

    final_fitness = max(0.0, total_fitness - penalty)

    return final_fitness, {
        "raw_fitness": total_fitness,
        "penalty": penalty,
        "max_sector_share": max_share,
        "unique_sectors": unique_sectors
    }


# =============================================
# [ZMIANA] 5. Inicjalizacja (Smart Seeding)
# =============================================
def initialize_segmented_population(pool, score_matrix, partition_sizes, pop_size):
    """
    Tworzy populację, gdzie geny w odpowiednich slotach są dobierane
    z puli najlepszych spółek dla danego segmentu.
    """
    population = []

    # Przygotuj "dobre pule" dla każdego segmentu (top 50%)
    top_pools = []
    for seg_idx in range(len(partition_sizes)):
        scores = score_matrix[seg_idx]
        # sortuj malejąco
        sorted_syms = sorted(pool, key=lambda s: scores[s], reverse=True)
        # Weź top 100 lub połowę puli
        cutoff = min(100, len(pool) // 2)
        top_pools.append(sorted_syms[:cutoff])

    for _ in range(pop_size):
        individual = []
        used_symbols = set()

        for seg_idx, size in enumerate(partition_sizes):
            if size == 0: continue

            segment_part = []
            pool_for_seg = top_pools[seg_idx]

            # Próbujemy dobrać unikalne z top puli
            candidates = [s for s in pool_for_seg if s not in used_symbols]

            # Jeśli brakuje kandydatów (rzadkie), dobierz z reszty świata
            if len(candidates) < size:
                others = [s for s in pool if s not in used_symbols]
                candidates.extend(others)

            chosen = random.sample(candidates, size)
            segment_part.extend(chosen)
            used_symbols.update(chosen)

            individual.extend(segment_part)

        population.append(individual)

    return population


# Operatory genetyczne (Turniej, Crossover, Mutacja, Repair)
# Pozostają podobne, ale Repair musi dbać o unikalność GLOBALNĄ
def tournament_selection(population, fitnesses, k=3):
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda i: fitnesses[i])
    return deepcopy(population[best])


def crossover(parent_a, parent_b, pool):
    size = len(parent_a)
    if size < 2: return deepcopy(parent_a), deepcopy(parent_b)
    point = random.randint(1, size - 1)
    # Standardowy one-point crossover
    # Ryzyko: może wymieszać segmenty, ale GA to "wygładzi" selekcją
    c1 = parent_a[:point] + [x for x in parent_b if x not in parent_a[:point]]
    c2 = parent_b[:point] + [x for x in parent_a if x not in parent_b[:point]]

    # Repair (uzupełnienie brakujących do pełnej długości)
    c1 = repair_individual(c1, size, pool)
    c2 = repair_individual(c2, size, pool)
    return c1, c2


def repair_individual(ind, size, pool):
    # Usuń duplikaty zachowując kolejność
    seen = set()
    unique = []
    for x in ind:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    # Jeśli za krótki, dolosuj z całej puli (proste repair)
    # Można by tu robić smart repair, ale losowe zwiększa różnorodność
    while len(unique) < size:
        cand = random.choice(pool)
        if cand not in seen:
            unique.append(cand)
            seen.add(cand)
    return unique[:size]


def mutate(individual, mutation_rate, pool):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Swap with random from global pool
            cand = random.choice(pool)
            if cand not in individual:
                individual[i] = cand
    return individual


# =============================================
# GŁÓWNA PĘTLA GA
# =============================================
def run_segmented_ga(pool, score_matrix, partition_sizes, global_rules,
                     pop_size=200, generations=100, mutation_rate=0.08, elite_size=5):
    population = initialize_segmented_population(pool, score_matrix, partition_sizes, pop_size)
    best_history = []

    # Główna pętla
    for gen in range(generations):
        fitnesses = []
        metas = []
        for ind in population:
            f, meta = evaluate_segmented_portfolio(ind, score_matrix, partition_sizes, global_rules)
            fitnesses.append(f)
            metas.append(meta)

        best_idx = int(np.argmax(fitnesses))
        best_history.append(float(fitnesses[best_idx]))

        # Elityzm
        sorted_indices = np.argsort(fitnesses)[::-1]
        next_pop = [population[i] for i in sorted_indices[:elite_size]]

        # Generowanie potomstwa
        while len(next_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)

            c1, c2 = crossover(p1, p2, pool)

            c1 = mutate(c1, mutation_rate, pool)
            c2 = mutate(c2, mutation_rate, pool)

            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        population = next_pop

    # Wynik końcowy
    final_fitnesses = []
    final_metas = []
    for ind in population:
        f, m = evaluate_segmented_portfolio(ind, score_matrix, partition_sizes, global_rules)
        final_fitnesses.append(f)
        final_metas.append(m)

    best_idx = int(np.argmax(final_fitnesses))
    return {
        "portfolio": population[best_idx],
        "fitness": final_fitnesses[best_idx],
        "meta": final_metas[best_idx],
        "convergence": best_history
    }


# =============================================
# WRAPPER API
# =============================================
def get_ga_recommendations(params: dict):
    """
    params musi zawierać:
      - portfolio_size (int)
      - global_concentration (int, 1-5)
      - segments (list of dicts): [{"ratio": 0.5, "risk": 5, ...}, ...]
    """

    # 1. Walidacja i ustawienia
    p_size = int(params.get("portfolio_size", 10))
    segments_input = params.get("segments", [])

    # Jeśli brak segmentów, stwórz domyślny 100% balanced
    if not segments_input:
        segments_input = [{"ratio": 1.0, "risk": 3, "investment_goal": "balanced"}]

    ratios = [float(s["ratio"]) for s in segments_input]
    # Normalizacja ratios do 1.0 (na wszelki wypadek)
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 0.01:
        ratios = [r / total_ratio for r in ratios]

    # 2. Oblicz wielkości partycji (Integer Partitioning)
    partition_sizes = get_partition_sizes(p_size, ratios)

    # 3. Mapowanie preferencji dla każdego segmentu
    segment_mappings = []
    for s_inp in segments_input:
        segment_mappings.append(map_single_segment_preferences(s_inp))

    global_rules = map_global_constraints(params)

    # 4. Filtrowanie puli (Excludes)
    pool = list(_STOCK_POOL)
    exclude_tickers = [t.upper() for t in params.get("exclude_tickers", [])]
    if exclude_tickers:
        pool = [s for s in pool if s.upper() not in exclude_tickers]

    # 5. Pre-kalkulacja wyników (Macierz: Segment x Spółka)
    score_matrix, meta_matrix = precalculate_all_segment_scores(pool, segment_mappings)

    # 6. Uruchomienie GA
    ga_res = run_segmented_ga(
        pool=pool,
        score_matrix=score_matrix,
        partition_sizes=partition_sizes,
        global_rules=global_rules,
        pop_size=200,
        generations=120  # Trochę mniej bo problem trudniejszy, ale smart init pomaga
    )

    # 7. Formatowanie wyniku (przypisanie spółek do segmentów)
    final_portfolio_structure = []
    portfolio_tickers = ga_res["portfolio"]
    current_idx = 0

    for i, size in enumerate(partition_sizes):
        if size == 0: continue
        segment_tickers = portfolio_tickers[current_idx: current_idx + size]

        # Pobierz metadane dla tego konkretnego segmentu
        seg_meta = meta_matrix[i]

        segment_data = {
            "segment_index": i,
            "segment_params": segments_input[i],
            "assigned_stocks": []
        }

        for ticker in segment_tickers:
            row = _STOCK_BY_SYMBOL[ticker]
            # Wynik szczegółowy dla TEGO segmentu
            score_details = seg_meta[ticker]

            stock_data = {
                "symbol": ticker,
                "name": row.get("shortName", ticker),
                "sector": row.get("sector"),
                "values": {
                    "beta": row.get("beta"),
                    "pe": row.get("pe_ratio"),
                    "div": row.get("dividend_yield")
                },
                "match_score": score_details  # Jak dobrze pasuje do TEGO segmentu
            }
            segment_data["assigned_stocks"].append(stock_data)

        final_portfolio_structure.append(segment_data)
        current_idx += size

    return {
        "structured_portfolio": final_portfolio_structure,
        "total_fitness": ga_res["fitness"],
        "stats": ga_res["meta"],
        "convergence": ga_res["convergence"]
    }
