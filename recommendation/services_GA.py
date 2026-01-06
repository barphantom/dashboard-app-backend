import random
import copy
import math
import pandas as pd
import numpy as np
import os
from django.conf import settings

CSV_FILE_NAME = "sp500_companies_final.csv"
POPULATION_SIZE = 50
GENERATIONS = 100
ELITISM_COUNT = 2


def load_data():
    csv_path = os.path.join(settings.BASE_DIR, 'recommendation', CSV_FILE_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {csv_path}")
    return pd.read_csv(csv_path)


def map_user_segment(user_segment: dict):
    ratio = user_segment.get("ratio", 3)

    risk_input = user_segment.get("risk", 3)
    goal_input = user_segment.get("investment_goal", "balanced")
    horizon_input = user_segment.get("time_horizon", "long")

    target_beta_map = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
    risk = user_segment.get("risk", 3)
    target_beta = target_beta_map.get(risk, 1.0)

    goal = user_segment.get("investment_goal", "balanced")
    if goal == "growth":
        target_pe_ratio = 40.0
        target_dividend_yield = 0.5
        weights = {'w_beta': 0.4, 'w_pe': 0.4, 'w_div': 0.2}
    elif goal == "income":
        target_pe_ratio = 15.0
        target_dividend_yield = 4.0
        weights = {'w_beta': 0.2, 'w_pe': 0.3, 'w_div': 0.5}
    else:
        target_pe_ratio = 20
        target_dividend_yield = 2.0
        weights = {'w_beta': 0.3, 'w_pe': 0.3, 'w_div': 0.4}

    time_horizon = user_segment.get("time_horizon", "long")
    if time_horizon == "long":
        target_market_cap = 1000000000000
    elif time_horizon == "medium":
        target_market_cap = 100000000000
    else:
        target_market_cap = 10000000000

    segment_mapping = {
        "ratio": float(ratio),
        "original_risk": risk_input,
        "original_goal": goal_input,
        "original_horizon": horizon_input,
        "target_beta": float(target_beta),
        "target_pe_ratio": float(target_pe_ratio),
        "target_dividend_yield": float(target_dividend_yield),
        "target_market_cap": float(target_market_cap),
        "weights": weights,
    }

    return segment_mapping


def map_user_request(user_request: dict):
    portfolio_size = int(user_request.get("portfolio_size", 10))

    target_concentration_map = {1: 30, 2: 35, 3: 40, 4: 45, 5: 50}
    concentration = int(user_request.get("concentration", 1))
    target_concentration = target_concentration_map.get(concentration, 30)

    user_segments = user_request.get("segments", [])
    segment_mappings = []
    for segment in user_segments:
        segment_mappings.append(map_user_segment(segment))

    mapping = {
        "portfolio_size": portfolio_size,
        "concentration_percent": float(target_concentration),
        "segments": segment_mappings,
    }

    return mapping


def calculate_similarity_score(row, targets):
    weights = targets.get("weights", {})

    diff_beta = abs(row["beta"] - targets["target_beta"])
    score_beta = 1.0 / (1.0 + diff_beta)

    diff_pe_ratio = abs(row["pe_ratio"] - targets["target_pe_ratio"])
    score_pe_ratio = 1.0 / (1.0 + (diff_pe_ratio / targets["target_pe_ratio"]))

    diff_dividend = abs(row["dividend_yield"] - targets["target_dividend_yield"])
    score_dividend = 1.0 / (1.0 + diff_dividend)

    final_score = (score_beta * weights["w_beta"]) + (score_pe_ratio * weights["w_pe"]) + (
                score_dividend * weights["w_div"])
    return final_score


def create_stock_pools(df, mapped_req: dict):
    stock_pools = []
    segments_criteria = mapped_req.get("segments")

    for segment in segments_criteria:
        scores = df.apply(lambda row: calculate_similarity_score(row, segment), axis=1)
        sorted_indexes = scores.argsort()[::-1]
        best_indexes = sorted_indexes[:100].to_list()

        stock_pools.append(best_indexes)

    return stock_pools


def get_partition_sizes(portfolio_size, segments):
    ratios = [percentage["ratio"] for percentage in segments]

    raw_sizes = [portfolio_size * ratio for ratio in ratios]
    int_sizes = [int(math.floor(raw_size)) for raw_size in raw_sizes]

    remainder = portfolio_size - sum(int_sizes)

    diffs = []
    for i in range(len(raw_sizes)):
        diff = raw_sizes[i] - int_sizes[i]
        diffs.append((i, diff))

    diffs.sort(key=lambda part: part[1], reverse=True)

    for n in range(remainder):
        int_sizes[diffs[n][0]] += 1

    return int_sizes


def initialize_population(population_size, pools, partition_sizes):
    population = []

    for _ in range(population_size):
        chromosome = []
        global_used_indices = set()

        for index, par_size in enumerate(partition_sizes):
            pool = pools[index]

            available_unique = [index for index in pool if index not in global_used_indices]

            segment = []

            if len(available_unique) >= par_size:
                segment = random.sample(available_unique, k=par_size)
            else:
                segment.extend(available_unique)

                stocks_indexes_needed = par_size - len(segment)
                remaining_in_pool = [x for x in pool if x not in segment]
                if len(remaining_in_pool) >= stocks_indexes_needed:
                    segment.extend(random.sample(remaining_in_pool, k=stocks_indexes_needed))
                else:
                    segment.extend(random.choices(pool, k=stocks_indexes_needed))

            # if len(pool) < par_size:
            #     segment = random.choices(pool, k=par_size)
            # else:
            #     segment = random.sample(pool, k=par_size)
            global_used_indices.update(segment)
            chromosome.append(segment)

        population.append(chromosome)

    return population


def calculate_concentration_penalty(all_chromosome_indexes, df, concentration_limit_percent):
    sectors = df.loc[all_chromosome_indexes, ["sector"]]
    if sectors.empty:
        return 0.0

    sector_counts = sectors.value_counts()

    max_sector_count = sector_counts.iloc[0]
    total_stocks = len(all_chromosome_indexes)

    current_sector_share = (max_sector_count / total_stocks) * 100.0
    if current_sector_share > concentration_limit_percent:
        diff = current_sector_share - concentration_limit_percent

        penalty_multiplier = 0.05
        return diff * penalty_multiplier

    return 0.0


def calculate_segment_fitness(segment, df, targets):
    weights = targets.get("weights", {'w_beta': 0.33, 'w_pe': 0.33, 'w_div': 0.33})

    beta_target = targets.get("target_beta")
    pe_target = targets.get("target_pe_ratio")
    div_target = targets.get("target_dividend_yield")

    betas = df.loc[segment, "beta"]
    pe_ratios = df.loc[segment, "pe_ratio"]
    div_yields = df.loc[segment, "dividend_yield"]

    avg_beta = np.mean(betas)
    avg_pe = np.mean(pe_ratios)
    avg_div = np.mean(div_yields)

    safe_beta_t = max(abs(beta_target), 0.01)
    safe_pe_t = max(abs(pe_target), 0.01)
    safe_div_t = max(abs(div_target), 0.01)

    error_avg_beta = abs(avg_beta - beta_target) / safe_beta_t
    error_avg_pe = abs(avg_pe - pe_target) / safe_pe_t
    if div_target < 0.1:
        error_avg_div = abs(avg_div - div_target)
    else:
        error_avg_div = abs(avg_div - div_target) / safe_div_t

    coherence_beta = np.std(betas) / safe_beta_t
    coherence_pe = np.std(pe_ratios) / safe_pe_t
    if div_target < 0.1:
        coherence_div = np.std(div_yields)
    else:
        coherence_div = np.std(div_yields) / safe_div_t

    total_penalty = 0.0
    total_penalty += weights['w_beta'] * (error_avg_beta + 0.5 * coherence_beta)
    total_penalty += weights['w_pe'] * (error_avg_pe + 0.3 * coherence_pe)
    total_penalty += weights['w_div'] * (error_avg_div + 0.2 * coherence_div)

    return 1.0 / (1.0 + total_penalty)


def calculate_total_fitness(chromosome, df, mapped_request):
    segments_criteria = mapped_request.get("segments")
    all_stocks_indexes = []
    segments_fitnesses = []

    for i, segment_indexes in enumerate(chromosome):
        all_stocks_indexes.extend(segment_indexes)
        segment_fitness = calculate_segment_fitness(segment_indexes, df, segments_criteria[i])
        segments_fitnesses.append(segment_fitness)

    average_chromosome_fitness = np.mean(segments_fitnesses)

    concentration_limit = mapped_request.get("concentration_percent", 30.0)
    concentration_penalty = calculate_concentration_penalty(all_stocks_indexes, df, concentration_limit)

    final_fitness = average_chromosome_fitness - concentration_penalty

    return max(0.0, final_fitness)


def tournament_selection(population, fitness_scores, k=3):
    candidates_indexes = random.sample(range(len(population)), k=k)

    best_score_index = candidates_indexes[0]
    best_score = fitness_scores[best_score_index]

    for index in candidates_indexes[1:]:
        if fitness_scores[index] > best_score:
            best_score = fitness_scores[index]
            best_score_index = index

    return copy.deepcopy(population[best_score_index])


def set_union_crossover(parent1, parent2):
    child1 = []
    child2 = []

    for i in range(len(parent1)):
        segment1 = parent1[i]
        segment2 = parent2[i]
        target_size = len(segment1)

        game_pool = list(set(segment1 + segment2))

        if len(game_pool) <= target_size:
            child1.append(segment1[:])
            child2.append(segment2[:])
            continue

        random.shuffle(game_pool)
        child1.append(game_pool[:target_size])

        random.shuffle(game_pool)
        child2.append(game_pool[:target_size])

    return child1, child2


def mutate(chromosome, stock_pools, mutation_rate=0.02):
    mutated_chromosome = []

    used_stocks = set()
    for segment in chromosome:
        used_stocks.update(segment)

    for seg_i, segment in enumerate(chromosome):
        mutated_segment = []

        for index_pos in range(len(segment)):
            if random.random() <= mutation_rate:
                available = [s for s in stock_pools[seg_i] if s not in used_stocks]

                if available:
                    old_stock = segment[index_pos]
                    new_stock = random.choice(available)

                    used_stocks.remove(old_stock)
                    used_stocks.add(new_stock)

                    mutated_segment.append(new_stock)
                else:
                    mutated_segment.append(segment[index_pos])
            else:
                mutated_segment.append(segment[index_pos])

        mutated_chromosome.append(mutated_segment)

    return mutated_chromosome


def repair_chromosome(chromosome, stock_pools):
    repaired_chromosome = []
    used_stocks = set()

    for i, segment in enumerate(chromosome):
        new_segment = []
        pool = stock_pools[i]

        for stock in segment:
            if stock not in used_stocks:
                used_stocks.add(stock)
                new_segment.append(stock)
            else:
                available = [s for s in pool if s not in used_stocks]

                if available:
                    new_stock = random.choice(available)
                    used_stocks.add(new_stock)
                    new_segment.append(new_stock)
                else:
                    new_segment.append(stock)

        repaired_chromosome.append(new_segment)

    return repaired_chromosome


def format_portfolio_response(best_chromosome, best_fitness, df, mapped_request):
    segments_criteria = mapped_request.get("segments")
    all_indices = []
    segments_result = []

    for i, segment_indices in enumerate(best_chromosome):
        criteria = segments_criteria[i]
        segment_df = df.loc[segment_indices]
        stocks_in_segment = []

        for idx, row in segment_df.iterrows():
            stocks_in_segment.append({
                "symbol": row['symbol'],
                "shortName": row.get('shortName', ''),
                "sector": row['sector'],
                "beta": row['beta'],
                "pe_ratio": row['pe_ratio'],
                "dividend_yield": row['dividend_yield'],
            })

        avg_beta = segment_df['beta'].mean()
        avg_pe = segment_df['pe_ratio'].mean()
        avg_div = segment_df['dividend_yield'].mean()

        segment_data = {
            "segment_id": i + 1,
            "config": {
                "ratio": criteria['ratio'],
                "goal": criteria.get('original_goal'),
                "risk": criteria.get('original_risk'),
                "horizon": criteria.get('original_horizon'),
            },
            "stats": {
                "avg_beta": round(avg_beta, 2) if not pd.isna(avg_beta) else 0,
                "avg_pe_ratio": round(avg_pe, 2) if not pd.isna(avg_pe) else 0,
                "avg_dividend_yield": round(avg_div, 2) if not pd.isna(avg_div) else 0
            },
            "stocks": stocks_in_segment
        }
        segments_result.append(segment_data)
        all_indices.extend(segment_indices)

    total_df = df.loc[all_indices]
    sectors = total_df['sector'].value_counts().to_dict()

    return {
        "fitness_score": round(best_fitness, 4),
        "total_items": len(all_indices),
        "sector_concentration": sectors,
        "segments": segments_result
    }


def run_genetic_algorithm(user_request_data):
    df = load_data()

    # Mapowanie requestu od użytkownika
    mapped_request = map_user_request(user_request_data)

    # Inicjalizacja
    stock_pools = create_stock_pools(df, mapped_request)
    partition_sizes = get_partition_sizes(mapped_request["portfolio_size"], mapped_request["segments"])
    population = initialize_population(POPULATION_SIZE, stock_pools, partition_sizes)

    global_best_fitness = -1.0
    global_best_chromosome = None

    # Pętla ewolucyjna
    for g in range(GENERATIONS):
        population_scores = []
        ranked_population = []

        for chromosome in population:
            fitness = calculate_total_fitness(chromosome, df, mapped_request)
            population_scores.append(fitness)
            ranked_population.append((fitness, chromosome))

        ranked_population.sort(key=lambda x: x[0], reverse=True)

        current_best_fitness = ranked_population[0][0]
        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_chromosome = copy.deepcopy(ranked_population[0][1])

        new_population = []

        # Elityzm
        for i in range(ELITISM_COUNT):
            new_population.append(copy.deepcopy(ranked_population[i][1]))

        slots_remaining = POPULATION_SIZE - ELITISM_COUNT
        pairs_needed = slots_remaining // 2

        for _ in range(pairs_needed):
            winner1 = tournament_selection(population, population_scores, k=3)
            winner2 = tournament_selection(population, population_scores, k=3)

            cross_ch1, cross_ch2 = set_union_crossover(winner1, winner2)

            child1 = mutate(cross_ch1, stock_pools)
            child2 = mutate(cross_ch2, stock_pools)

            child1 = repair_chromosome(child1, stock_pools)
            child2 = repair_chromosome(child2, stock_pools)

            new_population.append(child1)
            new_population.append(child2)

        if len(new_population) < POPULATION_SIZE:
            winner = tournament_selection(population, population_scores, k=3)
            mutated_winner = mutate(winner, stock_pools)
            new_population.append(repair_chromosome(mutated_winner, stock_pools))

        population = new_population

    return format_portfolio_response(global_best_chromosome, global_best_fitness, df, mapped_request)