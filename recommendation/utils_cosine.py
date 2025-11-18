import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("recommendation/sp500_companies_final.csv", header=None)
df.columns = ["symbol", "name", "sector", "beta", "pe_ratio", "market_cap", "dividend_yield"]

df["name"] = df["name"].str.replace('"', '', regex=False)
df["sector"] = df["sector"].str.replace('"', '', regex=False)
for col in ["beta", "pe_ratio", "market_cap", "dividend_yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].fillna(df[col].median(), inplace=True)

# normalizacja i one-hot encoding
scaler = MinMaxScaler()
numeric_cols = ["beta", "pe_ratio", "market_cap", "dividend_yield"]
numeric_scaled = scaler.fit_transform(df[numeric_cols])
numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_cols)

encoder = OneHotEncoder(sparse_output=False)
sector_encoded = encoder.fit_transform(df[["sector"]])
sector_df = pd.DataFrame(sector_encoded, columns=encoder.get_feature_names_out(["sector"]))

company_vectors = pd.concat([numeric_df, sector_df], axis=1)


def get_recommendations(sector_allocations, global_params, top_n=10):
    """
    sector_allocations: dict, np. {"Technology": 0.5, "Healthcare": 0.3, "Financial Services": 0.2}
    global_params: dict, np. {"beta": 0.5, "pe_ratio": 20, "market_cap": 200e9, "dividend_yield": 1.5}
    """

    results = []

    user_data = pd.DataFrame([[
        global_params["beta"],
        global_params["pe_ratio"],
        global_params["market_cap"],
        global_params["dividend_yield"]
    ]], columns=["beta", "pe_ratio", "market_cap", "dividend_yield"])

    user_numeric = scaler.transform(user_data)[0]

    for sector, weight in sector_allocations.items():
        sector_col = f"sector_{sector}"
        user_sector = np.zeros(len(sector_df.columns))
        if sector_col in sector_df.columns:
            user_sector[sector_df.columns.get_loc(sector_col)] = 1

        user_vector = np.concatenate([user_numeric, user_sector])

        similarities = cosine_similarity([user_vector], company_vectors)[0]
        df["similarity"] = similarities

        # tylko spółki z danego sektora
        sector_stocks = df[df["sector"] == sector].copy()
        if sector_stocks.empty:
            continue

        sector_stocks = sector_stocks.sort_values(by="similarity", ascending=False)
        n_stocks = max(1, round(top_n * weight))
        sector_recs = sector_stocks.head(n_stocks)

        results.append(sector_recs)

    final_recommendations = pd.concat(results).sort_values(by="similarity", ascending=False)
    return final_recommendations[["symbol", "name", "sector", "similarity"]].to_dict(orient="records")
