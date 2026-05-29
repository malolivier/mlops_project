"""Simule des données de production pour tester le monitoring de drift.

On part du dataset California Housing réel, on en échantillonne un sous-ensemble
et on applique des shifts intentionnels pour simuler des scénarios de drift
crédibles (inflation, vieillissement du parc immobilier, croissance démographique).

Sauvegarde dans `data/production/california_housing_production.duckdb`,
table `housing_production`.

Lance via `make simulate-prod` ou :
    uv run python -m mlops.simulate_production
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

PROD_DB = Path("data/production/california_housing_production.duckdb")
TABLE = "housing_production"
N_SAMPLES = 2000
SEED = 123


def simulate_drift(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Échantillonne le dataset réel et applique des shifts simulant du drift."""
    rng = np.random.default_rng(seed)
    full = fetch_california_housing(as_frame=True).frame
    sample = full.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    # Drift "inflation" : revenus médians +20 %
    sample["MedInc"] = sample["MedInc"] * 1.20
    # Drift "vieillissement du parc" : maisons +5 ans en moyenne
    sample["HouseAge"] = sample["HouseAge"] + 5
    # Drift "croissance démographique" : population +30 %
    sample["Population"] = sample["Population"] * 1.30
    # Bruit léger sur les coordonnées (nouveaux secteurs cartographiés)
    sample["Latitude"] = sample["Latitude"] + rng.normal(0, 0.1, n_samples)
    sample["Longitude"] = sample["Longitude"] + rng.normal(0, 0.1, n_samples)

    return sample


def main() -> None:
    print("--- Simulation des données de production ---")
    PROD_DB.parent.mkdir(parents=True, exist_ok=True)
    df = simulate_drift()
    conn = duckdb.connect(str(PROD_DB))
    conn.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM df")
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    conn.close()
    print(f"Données simulées : {count} lignes -> {PROD_DB}")


if __name__ == "__main__":
    main()
