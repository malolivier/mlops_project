"""Détection de data drift entre l'entraînement et la production via Evidently.

Compare la table `california_housing` (réf. entraînement) à la table
`housing_production` (données simulées par `simulate_production.py`) et génère
un rapport HTML dans `reports/drift_report.html`.

Lance via `make monitor` ou :
    uv run python -m mlops.monitor
"""
from pathlib import Path

import duckdb
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

TRAIN_DB = Path("data/processed/housing.duckdb")
TRAIN_TABLE = "california_housing"
PROD_DB = Path("data/production/california_housing_production.duckdb")
PROD_TABLE = "housing_production"
REPORT_PATH = Path("reports/drift_report.html")


def _load(db_path: Path, table: str) -> pd.DataFrame:
    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute(f"SELECT * FROM {table}").df()
    conn.close()
    return df


def main() -> None:
    print("--- Détection de drift via Evidently ---")
    if not TRAIN_DB.exists():
        raise FileNotFoundError(f"Reference DB manquante : {TRAIN_DB}. Lancer `make prepare`.")
    if not PROD_DB.exists():
        raise FileNotFoundError(f"Production DB manquante : {PROD_DB}. Lancer `make simulate-prod`.")

    reference = _load(TRAIN_DB, TRAIN_TABLE)
    current = _load(PROD_DB, PROD_TABLE)
    print(f"Référence : {reference.shape}, Production : {current.shape}")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    print(f"Rapport HTML : {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
