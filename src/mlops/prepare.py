import os
import duckdb
from sklearn.datasets import fetch_california_housing

def run_data_pipeline(db_dir="data/processed", db_name="housing.duckdb"):
    """Orchestre le chargement, le nettoyage et la sauvegarde dans DuckDB."""
    print("--- Démarrage du pipeline de données ---")
    
    # 1. Création du dossier de stockage
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, db_name)
    
    # 2. Chargement des données brutes (Équivalent à data_load.ipynb)
    print("Chargement des données depuis Scikit-Learn...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # 3. Traitement et Nettoyage (Équivalent à preparation.ipynb)
    print("Nettoyage des données (Filtrage des anomalies)...")
    # Exemple de nettoyage classique sur California Housing :
    # Les prix sont plafonnés à 5.0, ce qui crée un biais. On retire ce plafond.
    df_cleaned = df[df['MedHouseVal'] < 5.0]
    
    # 4. Stockage dans DuckDB
    print(f"Sauvegarde de la table nettoyée dans {db_path}...")
    conn = duckdb.connect(db_path)
    # On écrase/crée la table 'california_housing' avec le dataframe nettoyé
    conn.execute("CREATE OR REPLACE TABLE california_housing AS SELECT * FROM df_cleaned")
    
    # Vérification rapide du nombre de lignes
    count = conn.execute("SELECT COUNT(*) FROM california_housing").fetchone()[0]
    print(f"Pipeline terminé avec succès ! Nombre de lignes stockées : {count}")
    conn.close()

if __name__ == "__main__":
    run_data_pipeline()