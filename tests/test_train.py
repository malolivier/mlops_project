import os
import numpy as np
import pytest
import duckdb
import pandas as pd
from sklearn.datasets import fetch_california_housing
from mlops.train import load_data_from_duckdb

@pytest.fixture(scope="module")
def setup_test_duckdb():
    """Fixture qui crée une base DuckDB temporaire avec quelques lignes pour les tests."""
    test_db_dir = "data/processed"
    test_db_path = f"{test_db_dir}/housing.duckdb"
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(test_db_dir, exist_ok=True)
    
    # Charger un tout petit échantillon (100 lignes) pour que le test soit ultra rapide
    housing = fetch_california_housing(as_frame=True)
    df_sample = housing.frame.head(100)
    
    # Écrire dans la base DuckDB de test
    conn = duckdb.connect(test_db_path)
    conn.execute("CREATE OR REPLACE TABLE california_housing AS SELECT * FROM df_sample")
    conn.close()
    
    yield test_db_path
    
    # Nettoyage après les tests si nécessaire (optionnel)
    # os.remove(test_db_path)

def test_load_data_from_duckdb_shapes(setup_test_duckdb):
    """Vérifie que le découpage des données respecte les proportions (80/20) sur notre échantillon."""
    X_train, X_test, y_train, y_test = load_data_from_duckdb(db_path=setup_test_duckdb)

    # Sur 100 lignes mockées : 80 en train, 20 en test
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_prepare_data_scaling(setup_test_duckdb):
    """Vérifie que les données d'entraînement sont bien centrées-réduites (moyenne proche de 0)."""
    X_train, _, _, _ = load_data_from_duckdb(db_path=setup_test_duckdb)

    # Après un StandardScaler, la moyenne de chaque feature doit être très proche de 0
    means = np.mean(X_train, axis=0)
    for mean in means:
        assert abs(mean) < 1e-7