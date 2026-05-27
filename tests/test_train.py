import os
import duckdb
import numpy as np
import pytest
from mlops.prepare import run_data_pipeline
from mlops.train import load_data_from_duckdb


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Fixture qui prépare l'environnement de test en générant une base DuckDB dédiée."""
    # On utilise des fichiers temporaires pour ne pas polluer les vraies données de dev
    test_dir = "data/test_processed"
    test_db_name = "test_housing.duckdb"
    test_db_path = os.path.join(test_dir, test_db_name)

    # 1. On lance le pipeline de données pour générer la base de test
    run_data_pipeline(db_dir=test_dir, db_name=test_db_name)

    # On fournit le chemin de cette base aux fonctions de test
    yield test_db_path

    # 2. Nettoyage après l'exécution de TOUS les tests du module
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)


def test_pipeline_data_creation(setup_test_environment):
    """Vérifie que le pipeline de données crée bien la base et la table attendues."""
    db_path = setup_test_environment
    assert os.path.exists(db_path), "Le fichier DuckDB n'a pas été créé."

    # Connexion pour vérifier le contenu
    conn = duckdb.connect(db_path, read_only=True)
    tables = [
        t[0]
        for t in conn.execute("SHOW TABLES").fetchall()
    ]
    conn.close()

    assert (
        "california_housing" in tables
    ), "La table 'california_housing' est manquante."


def test_load_data_splits(setup_test_environment):
    """Vérifie que le chargement et le découpage Train/Test fonctionnent (80/20)."""
    db_path = setup_test_environment

    # On surcharge temporairement la fonction en lui passant notre base de test
    X_train, X_test, y_train, y_test = load_data_from_duckdb(
        db_path=db_path
    )

    # Vérification des proportions
    total_samples = (
        X_train.shape[0] + X_test.shape[0]
    )
    # Le dataset nettoyé (sans outliers > 5.0) fait environ 19648 lignes
    assert X_train.shape[0] == int(total_samples * 0.8)
    assert (
        len(y_train) == X_train.shape[0]
    )


def test_prepare_data_scaling(setup_test_environment):
    """Vérifie que la standardisation (StandardScaler) centre bien les données à 0."""
    db_path = setup_test_environment
    X_train, _, _, _ = load_data_from_duckdb(db_path=db_path)

    # La moyenne de chaque feature standardisée doit être extrêmement proche de 0
    means = np.mean(X_train, axis=0)
    for mean in means:
        assert (
            abs(mean) < 1e-7
        ), f"La feature n'est pas bien centrée (moyenne={mean})"