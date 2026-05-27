# Variables
UV = uv
MLFLOW_DB = mlflow.db

.PHONY: install format lint test prepare train mlflow-ui all clean

# 1. Installation et synchronisation des dépendances
install:
	@echo "--- Synchronisation des dépendances avec uv ---"
	$(UV) sync

# 2. Formatage avec Black
format:
	@echo "--- Formatage avec Black ---"
	$(UV) run black src/ tests/

# 3. Vérification syntaxique (Linting)
lint:
	@echo "--- Vérification syntaxique avec Flake8 ---"
	$(UV) run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(UV) run flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 4. Exécution des tests unitaires
test:
	@echo "--- Exécution des tests unitaires avec pytest ---"
	$(UV) run python -m pytest tests/ -v

# 5. Extraction et préparation des données (Mission 1)
prepare:
	@echo "--- Exécution du pipeline de données (DuckDB) ---"
	$(UV) run python -m mlops.prepare

# 6. Entraînement des modèles et tracking (Mission 2)
train: prepare
	@echo "--- Lancement de l'entraînement des modèles ---"
	$(UV) run python -m mlops.train

# 7. Analyse SHAP (Mission 3)
explain: prepare
	@echo "--- Génération des explications SHAP ---"
	$(UV) run python -m mlops.explain

# 8. Démarrage du serveur MLflow UI (Ajout MLOps)
mlflow-ui:
	@echo "--- Démarrage du serveur MLflow sur http://127.0.0.1:5000 ---"
	$(UV) run mlflow ui --backend-store-uri sqlite:///$(MLFLOW_DB)

# Nettoyage complet
clean:
	rm -rf .venv .pytest_cache .build __pycache__
	rm -rf src/mlops/__pycache__ tests/__pycache__
	rm -rf data/

# Pipeline de validation locale complet (sans bloquer sur l'UI)
all: install format lint test train