# Variables
UV = uv

.PHONY: install format lint test train all clean

# Installe et synchronise tout le pyproject.toml
install:
	@echo "--- Synchronisation des dépendances avec uv ---"
	$(UV) sync

# Formatage du code
format:
	@echo "--- Formatage avec Black ---"
	$(UV) run black src/ tests/

# Vérification syntaxique (Linting)
lint:
	@echo "--- Vérification avec Flake8 ---"
	$(UV) run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(UV) run flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Exécution des tests unitaires avec le bon exécutable python
test:
	@echo "--- Exécution des tests unitaires avec pytest ---"
	$(UV) run python -m pytest tests/ -v

# Lancement de l'entraînement
train:
	@echo "--- Lancement de l'entraînement du modèle ---"
	$(UV) run python -m mlops.train

# Nettoyage
clean:
	rm -rf .venv .pytest_cache .build __pycache__ src/mlops/__pycache__ tests/__pycache__ uv.lock

# Pipeline complet de CI local
all: install format lint test train