# ImmoPrix — Projet MLOps California Housing

Projet MLOps couvrant l'intégralité du cycle de vie d'un modèle ML : exploration, préparation, entraînement, déploiement, monitoring. Scénario : prédire le prix médian des maisons en Californie pour l'entreprise fictive **ImmoPrix**.

**Stack** : `uv` · `Python 3.10` · `scikit-learn` · `DuckDB` · `MLflow` · `FastAPI` · `Streamlit` · `Docker` · `Evidently` · `GitHub Actions`

---

## Sommaire

- [Couverture des attendus](#couverture-des-attendus)
- [Arborescence](#arborescence)
- [Installation](#installation)
- [Cycle de vie complet en 6 commandes](#cycle-de-vie-complet-en-6-commandes)
- [Mission 1 — Exploration & préparation](#mission-1--exploration--préparation)
- [Mission 2 — Entraînement & MLflow](#mission-2--entraînement--mlflow)
- [Mission 3 — Analyse des features (SHAP)](#mission-3--analyse-des-features-shap)
- [Mission 4 — API FastAPI + UI Streamlit + Docker](#mission-4--api-fastapi--ui-streamlit--docker)
- [Mission 5 — CI/CD GitHub Actions](#mission-5--cicd-github-actions)
- [Mission 6 — Monitoring drift Evidently](#mission-6--monitoring-drift-evidently)
- [Déploiement cloud](#déploiement-cloud)
- [Référence des cibles Make](#référence-des-cibles-make)

---

## Couverture des attendus

| Attendu | Livré | Où |
|---|---|---|
| Code source documenté sur GitHub | ✅ | `src/mlops/`, branches `main`/`dev`/feature, commits typés `[FEAT]`/`[CHORE]`/`[CI]`/`[FIX]`/`[DOC]`/`[TEST]` |
| Rapport d'analyse exploratoire | ✅ | `notebooks/exploration.ipynb` (EDA + heatmap + NaN/outliers) |
| Enregistrement des modèles et expérimentations dans MLflow | ✅ | `src/mlops/train.py` → backend SQLite `mlflow.db`, 3 modèles loggués sous `ImmoPrix_*` |
| API déployée sur le cloud avec documentation | ⚙️ | API + Dockerfile + procédure de déploiement Render documentée ci-dessous (section *Déploiement cloud*) |
| Interface Streamlit pour tester l'API | ✅ | `src/mlops/streamlit_app.py` (`make ui`) |
| Rapport HTML d'analyse du drift avec Evidently | ✅ | `src/mlops/monitor.py` → `reports/drift_report.html` (généré aussi en artifact CI) |

---

## Arborescence

```
.
├── src/mlops/                  # Code source
│   ├── prepare.py              # Pipeline data → DuckDB
│   ├── train.py                # Entraînement + tracking MLflow
│   ├── explain.py              # Analyse SHAP (Mission 3)
│   ├── api.py                  # API FastAPI (POST /predict, /health)
│   ├── streamlit_app.py        # UI Streamlit
│   ├── simulate_production.py  # Génération données de prod avec drift simulé
│   └── monitor.py              # Détection drift Evidently → HTML
├── notebooks/                  # EDA et rapport
│   ├── data_load.ipynb
│   ├── exploration.ipynb       # ⭐ Rapport EDA principal
│   ├── preparation.ipynb
│   └── rapport.ipynb
├── tests/                      # Tests unitaires (pytest)
│   ├── conftest.py             # Mock fetch_california_housing pour CI offline
│   ├── test_api.py             # Tests FastAPI (mock du modèle)
│   └── test_train.py           # Tests pipeline data + scaling
├── .github/workflows/
│   ├── ci.yml                  # pytest + génération rapport drift (artifact)
│   └── docker.yml              # build image Docker (validation)
├── docs/
│   └── retraining.md           # Stratégies de réentraînement en cas de drift
├── Dockerfile                  # Image API
├── Makefile                    # Toutes les cibles (install, train, api, ui, monitor…)
├── pyproject.toml              # Deps gérées par uv
└── uv.lock
```

---

## Installation

### Prérequis
- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (gestionnaire de paquets ultra-rapide)
- **make** (Linux/macOS — sur Windows utiliser WSL ou `nmake`)
- **Docker** (optionnel, pour la conteneurisation)

Installation de `uv` :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Cloner et installer
```bash
git clone https://github.com/malolivier/mlops_project.git
cd mlops_project
make install      # = uv sync --frozen
```

`uv sync` crée automatiquement un venv `.venv/` avec exactement les versions de `uv.lock`.

---

## Cycle de vie complet en 6 commandes

Le pipeline complet, dans l'ordre :

```bash
make install         # 1. Installation des dépendances
make prepare         # 2. Téléchargement + préparation des données → data/processed/housing.duckdb
make train           # 3. Entraînement de 3 modèles + log MLflow
make mlflow-ui       # 4. (terminal séparé) UI MLflow http://localhost:5000
make api             # 5. (terminal séparé) API FastAPI http://localhost:8000
make ui              # 6. (terminal séparé) UI Streamlit http://localhost:8501
```

Puis monitoring :
```bash
make monitor         # → reports/drift_report.html
```

Et tests :
```bash
make test            # pytest tests/
```

---

## Mission 1 — Exploration & préparation

**Livrable** : `notebooks/exploration.ipynb` (rapport EDA complet)

Couvre :
- Statistiques descriptives (`info()`, `describe().T`)
- Distributions (histogrammes 3×3)
- Matrice de corrélation (heatmap) → `MedInc` ressort comme top prédicteur
- Visualisation géographique (lat/lon coloré par prix)
- Détection des valeurs manquantes (aucune) et outliers (Tukey IQR — `AveOccup`, `AveRooms`, `Population` flaggés mais conservés car typiques du dataset)
- Conclusion : pas de NaN, target capping à 5.0, outliers gardés

Pour ouvrir le notebook :
```bash
jupyter lab notebooks/exploration.ipynb
```

Pipeline data en script (équivalent prod) :
```bash
make prepare         # → data/processed/housing.duckdb, table `california_housing` (19 648 lignes)
```

---

## Mission 2 — Entraînement & MLflow

**Livrable** : 3 modèles enregistrés dans MLflow Registry.

```bash
make train
```

Entraîne séquentiellement :
1. **`ImmoPrix_baseline`** — `LinearRegression`
2. **`ImmoPrix_rf`** — `RandomForestRegressor(n_estimators=100, max_depth=15)`
3. **`ImmoPrix_gradient_boosting`** — `GradientBoostingRegressor(n_estimators=150, learning_rate=0.05)`

Pour chaque modèle, MLflow enregistre :
- Hyperparamètres (`model_type`, `n_estimators`, etc.)
- Métriques (`RMSE`, `MAE`, `R²`)
- Artifact modèle (sklearn pickle)
- Version dans le Model Registry

**Backend** : SQLite local `mlflow.db` (path relatif, portable entre collaborateurs).

Visualisation des runs :
```bash
make mlflow-ui       # http://localhost:5000
```

---

## Mission 3 — Analyse des features (SHAP)

**Livrable** : Graphique global (Summary Plot) et graphique local (Waterfall Plot) (voir Mlflow -> SHAP_Explicability -> Artifacts)

```bash
make explain
```

Génère les explications SHAP (importance globale + locale) pour le modèle entraîné. Documente l'impact des features sur les prédictions.

---

## Mission 4 — API FastAPI + UI Streamlit + Docker

### API FastAPI

```bash
make api             # http://localhost:8000
```

- **Doc Swagger auto-générée** : http://localhost:8000/docs
- **Endpoints** :
  - `GET /health` → état du service + URI du modèle chargé
  - `POST /predict` → prédiction de prix

Exemple :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 20, "AveRooms": 5, "AveBedrms": 1,
       "Population": 1000, "AveOccup": 3, "Latitude": 34, "Longitude": -118}'
```

Réponse :
```json
{
  "predicted_price_100k_usd": 2.345,
  "model_uri": "models:/ImmoPrix_rf/latest"
}
```

Le modèle est chargé au démarrage depuis le MLflow Registry (`ImmoPrix_rf/latest`). Validation des entrées via **Pydantic** (bornes lat/lon, valeurs positives, etc.).

### UI Streamlit

Une fois **l'API lancée**, dans un second terminal à la racine du projet

```bash
make ui              # http://localhost:8501
```

Formulaire des 8 features + appel `/predict` + affichage du prix en $. Indicateur de santé de l'API en sidebar.

### Docker

```bash
docker build -t immoprix-api .
docker run --rm -p 8000:8000 \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  immoprix-api
```

L'image utilise `uv` et un build multi-étape pour le cache des deps. Volumes montés pour partager modèles et BDD.

### Serving natif via MLflow

Alternative à l'API FastAPI custom :
```bash
make mlflow-serve    # http://localhost:5001
```

Démarre `mlflow models serve` qui expose le modèle directement en REST sans code custom.

---

## Mission 5 — CI/CD GitHub Actions

**Livrable** : `.github/workflows/{ci,docker}.yml`.

### Workflows
- **`ci.yml`** déclenché sur push/PR vers `main` et `dev` :
  - Job `test` : `pytest tests/ -v` (8 tests : 5 sur l'API mockée + 3 sur le pipeline data)
  - Job `drift` : `prepare → simulate_production → monitor` + upload `drift_report.html` en **artifact GitHub Actions** (rétention 30 jours)
- **`docker.yml`** : valide que l'image Docker se construit (build no-push, cache GHA)

### Tests
- `tests/test_api.py` : `TestClient` FastAPI avec model + scaler mockés (rapide, aucun MLflow requis)
- `tests/test_train.py` : pipeline data + train/test split + standardization (utilise mock du dataset via `conftest.py` pour éviter le download réseau en CI)
- `tests/conftest.py` : fixture session-scope qui mock `fetch_california_housing` avec 1000 lignes synthétiques (seed=42)

Lancement local :
```bash
make test
```

---

## Mission 6 — Monitoring drift Evidently

**Livrable** : `reports/drift_report.html` + stratégies de réentraînement (`docs/retraining.md`).

```bash
make monitor         # = simulate-prod + monitor
```

1. `simulate_production.py` génère 2 000 lignes de "production" en appliquant des shifts intentionnels :
   - `MedInc` × 1.20 (inflation)
   - `HouseAge` + 5 (vieillissement du parc)
   - `Population` × 1.30 (croissance démo)
   - Bruit gaussien sur `Latitude` / `Longitude`
2. `monitor.py` compare les données d'entraînement à la production via Evidently `DataDriftPreset` (Wasserstein normalisé, seuil 0.1).
3. Rapport HTML interactif généré dans `reports/drift_report.html`.

**Résultats typiques de la simulation** : 4 features sur 9 driftées (`MedInc` 0.67, `HouseAge` 0.44, `Population` 0.33, `MedHouseVal` 0.20). Décision = réentraîner — voir `docs/retraining.md` pour les seuils, la stratégie champion/challenger et la boucle MLOps complète.

Le rapport est aussi **disponible en artifact GitHub Actions** à chaque push (onglet Actions → run → Artifacts → `drift-report`).

---

## Déploiement cloud

Le Dockerfile est compatible avec n'importe quel PaaS qui consomme une image. Procédure documentée pour **Render** (free tier, recommandé pour démo) :

### Option A — Render (Docker)

1. Pousser une image Docker publique vers GHCR ou Docker Hub :
   ```bash
   docker build -t ghcr.io/<user>/immoprix-api:latest .
   docker push ghcr.io/<user>/immoprix-api:latest
   ```

2. Sur [render.com](https://render.com) :
   - **New** → **Web Service** → **Existing image**
   - Image URL : `ghcr.io/<user>/immoprix-api:latest`
   - Port : `8000`
   - Health check path : `/health`

3. Limitation : Render free tier ne permet pas de monter de volumes. Il faut donc **embarquer le modèle dans l'image** (modifier le Dockerfile pour `COPY mlruns/`, `COPY mlflow.db`, ou pousser le modèle sur S3 et l'y récupérer au démarrage).

### Option B — Hugging Face Spaces (Streamlit + API combinées)

Pour un démo public gratuit, créer un Space type **Streamlit** ou **Docker** et y pousser le code. L'UI Streamlit peut alors faire des appels en local (`localhost:8000`) si l'API tourne dans le même container, ou directement appeler le modèle en `mlflow.sklearn.load_model`.

### Option C — Fly.io

`fly.toml` minimal :
```toml
app = "immoprix-api"
[[services]]
  internal_port = 8000
  [[services.ports]]
    port = 80
    handlers = ["http"]
```
Puis `fly launch --image-from-current-dir && fly deploy`.

---

## Référence des cibles Make

| Cible | Description |
|---|---|
| `make install` | Synchronise les dépendances via uv |
| `make format` | Formate le code avec Black |
| `make lint` | Vérification syntaxique Flake8 |
| `make test` | Lance pytest sur `tests/` |
| `make prepare` | Pipeline data → DuckDB |
| `make train` | Entraîne les 3 modèles + log MLflow |
| `make explain` | Analyse SHAP |
| `make api` | Démarre FastAPI sur :8000 (reload activé) |
| `make ui` | Démarre Streamlit sur :8501 |
| `make mlflow-ui` | Démarre l'UI MLflow sur :5000 |
| `make mlflow-serve` | Sert le modèle natively via MLflow sur :5001 |
| `make simulate-prod` | Génère des données de production avec drift |
| `make monitor` | `simulate-prod` puis génère le rapport Evidently |
| `make all` | install + format + lint + test + train |
| `make clean` | Supprime `.venv`, caches, `data/` |

---

## Branches Git

Workflow GitFlow simplifié :
- **`main`** : version stable
- **`dev`** : intégration des features
- **`feature/*`** : développements (ex. `api`, `ci-cd`, `monitoring`, `bdd`, `fix/mlflow-tracking`)

Toutes les features passent par une PR `feature → dev` validée par les workflows CI avant merge.

---

## Auteurs

Projet réalisé dans le cadre du cours MLOps — Université de Lille.
