# Stratégies de réentraînement en cas de data drift

Document de référence pour décider **quand** et **comment** réentraîner le modèle `ImmoPrix_rf` lorsque le monitoring (`make monitor`) détecte un drift.

---

## 1. Détecter — quels seuils ?

Le rapport Evidently donne, pour chaque feature, un score de drift (test statistique adapté au type de variable : Wasserstein pour les numériques, χ² pour les catégorielles).

| Niveau de drift | Critère | Action |
|---|---|---|
| 🟢 Aucun / faible | < 30 % des features driftées | Ne rien faire. Continuer à monitorer. |
| 🟡 Modéré | 30–50 % des features driftées **OU** une feature critique driftée (`MedInc`, `Latitude`, `Longitude`) | Investiguer + réentraîner sur fenêtre récente. |
| 🔴 Significatif | > 50 % des features driftées **OU** baisse mesurée du R² > 5 points sur jeu de validation récent | Réentraîner d'urgence + revue manuelle avant déploiement. |

> Les seuils sont indicatifs et doivent être calibrés sur l'historique réel du projet.

---

## 2. Réentraîner — quelle stratégie ?

### a) Pipeline complet (recommandé pour `ImmoPrix`)
1. Récupérer **toutes les données disponibles** (entraînement initial + nouvelles données de production de la fenêtre glissante, ex. 90 jours).
2. Relancer `make prepare` puis `make train` — le nouveau modèle est loggé dans MLflow sous un nouveau `run_id` mais le même `registered_model_name`.
3. Comparer les métriques (RMSE, MAE, R²) du nouveau modèle vs le modèle en production.

### b) Approche **champion / challenger**
- Le modèle actuel = **champion** (sert en prod).
- Le modèle réentraîné = **challenger** : il est promu à `stage="Staging"` dans MLflow Registry.
- Validation manuelle ou A/B test → promotion à `Production` si OK.
- Sinon, rollback en pointant l'API sur la précédente version (`models:/ImmoPrix_rf/<version_number>`).

### c) Fine-tuning vs réentraînement complet
- Pour un RandomForest/GBM, le fine-tuning n'est pas standard → on **réentraîne from scratch** sur la fenêtre élargie.
- Pour un modèle linéaire ou un réseau de neurones, on pourrait envisager un re-fit partiel.

---

## 3. Automatiser — quel déclencheur ?

| Trigger | Avantages | Inconvénients |
|---|---|---|
| **Périodique** (cron : `make monitor` hebdo) | Simple, prévisible | Peut rater un drift soudain |
| **Sur seuil** (workflow GitHub Actions qui parse le rapport Evidently) | Réactif, économe | Plus complexe à mettre en place |
| **Mixte** (cron + alerte sur seuil) | Le plus robuste | Surcoût d'infra |

Reco pour ce projet étudiant :
- Cron quotidien GitHub Actions qui lance `make monitor` et upload le HTML en artifact.
- Si on parse le rapport et qu'un seuil est franchi, ouvrir automatiquement une **issue GitHub** avec le détail des features driftées.

---

## 4. Boucle complète

```
┌──────────────┐     drift?     ┌──────────────┐     metrics OK?    ┌─────────────┐
│  monitor.py  │ ─────yes────►  │ prepare+train│ ──────yes────►     │ promote in  │
│  (Evidently) │                │ + new model  │                    │  registry   │
└──────────────┘                └──────────────┘                    └─────────────┘
       ▲                                                                    │
       │                                                                    │
       └────────────────────────── serve via API ───────────────────────────┘
```

---

## 5. À surveiller au-delà du data drift

Le drift sur les **features** n'est qu'une partie de l'histoire. Idéalement on suit aussi :
- **Concept drift** : la relation X → y change (les prix ne dépendent plus du revenu de la même façon).
- **Model performance drift** : R² / MAE mesurés sur des transactions réelles vs prédictions.
- **Prediction drift** : distribution des prédictions du modèle dans le temps.

Evidently propose des presets pour chacun (`TargetDriftPreset`, `RegressionPreset`).
