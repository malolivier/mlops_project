import os
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def run_shap_analysis(output_dir="reports/figures"):
    """Calcule et sauvegarde les explications SHAP globales et locales."""
    print("--- Démarrage de l'analyse SHAP ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chargement des données depuis DuckDB
    conn = duckdb.connect("data/processed/housing.duckdb", read_only=True)
    df = conn.execute("SELECT * FROM california_housing").df()
    conn.close()

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraînement du modèle Champion (Random Forest)
    # Note MLOps : On utilise les données brutes non-scalées car le Random Forest
    # y est insensible, et cela rend les graphiques SHAP beaucoup plus lisibles pour les humains !
    print("Entraînement du modèle Random Forest Champion...")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42
    )
    model.fit(X_train, y_train)

    # 2. Calcul des valeurs SHAP
    print(
        "Calcul des valeurs SHAP (sur un échantillon de 100 maisons pour aller vite)..."
    )
    X_test_sample = X_test.head(100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample)

    # 3. Analyse Globale (Summary Plot)
    print("Génération du graphique global (Summary Plot)...")
    shap.summary_plot(shap_values, X_test_sample, show=False)
    summary_path = os.path.join(output_dir, "shap_global_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.clf()  # Nettoie la figure courante
    print(f"Graphique global sauvegardé dans : {summary_path}")

    # 4. Analyse Locale (Waterfall Plot pour la première maison du test set)
    print("Génération du graphique local (Waterfall Plot)...")
    shap.plots.waterfall(shap_values[0], show=False)
    local_path = os.path.join(output_dir, "shap_local_waterfall.png")
    plt.savefig(local_path, bbox_inches="tight")
    plt.clf()
    print(f"Graphique local sauvegardé dans : {local_path}")

    print("--- Analyse SHAP terminée avec succès ! ---")


if __name__ == "__main__":
    run_shap_analysis()