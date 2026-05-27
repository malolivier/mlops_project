import os
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_shap_analysis(output_dir="reports/figures"):
    """Calcule, sauvegarde en local et enregistre dans MLflow les explications SHAP."""
    print("--- Démarrage de l'analyse SHAP et Logging MLflow ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Configuration de l'expérience MLflow
    mlflow.set_experiment("ImmoPrix_California")

    # 2. Chargement des données depuis DuckDB
    conn = duckdb.connect("data/processed/housing.duckdb", read_only=True)
    df = conn.execute("SELECT * FROM california_housing").df()
    conn.close()

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Réentraînement rapide du Champion (sans scaling pour lisibilité SHAP)
    print("Entraînement du modèle Random Forest Champion...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    print("Calcul des valeurs SHAP...")
    X_test_sample = X_test.head(100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample)

    # On démarre un Run spécifique pour l'explicabilité ou on peut se brancher
    # sur un run existant. Ici, on crée un Run dédié "SHAP_Analysis"
    with mlflow.start_run(run_name="SHAP_Explicability"):
        
        # Log du type de modèle analysé pour s'y retrouver
        mlflow.log_param("analyzed_model", "RandomForest_Champion")

        # 3. Génération et Sauvegarde du Graphique Global
        print("Génération du graphique global (Summary Plot)...")
        shap.summary_plot(shap_values, X_test_sample, show=False)
        summary_path = os.path.join(output_dir, "shap_global_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.clf()

        mlflow.log_artifact(summary_path, artifact_path="shap_plots")
        print(f"Graphique global envoyé à MLflow.")

        # 4. Génération et Sauvegarde du Graphique Local
        print("Génération du graphique local (Waterfall Plot)...")
        shap.plots.waterfall(shap_values[0], show=False)
        local_path = os.path.join(output_dir, "shap_local_waterfall.png")
        plt.savefig(local_path, bbox_inches="tight")
        plt.clf()

        mlflow.log_artifact(local_path, artifact_path="shap_plots")
        print(f"Graphique local envoyé à MLflow.")

    print("--- Analyse SHAP et synchronisation MLflow terminées ! ---")

if __name__ == "__main__":
    run_shap_analysis()