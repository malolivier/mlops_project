import mlflow
import mlflow.sklearn
import numpy as np
import duckdb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tracking store relatif au CWD (= racine projet via `make train`), portable entre collaborateurs.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_data_from_duckdb(db_path="data/processed/housing.duckdb"):
    """Charge les données préparées depuis la base DuckDB."""
    # Connexion à DuckDB (en mode lecture seule pour éviter les verrous)
    conn = duckdb.connect(db_path, read_only=True)
    
    # On récupère les données directement dans un DataFrame Pandas
    df = conn.execute("SELECT * FROM california_housing").df()
    conn.close()
    
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    
    # Découpage Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_log_model(model_name="baseline", **kwargs):
    """Entraîne un modèle et enregistre les hyperparamètres et métriques dans MLflow."""
    X_train, X_test, y_train, y_test = load_data_from_duckdb()

    # Définition de l'expérience globale du projet
    mlflow.set_experiment("ImmoPrix_California")

    # Démarrage d'une itération (Run)
    with mlflow.start_run(run_name=f"Run_{model_name}"):
        
        # 1. Sélection et configuration du modèle
        if model_name == "baseline":
            model = LinearRegression()
            mlflow.log_param("model_type", "LinearRegression")
            
        elif model_name == "rf":
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", None)
            
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
        elif model_name == "gradient_boosting":
            n_estimators = kwargs.get("n_estimators", 100)
            learning_rate = kwargs.get("learning_rate", 0.1)
            
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            
        else:
            raise ValueError(f"Modèle '{model_name}' non supporté.")

        # 2. Entraînement
        model.fit(X_train, y_train)

        # 3. Évaluation
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # 4. Logging dans MLflow (Métriques)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # 5. Enregistrement du modèle (Artifacts + Model Registry)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"ImmoPrix_{model_name}"
        )

        print(f"[{model_name}] R²: {r2:.4f} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    print("--- 1. Entraînement de la Baseline ---")
    train_and_log_model(model_name="baseline")

    print("\n--- 2. Entraînement du premier modèle complexe (Random Forest) ---")
    train_and_log_model(model_name="rf", n_estimators=100, max_depth=15)

    print("\n--- 3. Entraînement du second modèle complexe (Gradient Boosting) ---")
    train_and_log_model(model_name="gradient_boosting", n_estimators=150, learning_rate=0.05)