"""FastAPI service exposing the trained California Housing model.

Lance via `make api` (uvicorn) ou directement :
    uv run uvicorn mlops.api:app --reload
"""
from contextlib import asynccontextmanager

import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/ImmoPrix_rf/latest"
DB_PATH = "data/processed/housing.duckdb"
TABLE_NAME = "california_housing"
TARGET = "MedHouseVal"
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

state: dict = {"model": None, "scaler": None}


def _fit_scaler_like_training() -> StandardScaler:
    # train.py n'enregistre pas le scaler avec le modèle (pas de sklearn.Pipeline).
    # On le re-fit ici avec exactement le même split (random_state=42, test_size=0.2)
    # pour éviter le train/serve skew. À refacto en Pipeline côté train.py à terme.
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute(f"SELECT * FROM {TABLE_NAME}").df()
    conn.close()
    X = df[FEATURE_NAMES]
    y = df[TARGET]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    state["scaler"] = _fit_scaler_like_training()
    yield
    state.clear()


app = FastAPI(
    title="ImmoPrix API",
    description="Prédiction du prix médian des maisons en Californie",
    version="1.0.0",
    lifespan=lifespan,
)


class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Revenu médian (10K$)", examples=[3.5])
    HouseAge: float = Field(..., ge=0, le=200, description="Âge médian", examples=[20])
    AveRooms: float = Field(..., gt=0, description="Pièces par logement", examples=[5.0])
    AveBedrms: float = Field(..., gt=0, description="Chambres par logement", examples=[1.0])
    Population: float = Field(..., ge=0, description="Population du secteur", examples=[1000])
    AveOccup: float = Field(..., gt=0, description="Occupation moyenne", examples=[3.0])
    Latitude: float = Field(..., ge=32, le=42, description="Latitude", examples=[34.0])
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude", examples=[-118.0])


class PredictionResponse(BaseModel):
    predicted_price_100k_usd: float = Field(..., description="Prix médian prédit en 100K$")
    model_uri: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures) -> PredictionResponse:
    if state["model"] is None or state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    X = pd.DataFrame([features.model_dump()])[FEATURE_NAMES]
    X_scaled = state["scaler"].transform(X)
    y_pred = float(state["model"].predict(X_scaled)[0])
    return PredictionResponse(predicted_price_100k_usd=y_pred, model_uri=MODEL_URI)
