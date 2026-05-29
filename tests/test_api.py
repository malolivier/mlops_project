"""Tests unitaires pour l'endpoint FastAPI de prédiction.

Le modèle et le scaler sont mockés pour que les tests soient rapides et n'aient
pas besoin d'un MLflow tracking server ni d'une base DuckDB préparée.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


VALID_PAYLOAD = {
    "MedInc": 3.5,
    "HouseAge": 20.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0,
}

PREDICTED_VALUE = 2.345


@pytest.fixture
def client(monkeypatch):
    """Construit un TestClient avec model + scaler mockés."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([PREDICTED_VALUE])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 8))

    monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: mock_model)

    from mlops import api as api_module

    monkeypatch.setattr(api_module, "_fit_scaler_like_training", lambda: mock_scaler)

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_payload(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_price_100k_usd"] == pytest.approx(PREDICTED_VALUE)
    assert data["model_uri"].startswith("models:/")


def test_predict_missing_field_returns_422(client):
    payload = {**VALID_PAYLOAD}
    del payload["MedInc"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_latitude_out_of_bounds_returns_422(client):
    payload = {**VALID_PAYLOAD, "Latitude": 100.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_population_returns_422(client):
    payload = {**VALID_PAYLOAD, "Population": -5.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
