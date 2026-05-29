"""Configuration pytest partagée.

Mock `fetch_california_housing` avec un dataset synthétique pour éviter
le téléchargement réseau dans le CI (qui peut bloquer plusieurs minutes).
Les distributions respectent les bornes du vrai dataset.
"""
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch


def _build_synthetic_california_housing(n_samples: int = 1000) -> Bunch:
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        "MedInc": rng.uniform(0.5, 15.0, n_samples),
        "HouseAge": rng.uniform(1.0, 52.0, n_samples),
        "AveRooms": rng.uniform(1.0, 15.0, n_samples),
        "AveBedrms": rng.uniform(0.5, 5.0, n_samples),
        "Population": rng.uniform(50.0, 5000.0, n_samples),
        "AveOccup": rng.uniform(1.0, 10.0, n_samples),
        "Latitude": rng.uniform(32.0, 42.0, n_samples),
        "Longitude": rng.uniform(-125.0, -114.0, n_samples),
        "MedHouseVal": rng.uniform(0.5, 6.0, n_samples),
    })
    return Bunch(frame=df)


@pytest.fixture(scope="session", autouse=True)
def mock_fetch_california_housing():
    """Patch global de fetch_california_housing pour toute la session pytest."""
    fake_bunch = _build_synthetic_california_housing()
    with mock.patch("mlops.prepare.fetch_california_housing", return_value=fake_bunch):
        yield
