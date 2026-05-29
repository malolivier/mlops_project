"""Streamlit UI pour tester l'API ImmoPrix.

Lance via `make ui` ou directement :
    uv run streamlit run src/mlops/streamlit_app.py

Prérequis : l'API doit tourner (par défaut sur http://127.0.0.1:8000).
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("IMMOPRIX_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ImmoPrix — Prédiction de prix", page_icon="🏠", layout="centered")

st.title("🏠 ImmoPrix — Estimation du prix médian")
st.caption("Entrez les caractéristiques d'un secteur de Californie pour obtenir une estimation.")

# --- Indicateur de santé de l'API ---
with st.sidebar:
    st.header("API")
    st.code(API_URL, language="text")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("model_loaded"):
            st.success(f"✅ Modèle chargé\n\n`{health.get('model_uri', '?')}`")
        else:
            st.warning("⚠️  API joignable mais modèle non chargé")
    except requests.RequestException as e:
        st.error(f"❌ API injoignable\n\n{e}")

# --- Formulaire ---
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.number_input("MedInc (revenu médian, 10K$)", min_value=0.0, value=3.5, step=0.1)
        house_age = st.number_input("HouseAge (âge médian)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
        ave_rooms = st.number_input("AveRooms (pièces / logement)", min_value=0.1, value=5.0, step=0.1)
        ave_bedrms = st.number_input("AveBedrms (chambres / logement)", min_value=0.1, value=1.0, step=0.1)
    with col2:
        population = st.number_input("Population du secteur", min_value=0.0, value=1000.0, step=100.0)
        ave_occup = st.number_input("AveOccup (occupation moyenne)", min_value=0.1, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01, format="%.2f")
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Prédire le prix", type="primary", use_container_width=True)

# --- Appel API ---
if submitted:
    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        price_100k = result["predicted_price_100k_usd"]
        st.metric(
            label="Prix médian prédit",
            value=f"{price_100k * 100_000:,.0f} $".replace(",", " "),
            help=f"Soit {price_100k:.3f} (en 100K$, unité du modèle)",
        )
        with st.expander("Réponse brute"):
            st.json(result)
    except requests.HTTPError as e:
        st.error(f"Erreur API : {e.response.status_code}\n\n{e.response.text}")
    except requests.RequestException as e:
        st.error(f"Erreur réseau : {e}")
