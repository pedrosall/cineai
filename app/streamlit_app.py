"""
CineAI — Movie Success Predictor
Streamlit FRONTEND. No carga el modelo ni calcula nada: solo pinta el
formulario, llama a la API con `requests`, y muestra lo que la API responde.
"""

import streamlit as st
import requests
import os

# ---------------------------------------------------------------------------
# Configuración — URL de la API leída de una variable de entorno.
# En local (sin definir nada) apunta a localhost:8000.
# En Streamlit Cloud, se define en "Settings -> Secrets" (lo vemos en Fase 4).
# En Docker Compose, se define en docker-compose.yml (lo vemos en Fase 3).
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineAI — Movie Success Predictor",
    page_icon="🎬",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Custom styling — dark cinema theme (sin cambios respecto al original)
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E6E6E6; }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div { color: #E6E6E6; }
    h1, h2, h3 { font-family: 'Georgia', serif; color: #F5C518; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #B0B0B0 !important; }
    [data-testid="stWidgetLabel"] p { color: #E6E6E6 !important; font-weight: 500; }
    .stButton button {
        background-color: #F5C518; color: #0E1117; font-weight: 600;
        border-radius: 6px; border: none; padding: 0.6rem 1.5rem;
    }
    .stButton button:hover { background-color: #D4A917; }
    .result-card { padding: 1.5rem; border-radius: 10px; margin-top: 1rem; text-align: center; }
    .result-card h2, .result-card p { color: #FFFFFF !important; }
    .success-card { background-color: rgba(46, 204, 113, 0.15); border: 1px solid #2ecc71; }
    .failure-card { background-color: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; }
    .feature-bar-positive { background-color: #2ecc71; }
    .feature-bar-negative { background-color: #e74c3c; }
    </style>
""", unsafe_allow_html=True)

GENRES = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
          'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
          'romance', 'science_fiction', 'thriller', 'war', 'western']

SEASON_MAP = {
    "Summer (Jun-Aug)": "summer",
    "Christmas (Nov-Dec)": "christmas",
    "Spring (Mar-May)": "spring",
    "Off-season (rest of year)": "off_season",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎬 CineAI")
st.caption("Predict whether your movie idea would be a commercial success — before it's made.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Input form (idéntico al original — el usuario no nota diferencia visual)
# ---------------------------------------------------------------------------
with st.form("movie_form"):
    st.subheader("Movie details")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget (USD)", min_value=100_000, max_value=400_000_000,
                                  value=50_000_000, step=1_000_000)
        runtime = st.slider("Runtime (minutes)", min_value=60, max_value=240, value=110)
    with col2:
        release_year = st.number_input("Release year", min_value=1980, max_value=2030, value=2026)
        season_label = st.selectbox("Release season", list(SEASON_MAP.keys()))

    st.markdown("**Genres** (select all that apply)")
    selected_genres = st.multiselect("Genres", [g.replace('_', ' ').title() for g in GENRES],
                                      default=["Action"], label_visibility="collapsed")

    st.markdown("**Talent**")
    col3, col4 = st.columns(2)
    with col3:
        director_name = st.text_input("Director", placeholder="e.g. Christopher Nolan")
    with col4:
        company_name = st.text_input("Production company", placeholder="e.g. Warner Bros.")

    cast_input = st.text_input("Top 3 cast members (comma-separated)",
                                placeholder="e.g. Tom Hanks, Emma Stone, Idris Elba")

    submitted = st.form_submit_button("Predict success 🎯")

# ---------------------------------------------------------------------------
# Llamada a la API
# ---------------------------------------------------------------------------
def call_predict_api(payload: dict) -> dict:
    """Llama a POST /predict. Lanza excepción si la API no responde o da error."""
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()  # lanza excepción si el código no es 2xx
    return response.json()


if submitted:
    # Construimos el JSON exactamente con la forma que espera MovieInput en el backend.
    # Esta es la parte más delicada: el "contrato" tiene que coincidir en ambos lados.
    payload = {
        "budget": budget,
        "runtime": runtime,
        "release_year": release_year,
        "season": SEASON_MAP[season_label],
        "genres": [g.lower().replace(' ', '_') for g in selected_genres],
        "director": director_name or None,
        "production_company": company_name or None,
        "cast": [n.strip() for n in cast_input.split(",") if n.strip()][:3],
    }

    try:
        with st.spinner("Consultando el modelo..."):
            result = call_predict_api(payload)
    except requests.exceptions.ConnectionError:
        st.error(
            f"⚠️ No se pudo conectar con la API en `{API_URL}`. "
            "¿Está el backend corriendo? (`uvicorn main:app` en la carpeta backend/)"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ La API devolvió un error: {e.response.status_code} — {e.response.text}")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⚠️ La API tardó demasiado en responder (timeout).")
        st.stop()

    proba = result["success_probability"]
    prediction = "Success" if proba >= 0.5 else "Failure"

    st.markdown("---")
    st.subheader("Result")

    card_class = "success-card" if prediction == "Success" else "failure-card"
    emoji = "🟢" if prediction == "Success" else "🔴"

    st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{emoji} Predicted: {prediction}</h2>
            <p style="font-size: 1.3rem;">Success probability: <b>{proba:.1%}</b></p>
        </div>
    """, unsafe_allow_html=True)

    # unknown_entities ahora la calcula y la manda el backend, el frontend solo la pinta.
    for entity in result.get("unknown_entities", []):
        st.caption(f"ℹ️ '{entity}' not found in historical data — used dataset average.")

    # ------------------------------------------------------------------
    # Explicación: ya no dibujamos el waterfall completo de SHAP (eso
    # requeriría mandar TODAS las features desde el backend). Nuestra
    # API, por decisión de diseño, solo manda el top 3 — así que pintamos
    # un gráfico de barras horizontal simple con esas 3.
    # ------------------------------------------------------------------
    st.markdown("### Why this prediction?")
    top_features = result["top_features"]

    for feat in top_features:
        name = feat["feature"].replace("_", " ").title()
        value = feat["value"]
        direction = "pushed toward success" if value > 0 else "pushed toward failure"
        bar_class = "feature-bar-positive" if value > 0 else "feature-bar-negative"
        width_pct = min(abs(value) * 400, 100)  # escala visual, no un valor exacto
        st.markdown(f"""
            <div style="margin-bottom: 0.6rem;">
                <span>{name}</span> — <i>{direction}</i> ({value:+.3f})
                <div style="background-color:#333; border-radius:4px; height:10px; margin-top:2px;">
                    <div class="{bar_class}" style="width:{width_pct}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.caption(
        "These are the 3 factors that most influenced this specific prediction, "
        "computed with SHAP on the backend. Positive pushes toward success, negative toward failure."
    )

st.markdown("---")
st.caption("Built with a Random Forest classifier (ROC-AUC 0.768) trained on the TMDB 5000 Movie Dataset. "
           "Predictions are based only on pre-release information — they cannot account for execution quality.")