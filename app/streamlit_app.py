"""
CineAI — Movie Success Predictor
Streamlit app for predicting movie commercial success using a trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineAI — Movie Success Predictor",
    page_icon="🎬",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Custom styling — dark cinema theme
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E6E6E6; }

    /* Texto general — párrafos, labels, captions, markdown */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #E6E6E6;
    }

    h1, h2, h3 { font-family: 'Georgia', serif; color: #F5C518; }

    /* Captions (texto pequeño gris por defecto en Streamlit) */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #B0B0B0 !important;
    }

    /* Labels de los inputs del formulario */
    [data-testid="stWidgetLabel"] p {
        color: #E6E6E6 !important;
        font-weight: 500;
    }

    .stButton button {
        background-color: #F5C518;
        color: #0E1117;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.5rem;
    }
    .stButton button:hover { background-color: #D4A917; }

    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .result-card h2, .result-card p { color: #FFFFFF !important; }

    .success-card { background-color: rgba(46, 204, 113, 0.15); border: 1px solid #2ecc71; }
    .failure-card { background-color: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model artifacts (cached so it only loads once per session)
# ---------------------------------------------------------------------------
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(os.path.join(MODELS, 'random_forest.pkl'))
    scaler = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
    cols_to_scale = joblib.load(os.path.join(MODELS, 'cols_to_scale.pkl'))
    director_map = joblib.load(os.path.join(MODELS, 'director_map.pkl'))
    company_map = joblib.load(os.path.join(MODELS, 'company_map.pkl'))
    cast_map = joblib.load(os.path.join(MODELS, 'cast_map.pkl'))
    global_rate = joblib.load(os.path.join(MODELS, 'global_rate.pkl'))
    explainer = shap.TreeExplainer(rf_model)
    return rf_model, scaler, cols_to_scale, director_map, company_map, cast_map, global_rate, explainer

rf_model, scaler, cols_to_scale, director_map, company_map, cast_map, global_rate, explainer = load_artifacts()

GENRES = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
          'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
          'romance', 'science_fiction', 'thriller', 'war', 'western']

FEATURE_ORDER = (
    ['budget', 'runtime', 'release_year', 'director_success_rate',
     'cast_success_rate', 'company_success_rate'] +
    [f'genre_{g}' for g in GENRES] +
    ['season_christmas', 'season_off_season', 'season_spring', 'season_summer']
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎬 CineAI")
st.caption("Predict whether your movie idea would be a commercial success — before it's made.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Input form
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
        season = st.selectbox("Release season",
                               ["Summer (Jun-Aug)", "Christmas (Nov-Dec)",
                                "Spring (Mar-May)", "Off-season (rest of year)"])

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
# Helper functions
# ---------------------------------------------------------------------------
def get_rate(name, mapping, global_rate):
    """Looks up an entity's historical rate, falling back to global rate if unknown."""
    if not name or name.strip() == "":
        return global_rate
    return mapping.get(name.strip(), global_rate)

def get_cast_rate(cast_str, mapping, global_rate):
    """Averages the historical rate of up to 3 cast members."""
    if not cast_str or cast_str.strip() == "":
        return global_rate
    names = [n.strip() for n in cast_str.split(",") if n.strip()][:3]
    if not names:
        return global_rate
    rates = [mapping.get(n, global_rate) for n in names]
    return np.mean(rates)

def build_feature_vector(budget, runtime, release_year, season, genres,
                          director_rate, cast_rate, company_rate):
    row = {col: 0 for col in FEATURE_ORDER}
    row['budget'] = budget
    row['runtime'] = runtime
    row['release_year'] = release_year
    row['director_success_rate'] = director_rate
    row['cast_success_rate'] = cast_rate
    row['company_success_rate'] = company_rate

    for g in genres:
        key = f"genre_{g.lower().replace(' ', '_')}"
        if key in row:
            row[key] = 1

    season_map = {
        "Summer (Jun-Aug)": "season_summer",
        "Christmas (Nov-Dec)": "season_christmas",
        "Spring (Mar-May)": "season_spring",
        "Off-season (rest of year)": "season_off_season"
    }
    row[season_map[season]] = 1

    return pd.DataFrame([row])[FEATURE_ORDER]

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    director_rate = get_rate(director_name, director_map, global_rate)
    company_rate = get_rate(company_name, company_map, global_rate)
    cast_rate = get_cast_rate(cast_input, cast_map, global_rate)

    X_input = build_feature_vector(
        budget, runtime, release_year, season, selected_genres,
        director_rate, cast_rate, company_rate
    )

    X_scaled = X_input.copy()
    X_scaled[cols_to_scale] = scaler.transform(X_input[cols_to_scale])

    proba = rf_model.predict_proba(X_scaled)[0, 1]
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

    if director_name and director_name.strip() not in director_map:
        st.caption(f"ℹ️ '{director_name}' not found in historical data — used dataset average ({global_rate:.1%}).")
    if company_name and company_name.strip() not in company_map:
        st.caption(f"ℹ️ '{company_name}' not found in historical data — used dataset average ({global_rate:.1%}).")

    # SHAP explanation
    st.markdown("### Why this prediction?")
    shap_values = explainer.shap_values(X_scaled)
    shap_values_success = shap_values[:, :, 1] if len(np.array(shap_values).shape) == 3 else shap_values

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_success[0],
            base_values=explainer.expected_value[1],
            data=X_scaled.iloc[0].values,
            feature_names=X_scaled.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "This chart shows how each factor pushed the prediction up (toward success) "
        "or down (toward failure), starting from the model's average prediction."
    )

st.markdown("---")
st.caption("Built with a Random Forest classifier (ROC-AUC 0.768) trained on the TMDB 5000 Movie Dataset. "
           "Predictions are based only on pre-release information — they cannot account for execution quality.")