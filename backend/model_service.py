"""
model_service.py — Carga los artefactos del modelo UNA VEZ y expone la
lógica de predicción + explicación SHAP.

Por qué una clase y no funciones sueltas con variables globales:
- Facilita testear (puedes crear una instancia con artefactos de mentira).
- Deja claro, con `__init__`, qué necesita este servicio para funcionar.
- FastAPI la instancia una sola vez al arrancar (ver lifespan en main.py)
  y la reutiliza en cada request — así el modelo NUNCA se recarga por petición.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import joblib
import shap

GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "science_fiction", "thriller", "war", "western",
]

FEATURE_ORDER = (
    ["budget", "runtime", "release_year", "director_success_rate",
     "cast_success_rate", "company_success_rate"]
    + [f"genre_{g}" for g in GENRES]
    + ["season_christmas", "season_off_season", "season_spring", "season_summer"]
)

VALID_SEASONS = {"summer", "christmas", "spring", "off_season"}


class ModelService:
    def __init__(self, models_dir: str):
        self.rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.cols_to_scale = joblib.load(os.path.join(models_dir, "cols_to_scale.pkl"))

        # Tasas históricas: ya no viven en .pkl, viven en SQLite (ver scripts/build_db.py).
        # check_same_thread=False porque FastAPI puede atender requests desde
        # threads distintos; como esta base es de solo lectura, es seguro.
        db_path = os.path.join(models_dir, "cineai.db")
        self.db = sqlite3.connect(db_path, check_same_thread=False)

        row = self.db.execute("SELECT value FROM meta WHERE key = 'global_rate'").fetchone()
        self.global_rate = float(row[0])

        # El TreeExplainer inspecciona la estructura de los árboles UNA vez;
        # calcular esto por request sería carísimo (por eso vive aquí, no en /predict).
        self.explainer = shap.TreeExplainer(self.rf_model)

    # -- Lookups con fallback a la media global -----------------------------
    def _query_rate(self, table: str, name: str):
        row = self.db.execute(
            f"SELECT success_rate FROM {table} WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def _rate_for(self, name, table: str):
        if not name or not name.strip():
            return self.global_rate, False
        name = name.strip()
        rate = self._query_rate(table, name)
        if rate is not None:
            return rate, False
        return self.global_rate, True  # True = "era desconocido"

    def _cast_rate(self, names):
        names = [n.strip() for n in names if n and n.strip()][:3]
        if not names:
            return self.global_rate, []
        unknown = []
        rates = []
        for n in names:
            rate = self._query_rate("cast_members", n)
            if rate is None:
                unknown.append(n)
                rates.append(self.global_rate)
            else:
                rates.append(rate)
        return float(np.mean(rates)), unknown

    # -- Construcción del vector de features ---------------------------------
    def _build_feature_vector(self, movie) -> pd.DataFrame:
        if movie.season not in VALID_SEASONS:
            raise ValueError(
                f"season debe ser una de {VALID_SEASONS}, recibido: '{movie.season}'"
            )

        director_rate, director_unknown = self._rate_for(movie.director, "directors")
        company_rate, company_unknown = self._rate_for(movie.production_company, "companies")
        cast_rate, cast_unknown = self._cast_rate(movie.cast)

        row = {col: 0 for col in FEATURE_ORDER}
        row["budget"] = movie.budget
        row["runtime"] = movie.runtime
        row["release_year"] = movie.release_year
        row["director_success_rate"] = director_rate
        row["cast_success_rate"] = cast_rate
        row["company_success_rate"] = company_rate

        for g in movie.genres:
            key = f"genre_{g.lower().strip().replace(' ', '_')}"
            if key in row:
                row[key] = 1

        row[f"season_{movie.season}"] = 1

        unknown_entities = []
        if director_unknown and movie.director:
            unknown_entities.append(movie.director.strip())
        if company_unknown and movie.production_company:
            unknown_entities.append(movie.production_company.strip())
        unknown_entities.extend(cast_unknown)

        X = pd.DataFrame([row])[FEATURE_ORDER]
        return X, unknown_entities

    # -- Predicción + SHAP -----------------------------------------------------
    def predict(self, movie, top_k: int = 3):
        X, unknown_entities = self._build_feature_vector(movie)

        X_scaled = X.copy()
        X_scaled[self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])

        proba = float(self.rf_model.predict_proba(X_scaled)[0, 1])

        shap_values = self.explainer.shap_values(X_scaled)
        # En sklearn>=1.x con clasificación binaria, shap_values puede venir
        # como array 3D (samples, features, clases) o como lista [clase0, clase1].
        if isinstance(shap_values, list):
            shap_row = shap_values[1][0]
        elif np.array(shap_values).ndim == 3:
            shap_row = shap_values[:, :, 1][0]
        else:
            shap_row = shap_values[0]

        contributions = list(zip(FEATURE_ORDER, shap_row))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
        top_features = [
            {"feature": name, "value": float(val)} for name, val in contributions[:top_k]
        ]

        return proba, top_features, unknown_entities