"""
main.py — Punto de entrada de la API.

Concepto clave: "lifespan".
FastAPI permite registrar código que corre UNA VEZ al arrancar el servidor
(antes de aceptar la primera petición) y UNA VEZ al apagarlo. Usamos esto
para cargar el modelo al arrancar, guardarlo en app.state, y que cada
request lo reutilice — en vez de leerlo de disco en cada llamada.

Analogía: es como encender el horno antes de que lleguen los clientes al
restaurante, no encenderlo por cada plato que se pide.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import MovieInput, PredictionOutput, HealthOutput, ShapContribution
from model_service import ModelService

MODELS_DIR = "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    app.state.model_service = ModelService(models_dir=MODELS_DIR)
    yield
    # --- Shutdown (nada que limpiar aquí, pero el hueco existe para eso) ---


app = FastAPI(
    title="CineAI API",
    description="Predice la probabilidad de éxito comercial de una película en pre-producción.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permite que Streamlit Cloud (otro dominio) pueda llamar a esta API
# desde el navegador. Sin esto, el navegador bloquearía la petición aunque
# la API responda bien — es una protección del propio navegador, no de FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en un proyecto con más en juego, aquí iría la URL exacta de Streamlit Cloud
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOutput)
def health():
    """Usado por Render (y por ti) para comprobar que el servicio está vivo
    y el modelo cargado, sin tener que mandar una predicción completa."""
    loaded = hasattr(app.state, "model_service")
    return HealthOutput(status="ok" if loaded else "loading", model_loaded=loaded)


@app.post("/predict", response_model=PredictionOutput)
def predict(movie: MovieInput):
    service: ModelService = app.state.model_service
    try:
        proba, top_features, unknown_entities = service.predict(movie)
    except ValueError as e:
        # Errores de negocio (p.ej. season inválida) -> 400, no 500.
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionOutput(
        success_probability=proba,
        top_features=[ShapContribution(**f) for f in top_features],
        unknown_entities=unknown_entities,
    )