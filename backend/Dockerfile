# Imagen base: Python 3.13 "slim" — sistema mínimo, sin herramientas de más.
FROM python:3.13-slim

WORKDIR /app

# NOTA sobre las rutas de este archivo: el "contexto" de build es la RAÍZ
# del repo (lo fijamos en docker-compose.yml con "context: ."), no esta
# carpeta backend/. Por eso todas las rutas de COPY empiezan con "backend/"
# o "models/" — son relativas a la raíz, no a donde vive este Dockerfile.

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py backend/schemas.py backend/model_service.py ./

# Copiamos SOLO los artefactos que están en git (ver .gitignore) — así el
# contenedor local se construye con exactamente lo mismo que Render tendrá
# cuando clone el repo desde GitHub.
COPY models/random_forest.pkl models/scaler.pkl models/cols_to_scale.pkl models/cineai.db ./models/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]