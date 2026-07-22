# Este Dockerfile SOLO se usa para desarrollo local con docker-compose.
# Streamlit Cloud (donde desplegaremos de verdad el frontend) no lo lee:
# instala directamente desde requirements.txt con su propio sistema.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/streamlit_app.py ./app/streamlit_app.py

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
