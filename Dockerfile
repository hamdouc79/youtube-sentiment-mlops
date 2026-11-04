FROM python:3.10-slim

WORKDIR /app

# Dépendances de prod uniquement
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Code source et modèle
COPY src/api/app.py src/api/app.py
COPY src/api/__init__.py src/api/__init__.py
COPY src/__init__.py src/__init__.py
COPY models/sentiment_pipeline.joblib models/sentiment_pipeline.joblib

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
