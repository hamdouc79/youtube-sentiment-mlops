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

# Hugging Face Spaces utilise le port 7860
EXPOSE 7860

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
