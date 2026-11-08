"""
Phase 4 - API FastAPI pour servir le modèle de sentiment
"""

import time
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"

LABEL_MAP = {-1: "négatif", 0: "neutre", 1: "positif"}

app = FastAPI(
    title="YouTube Sentiment API",
    description="Analyse de sentiment des commentaires YouTube",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # extension Chrome depuis n'importe quelle origine
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Modèle non trouvé. Lancez src/models/train.py d'abord.")
        _model = joblib.load(MODEL_PATH)
        # Patch sklearn <1.5 models that stored multi_class as an instance attribute
        clf = _model.named_steps.get("clf")
        if clf is not None and not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"
    return _model


# --- Schémas Pydantic ---

class PredictRequest(BaseModel):
    comments: list[str]

    @field_validator("comments")
    @classmethod
    def not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("La liste de commentaires ne peut pas être vide")
        if len(v) > 500:
            raise ValueError("Maximum 500 commentaires par requête")
        return v


class CommentResult(BaseModel):
    comment: str
    label: int
    sentiment: str
    confidence: float


class PredictResponse(BaseModel):
    results: list[CommentResult]
    stats: dict
    inference_ms: float


# --- Endpoints ---

@app.get("/health")
def health():
    try:
        model = get_model()
        return {"status": "ok", "model_loaded": model is not None}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(request: PredictRequest):
    model = get_model()

    start = time.perf_counter()
    labels = model.predict(request.comments).tolist()
    probas = model.predict_proba(request.comments).tolist()
    elapsed_ms = (time.perf_counter() - start) * 1000

    classes = model.classes_.tolist()
    results = []
    for comment, label, proba in zip(request.comments, labels, probas):
        idx = classes.index(label)
        results.append(CommentResult(
            comment=comment,
            label=int(label),
            sentiment=LABEL_MAP.get(int(label), str(label)),
            confidence=round(proba[idx], 4),
        ))

    counts = {LABEL_MAP.get(int(l), str(l)): labels.count(l) for l in set(labels)}
    total = len(labels)
    stats = {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in counts.items()}

    return PredictResponse(results=results, stats=stats, inference_ms=round(elapsed_ms, 2))
