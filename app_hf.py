"""
App Gradio pour Hugging Face Spaces (SDK gratuit)
Expose le modèle via l'API Gradio + interface visuelle
"""

import json
import time
import joblib
from pathlib import Path
import gradio as gr

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.joblib"
LABEL_MAP = {-1: "négatif", 0: "neutre", 1: "positif"}

model = joblib.load(MODEL_PATH)


def predict_batch(comments_json: str) -> str:
    """
    Reçoit un JSON array de commentaires, retourne les prédictions JSON.
    Compatible avec l'extension Chrome.
    """
    try:
        comments = json.loads(comments_json)
        if not isinstance(comments, list) or len(comments) == 0:
            return json.dumps({"error": "Liste vide ou format invalide"})
        if len(comments) > 500:
            return json.dumps({"error": "Maximum 500 commentaires"})
    except json.JSONDecodeError:
        return json.dumps({"error": "JSON invalide"})

    start = time.perf_counter()
    labels = model.predict(comments).tolist()
    probas = model.predict_proba(comments).tolist()
    elapsed_ms = (time.perf_counter() - start) * 1000

    classes = model.classes_.tolist()
    results = []
    for comment, label, proba in zip(comments, labels, probas):
        idx = classes.index(label)
        results.append({
            "comment": comment,
            "label": int(label),
            "sentiment": LABEL_MAP.get(int(label), str(label)),
            "confidence": round(proba[idx], 4),
        })

    counts: dict[str, int] = {}
    for lbl in labels:
        key = LABEL_MAP.get(int(lbl), str(lbl))
        counts[key] = counts.get(key, 0) + 1
    total = len(labels)
    stats = {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in counts.items()}

    return json.dumps({
        "results": results,
        "stats": stats,
        "inference_ms": round(elapsed_ms, 2),
    }, ensure_ascii=False)


def predict_single(text: str) -> str:
    """Interface visuelle pour tester un commentaire."""
    if not text.strip():
        return "Entrez un commentaire."
    result = json.loads(predict_batch(json.dumps([text])))
    r = result["results"][0]
    emoji = {"positif": "😊", "neutre": "😐", "négatif": "😠"}.get(r["sentiment"], "🤔")
    return f"{emoji} **{r['sentiment'].upper()}** (confiance : {r['confidence']*100:.1f}%)"


with gr.Blocks(title="YouTube Sentiment Analyzer") as demo:
    gr.Markdown("# 🎬 YouTube Sentiment Analyzer\nAnalyse de sentiment des commentaires YouTube · TF-IDF + Logistic Regression")

    with gr.Tab("Test rapide"):
        txt_in = gr.Textbox(label="Commentaire", placeholder="Super vidéo !")
        txt_out = gr.Markdown()
        gr.Button("Analyser").click(predict_single, txt_in, txt_out)

    with gr.Tab("API batch (extension Chrome)"):
        gr.Markdown("""**Endpoint utilisé par l'extension Chrome :**
```
POST /api/predict
{"data": ["[\"comment1\", \"comment2\"]"]}
```""")
        batch_in = gr.Textbox(label='JSON array de commentaires', value='["Super vidéo!", "Nul ce contenu", "Bof"]')
        batch_out = gr.Textbox(label="Résultat JSON", lines=10)
        gr.Button("Tester").click(predict_batch, batch_in, batch_out)

demo.launch()
