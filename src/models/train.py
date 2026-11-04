"""
Phase 3 - Entraînement du modèle TF-IDF + Logistic Regression
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def train(train_path: Path | None = None, test_path: Path | None = None) -> None:
    if train_path is None:
        train_path = PROCESSED_DIR / "train.csv"
    if test_path is None:
        test_path = PROCESSED_DIR / "test.csv"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df["text"].astype(str), train_df["label"]
    X_test, y_test = test_df["text"].astype(str), test_df["label"]

    # GridSearch sur les hyperparamètres clés
    param_grid = {
        "tfidf__max_features": [30_000, 50_000],
        "clf__C": [0.1, 1.0, 10.0],
    }

    print("GridSearchCV en cours (peut prendre quelques minutes)...")
    grid = GridSearchCV(
        build_pipeline(),
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"\nMeilleurs paramètres : {grid.best_params_}")

    # Évaluation
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"\nRapport de classification :\n{report}")

    # Sauvegarde du modèle complet (pipeline = vectoriseur + classifieur)
    model_path = MODELS_DIR / "sentiment_pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"\nModèle sauvegardé : {model_path}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    plt.title(f"Matrice de confusion (accuracy={acc:.2%})")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(LOGS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()
    print("Matrice de confusion sauvegardée dans logs/confusion_matrix.png")

    # Log des métriques
    with open(LOGS_DIR / "metrics.txt", "w") as f:
        f.write(f"Meilleurs paramètres : {grid.best_params_}\n")
        f.write(f"Accuracy : {acc:.4f}\n\n")
        f.write(report)


if __name__ == "__main__":
    train()
