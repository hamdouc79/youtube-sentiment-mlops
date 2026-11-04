"""
Phase 2 - Nettoyage et préparation des données
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                  # mentions
    text = re.sub(r"r/\w+", "", text)                 # subreddits
    text = re.sub(r"[^a-z\s]", " ", text)             # caractères spéciaux (garde emojis via espaces)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Colonnes brutes : {list(df.columns)}")

    # Normalise les noms de colonnes selon ce que contient le CSV Reddit
    # Le dataset a 'clean_comment' et 'category'
    col_map = {}
    for col in df.columns:
        if col.lower() in ("clean_comment", "comment", "text", "body"):
            col_map[col] = "text"
        elif col.lower() in ("category", "label", "sentiment"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    # Garde uniquement les colonnes utiles
    df = df[["text", "label"]].copy()
    return df


def prepare_dataset(input_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_path is None:
        input_path = RAW_DIR / "reddit.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize(input_path)

    print(f"\nAvant nettoyage : {len(df)} lignes")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 5]  # supprime les textes vides/trop courts
    print(f"Après nettoyage : {len(df)} lignes")

    # Distribution des classes
    print(f"\nDistribution des labels :\n{df['label'].value_counts()}")

    # Vérification du minimum par classe
    min_per_class = df["label"].value_counts().min()
    if min_per_class < 300:
        print(f"Attention : classe minoritaire a seulement {min_per_class} exemples")

    # Train/test split stratifié (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\nTrain : {len(train_df)} | Test : {len(test_df)}")
    print("Sauvegardé dans data/processed/")

    return train_df, test_df


if __name__ == "__main__":
    prepare_dataset()
