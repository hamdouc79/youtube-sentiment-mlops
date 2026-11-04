"""
Phase 2 - Script de téléchargement du dataset Reddit Sentiment
"""

import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"


def download_dataset() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "reddit.csv"

    print(f"Téléchargement depuis : {DATA_URL}")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Sauvegardé : {output_path}")

    df = pd.read_csv(output_path)
    return df


def print_stats(df: pd.DataFrame) -> None:
    print("\n=== Statistiques du dataset ===")
    print(f"Nombre total de commentaires : {len(df)}")
    print(f"Colonnes : {list(df.columns)}")
    print(f"\nDistribution des labels :")
    if "category" in df.columns:
        print(df["category"].value_counts())
    elif "label" in df.columns:
        print(df["label"].value_counts())
    print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
    print(f"\nAperçu :\n{df.head(3)}")


if __name__ == "__main__":
    df = download_dataset()
    print_stats(df)
