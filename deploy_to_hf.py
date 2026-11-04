"""
Déploiement automatique sur Hugging Face Spaces (Docker)

Usage :
    python deploy_to_hf.py --username <ton_username_hf> --space <nom_du_space>

Exemple :
    python deploy_to_hf.py --username mohamad123 --space youtube-sentiment-api

Prérequis : être loggé via  huggingface-cli login  (ou variable HF_TOKEN)
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
import os

ROOT = Path(__file__).parent

FILES_TO_UPLOAD = [
    # (chemin local,              chemin dans le Space)
    (ROOT / "app_hf.py",                         "app.py"),
    (ROOT / "requirements_hf.txt",               "requirements.txt"),
    (ROOT / "models" / "sentiment_pipeline.joblib", "models/sentiment_pipeline.joblib"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Ton username Hugging Face")
    parser.add_argument("--space", default="youtube-sentiment-api", help="Nom du Space")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space}"

    # Vérifier que le token est disponible
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    # Créer le Space s'il n'existe pas
    print(f"\n→ Création/vérification du Space : {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print("  Space prêt.")
    except Exception as e:
        print(f"  Erreur création Space : {e}")
        sys.exit(1)

    # Upload des fichiers
    print("\n→ Upload des fichiers...")
    for local_path, hf_path in FILES_TO_UPLOAD:
        if not local_path.exists():
            print(f"  MANQUANT : {local_path}  — skipped")
            continue
        size_kb = local_path.stat().st_size / 1024
        print(f"  {hf_path}  ({size_kb:.1f} KB) ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Upload {hf_path}",
        )
        print("OK")

    url = f"https://huggingface.co/spaces/{repo_id}"
    api_url = f"https://{args.username}-{args.space}.hf.space"

    print(f"""
✅ Déploiement terminé !

   Space :  {url}
   API    :  {api_url}
   Health :  {api_url}/health
   Docs   :  {api_url}/docs

⏳ Le build Docker prend 2-5 minutes. Surveille les logs sur {url}

📌 Mets à jour l'extension Chrome avec cette URL :
   {api_url}
""")

    # Écrire l'URL prod dans un fichier pour l'extension
    prod_url_file = ROOT / "chrome-extension" / "api_url_prod.txt"
    prod_url_file.write_text(api_url)
    print(f"   URL sauvegardée dans {prod_url_file}")


if __name__ == "__main__":
    main()
