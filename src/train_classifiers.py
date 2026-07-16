"""
Train lightweight classifiers (SVM) on top of embeddings using LLM labels as ground truth.
Uses leave-one-out cross-validation to evaluate accuracy on small datasets.
Saves trained models to models/ directory for use in the pipeline.

Usage:
    python src/train_classifiers.py --results output/results_20260429_125950.csv
"""
import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
DELAY = 0.05


def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[: -len("/openai/v1")]
    return AzureOpenAI(
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01"),
        azure_endpoint=endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def _embed_batch(client: AzureOpenAI, texts: list[str], batch_size: int = 20) -> list[list[float]]:
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=deployment)
        all_embeddings.extend([item.embedding for item in response.data])
        time.sleep(DELAY)
    return all_embeddings


def _clean_summary(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def train_and_eval(X: np.ndarray, y: list[str], field: str) -> tuple:
    """Train SVM + Logistic Regression, evaluate with LOO CV, return best model."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    loo = LeaveOneOut()

    # try both models
    candidates = {
        "SVM (rbf)": SVC(kernel="rbf", C=5.0, probability=True),
        "SVM (linear)": SVC(kernel="linear", C=1.0, probability=True),
        "LogReg": LogisticRegression(max_iter=1000, C=1.0),
    }

    print(f"\n  {field}  ({len(set(y))} classi, {len(y)} esempi)")
    best_score = -1
    best_name = None
    best_model = None

    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y_enc, cv=loo, scoring="accuracy")
        acc = scores.mean()
        print(f"    {name}: LOO accuracy = {acc:.2f}")
        if acc > best_score:
            best_score = acc
            best_name = name
            best_model = clf

    # train final model on all data
    best_model.fit(X, y_enc)
    print(f"  → Scelto: {best_name} (acc={best_score:.2f})")

    # class distribution
    from collections import Counter
    dist = Counter(y)
    print(f"  Distribuzione classi: {dict(dist.most_common())}")

    return best_model, le, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    print(f"Caricati {len(rows)} articoli da {results_path.name}")

    client = _get_client()

    # build texts
    texts = []
    for row in rows:
        summary = _clean_summary(row.get("summary", ""))
        texts.append(f"{row['title']}. {summary}")

    print(f"\nCalcolo embedding per {len(texts)} articoli...")
    embeddings = _embed_batch(client, texts)
    X = np.array(embeddings)
    print(f"Matrice embedding: {X.shape}")

    # extract labels — skip rows with empty labels
    fields = {
        "content_type": [],
        "target_audience": [],
    }

    valid_idx = {f: [] for f in fields}
    for i, row in enumerate(rows):
        for f in fields:
            val = row.get(f, "").strip()
            if val:
                fields[f].append(val)
                valid_idx[f].append(i)

    MODELS_DIR.mkdir(exist_ok=True)
    results = {}

    for field, labels in fields.items():
        idx = valid_idx[field]
        X_field = X[idx]
        model, le, score = train_and_eval(X_field, labels, field)
        results[field] = {"model": model, "label_encoder": le, "loo_accuracy": score}

        # save
        out_path = MODELS_DIR / f"{field}_classifier.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "label_encoder": le, "loo_accuracy": score}, f)
        print(f"  Salvato: {out_path}")

    print("\n=== RIEPILOGO ===")
    for field, r in results.items():
        print(f"  {field}: LOO accuracy = {r['loo_accuracy']:.2f}")

    print("\nPer usare i modelli nel pipeline, importa predict_taxonomy da embed_classify.py")


if __name__ == "__main__":
    main()
