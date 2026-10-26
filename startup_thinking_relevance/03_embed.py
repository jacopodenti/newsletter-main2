"""
STEP 3 — Embedding del dataset sul TESTO INTERO.

Legge data/dataset.jsonl, embedda `full_text` con text-embedding-3-large (3072 dim),
salva i vettori + label allineati (stesso schema di ocx_relevance/03_embed.py).

Output: models/dataset_vectors.npz  (X = vettori, y = label, ids)
Uso:    python startup_thinking_relevance/03_embed.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import log, embed_texts, EMBED_MODEL, EMBED_DIM

DATA = HERE / "data"
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)


def main():
    log("STEP 3 — embedding del dataset startup-thinking (testo intero)")
    rows = [json.loads(l) for l in open(DATA / "dataset.jsonl", encoding="utf-8")]
    log(f"  righe: {len(rows)}  |  modello: {EMBED_MODEL}  |  dim: {EMBED_DIM}")

    texts = [f"{r['title']}. {r['full_text']}" for r in rows]
    log("  chiamo l'API di embedding...")
    X, tok = embed_texts(texts)
    y = np.array([r["label"] for r in rows])
    ids = np.array([r["id"] for r in rows])

    np.savez_compressed(MODELS / "dataset_vectors.npz", X=X, y=y, ids=ids)
    log(f"  vettori: shape {X.shape}")
    log(f"  token consumati: {tok}  (~{tok/max(len(rows),1):.0f}/articolo)")
    log(f"  salvato: {MODELS/'dataset_vectors.npz'}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
