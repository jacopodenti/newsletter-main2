"""
STEP 3 — Embedding del dataset sul TESTO INTERO.

Legge data/dataset.jsonl, embedda `full_text` con text-embedding-3-large (3072 dim),
salva i vettori + label allineati.

Output: models/dataset_vectors.npz  (X = vettori, y = label, ids)
Uso:    python ocx_relevance/03_embed.py
"""
import json
import sys

import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, embed_texts, DATA, MODELS, EMBED_MODEL, EMBED_DIM


def main():
    log("STEP 3 — embedding del dataset (testo intero)")
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
