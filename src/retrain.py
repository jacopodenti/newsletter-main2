"""
Incremental retraining of classifiers from reviewed training data.
Uses embedding cache to avoid redundant API calls.

Usage:
    python src/retrain.py [--min-reviewed 10]
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlhf_store import get_training_set

MODELS_DIR = ROOT / "models"
CACHE_PATH = ROOT / "data" / "embedding_cache.pkl"
DRIFT_PATH = ROOT / "data" / "drift_baseline.json"
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


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _embed_with_cache(client: AzureOpenAI, rows: list[dict]) -> np.ndarray:
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    cache = _load_cache()

    embeddings = []
    to_compute = [(i, row) for i, row in enumerate(rows) if row["article_id"] not in cache]

    if to_compute:
        print(f"  Calcolo {len(to_compute)} nuovi embedding (cache: {len(cache)} esistenti)...")
        batch_size = 20
        for batch_start in range(0, len(to_compute), batch_size):
            batch = to_compute[batch_start: batch_start + batch_size]
            texts = [row["text_for_embed"] for _, row in batch]
            response = client.embeddings.create(input=texts, model=deployment)
            for (_, row), item in zip(batch, response.data):
                cache[row["article_id"]] = item.embedding
            time.sleep(DELAY)
        _save_cache(cache)

    for row in rows:
        embeddings.append(cache[row["article_id"]])

    return np.array(embeddings)


def _train_single(X: np.ndarray, y: list[str], field: str) -> tuple:
    """Train SVM/LogReg with LOO CV. Returns (model, label_encoder, loo_accuracy)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    loo = LeaveOneOut()

    n_classes = len(set(y))
    use_balanced = min(Counter(y).values()) < 5

    candidates = {
        "SVM (rbf)": SVC(kernel="rbf", C=5.0, probability=True,
                         class_weight="balanced" if use_balanced else None),
        "SVM (linear)": SVC(kernel="linear", C=1.0, probability=True,
                            class_weight="balanced" if use_balanced else None),
        "LogReg": LogisticRegression(max_iter=1000, C=1.0,
                                     class_weight="balanced" if use_balanced else None),
    }

    print(f"\n  {field}  ({n_classes} classi, {len(y)} esempi{', balanced' if use_balanced else ''})")
    best_score, best_name, best_model = -1, None, None

    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y_enc, cv=loo, scoring="accuracy")
        acc = scores.mean()
        print(f"    {name}: LOO accuracy = {acc:.2f}")
        if acc > best_score:
            best_score, best_name, best_model = acc, name, clf

    from sklearn.calibration import CalibratedClassifierCV
    min_class_count = min(Counter(y_enc).values())
    cv = min(5, min_class_count)
    best_model = CalibratedClassifierCV(best_model, method="isotonic", cv=max(2, cv))
    best_model.fit(X, y_enc)
    print(f"  → Scelto: {best_name} calibrato (acc={best_score:.2f})")
    print(f"  Distribuzione: {dict(Counter(y).most_common())}")
    return best_model, le, best_score


def _train_multilabel(X: np.ndarray, y_raw: list[list[str]], field: str) -> tuple:
    """Train OneVsRest SVM for multi-label field (macro_topics)."""
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_raw)

    clf = OneVsRestClassifier(SVC(kernel="linear", C=1.0, probability=True))
    clf.fit(X, Y)

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score
    if len(X) >= 5:
        Y_pred = cross_val_predict(clf, X, Y, cv=min(5, len(X)), method="predict")
        f1 = f1_score(Y, Y_pred, average="micro", zero_division=0)
    else:
        f1 = 0.0

    print(f"\n  {field}  ({len(mlb.classes_)} classi, {len(y_raw)} esempi)")
    print(f"  micro-F1 (CV): {f1:.2f}")
    print(f"  Classi: {list(mlb.classes_)[:10]}{'...' if len(mlb.classes_) > 10 else ''}")
    return clf, mlb, f1


def _load_existing_accuracy(field: str) -> float:
    path = MODELS_DIR / f"{field}_classifier.pkl"
    if not path.exists():
        return -1.0
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("loo_accuracy", -1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-reviewed", type=int, default=20,
                        help="Minimum reviewed examples required per field")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    client = _get_client()

    print("Caricamento training set (reviewed=True)...")
    all_reviewed = get_training_set()
    print(f"  {len(all_reviewed)} articoli revisionati totali")

    results = {}

    # --- single-label fields ---
    for field in ["content_type", "target_audience"]:
        rows = [r for r in all_reviewed if r.get(field, "").strip()]
        if len(rows) < args.min_reviewed:
            print(f"\n  {field}: solo {len(rows)} esempi (minimo {args.min_reviewed}) — skip")
            results[field] = {"skipped": True, "n": len(rows)}
            continue

        X_all = _embed_with_cache(client, rows)
        labels_all = [r[field] for r in rows]
        # rimuovi classi con un solo esempio (non calibrabili)
        class_counts = Counter(labels_all)
        mask = [i for i, l in enumerate(labels_all) if class_counts[l] >= 2]
        removed = {k for k, v in class_counts.items() if v < 2}
        if removed:
            print(f"  Classi rimosse per dati insufficienti: {removed}")
        X = X_all[mask]
        labels = [labels_all[i] for i in mask]
        model, le, score = _train_single(X, labels, field)

        old_score = _load_existing_accuracy(field)
        if score < old_score - 0.05:
            print(f"  ⚠ Regressione: {score:.2f} < {old_score:.2f} - 0.05 — modello NON salvato")
            results[field] = {"saved": False, "new_acc": score, "old_acc": old_score, "n": len(rows)}
            continue

        out_path = MODELS_DIR / f"{field}_classifier.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "label_encoder": le, "loo_accuracy": score}, f)
        print(f"  Salvato: {out_path}  (old={old_score:.2f} → new={score:.2f})")
        results[field] = {"saved": True, "new_acc": score, "old_acc": old_score, "n": len(rows)}

    # --- multi-label: macro_topics ---
    field = "macro_topics"
    rows_mt = [r for r in all_reviewed if r.get(field, "").strip()]
    if len(rows_mt) < args.min_reviewed:
        print(f"\n  {field}: solo {len(rows_mt)} esempi (minimo {args.min_reviewed}) — skip")
        results[field] = {"skipped": True, "n": len(rows_mt)}
    else:
        X_mt = _embed_with_cache(client, rows_mt)
        y_mt = [[s.strip() for s in r[field].split("|") if s.strip()] for r in rows_mt]
        clf, mlb, f1 = _train_multilabel(X_mt, y_mt, field)

        old_score = _load_existing_accuracy(field)
        if f1 < old_score - 0.05:
            print(f"  ⚠ Regressione: {f1:.2f} < {old_score:.2f} - 0.05 — modello NON salvato")
            results[field] = {"saved": False, "new_acc": f1, "old_acc": old_score, "n": len(rows_mt)}
        else:
            out_path = MODELS_DIR / f"{field}_classifier.pkl"
            with open(out_path, "wb") as f:
                pickle.dump({"model": clf, "label_encoder": mlb, "loo_accuracy": f1, "multilabel": True}, f)
            print(f"  Salvato: {out_path}  (old={old_score:.2f} → new={f1:.2f})")
            results[field] = {"saved": True, "new_acc": f1, "old_acc": old_score, "n": len(rows_mt)}

    # --- save drift baseline ---
    baseline = {}
    for field in ["content_type", "target_audience", "macro_topics"]:
        rows = [r for r in all_reviewed if r.get(field, "").strip()]
        if field == "macro_topics":
            counts: Counter = Counter()
            for r in rows:
                for s in r[field].split("|"):
                    if s.strip():
                        counts[s.strip()] += 1
        else:
            counts = Counter(r[field] for r in rows)
        baseline[field] = dict(counts)
    DRIFT_PATH.parent.mkdir(exist_ok=True)
    DRIFT_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBaseline drift salvato: {DRIFT_PATH}")

    # --- summary ---
    print("\n=== RIEPILOGO ===")
    for field, r in results.items():
        if r.get("skipped"):
            print(f"  {field}: SKIP ({r['n']} esempi)")
        elif r.get("saved"):
            print(f"  {field}: ✓ {r['old_acc']:.2f} → {r['new_acc']:.2f}  ({r['n']} esempi)")
        else:
            print(f"  {field}: ✗ regressione {r['old_acc']:.2f} → {r['new_acc']:.2f}  ({r['n']} esempi)")


if __name__ == "__main__":
    main()
