import hashlib
import html  # [MOD] serve per html.unescape() in normalize_text (decodifica entità HTML)
import json
import math
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional


import numpy as np
from openai import AzureOpenAI

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


# [MOD-STEP2] Upgrade del modello: da text-embedding-3-small a -3-large.
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
# [MOD] Dimensione del vettore di embedding. Usata sia come parametro esplicito
# dimensions= nella chiamata API, sia come riferimento per il dimension guard
# che verifica la compatibilità dei classificatori già addestrati.
# [MOD-STEP2] text-embedding-3-large produce vettori a 3072 dimensioni.
# NOTA: text-embedding-3-large supporta il troncamento "Matryoshka" — è possibile
# ridurre EMBEDDING_DIM (es. 1024) passandolo come dimensions= per abbattere costi
# di storage/confronto mantenendo buona qualità, senza cambiare modello.
EMBEDDING_DIM = 3072
RATE_LIMIT_DELAY = 0.1


def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # strip /openai/v1/ suffix added by Responses API config — embeddings use the base endpoint
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[: -len("/openai/v1")]
    return AzureOpenAI(
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01"),
        azure_endpoint=endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def embedding_model_name() -> str:
    """Nome del deployment di embedding attualmente in uso (env override o default)."""
    return os.getenv("AZURE_EMBEDDING_DEPLOYMENT", EMBEDDING_DEPLOYMENT)


def _embed(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    deployment = embedding_model_name()
    # [MOD] dimensions= esplicito: rende la dimensione del vettore deterministica e
    # allineata a EMBEDDING_DIM (necessario per il guard sui classificatori).
    response = client.embeddings.create(input=texts, model=deployment, dimensions=EMBEDDING_DIM)
    return [item.embedding for item in response.data]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# [MOD-NUOVA] Pipeline di preprocessing UNICA usata sia per gli articoli sia per
# gli osservatori, così i due lati del confronto cosine sono trattati allo stesso
# modo (prima erano asimmetrici: articoli con _strip_html, osservatori grezzi).
def normalize_text(text: str) -> str:
    """Single preprocessing path applied to both article and observatory text
    before embedding: decode HTML entities, strip HTML tags, lowercase, and
    collapse whitespace."""
    text = html.unescape(text)            # [MOD] decodifica entità HTML (&amp; -> &, &nbsp; -> spazio)
    text = re.sub(r"<[^>]+>", " ", text)  # [MOD] rimuove i tag HTML
    text = text.lower()                   # [MOD] lowercasing
    text = re.sub(r"\s+", " ", text)      # [MOD] collassa spazi multipli in uno solo
    return text.strip()                   # [MOD] rimuove spazi iniziali/finali


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# [MOD-RISCRITTA] Prima prendeva UN solo campo (descrizione_enhanced or descrizione).
# Ora concatena tutti i campi descrittivi disponibili -> rappresentazione più ricca
# dell'osservatorio (es. testo OCX passato da ~1000 a ~5500 caratteri).
def _observatory_text(o: dict) -> str:
    """Concatenate all available descriptive fields for an observatory:
    enhanced description + research topics + FAQ question/answer pairs."""
    parts: list[str] = []

    # [MOD] descrizione arricchita, con fallback su quella base se assente
    description = o.get("descrizione_enhanced") or o.get("descrizione")
    if description:
        parts.append(description)

    # [MOD] aggiunge i temi di ricerca (lista di stringhe) uniti da virgola
    topics = o.get("research_topics")
    if topics:
        parts.append(", ".join(topics))

    # [MOD] aggiunge le FAQ come coppie domanda + risposta (la risposta può essere lista)
    faq = o.get("faq")
    if faq:
        faq_parts = []
        for item in faq:
            question = item.get("q", "")
            answer = item.get("a", "")
            if isinstance(answer, list):
                answer = " ".join(answer)
            faq_parts.append(f"{question} {answer}".strip())
        if faq_parts:
            parts.append(" ".join(faq_parts))

    return " ".join(parts)


CACHE_PATH = MODELS_DIR / "observatory_embeddings.pkl"


def _osservatori_hash(osservatori: list[dict]) -> str:
    # [MOD] L'hash ora si basa sullo STESSO testo normalizzato che viene davvero
    # embeddato. Prima guardava solo un campo: modifiche a research_topics/faq NON
    # invalidavano la cache (bug latente). Ora la cache si rigenera correttamente.
    content = json.dumps([normalize_text(_observatory_text(o)) for o in osservatori], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def build_observatory_embeddings(osservatori: list[dict], client: AzureOpenAI) -> list[dict]:
    """Pre-compute embeddings for all observatories, with disk cache invalidated by content hash."""
    current_hash = _osservatori_hash(osservatori)

    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("hash") == current_hash:
            print("  Embedding osservatori caricati dalla cache.")
            return cached["embeddings"]

    # [MOD] applica normalize_text() al testo osservatorio prima dell'embedding
    texts = [normalize_text(_observatory_text(o)) for o in osservatori]
    print(f"  Embedding {len(texts)} osservatori (ricalcolo)...")
    embeddings = _embed(client, texts)
    result = [
        {"slug": o["slug"], "id": o["id"], "embedding": emb}
        for o, emb in zip(osservatori, embeddings)
    ]

    MODELS_DIR.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"hash": current_hash, "embeddings": result}, f)

    return result


def classify_article_embed(
    article: dict,
    client: AzureOpenAI,
    observatory_embeddings: list[dict],
    threshold: float = 0.35,
    top_k: int = 10,
) -> dict:
    """
    Classify article by cosine similarity against pre-computed observatory embeddings.
    Returns observatories above threshold, up to top_k, sorted by score descending.
    """
    # [MOD] testo articolo (titolo + sommario) passato per la stessa normalize_text()
    # degli osservatori; prima usava _strip_html senza unescape/lowercasing
    text = normalize_text(f"{article['title']}. {article['summary']}")
    [article_emb] = _embed(client, [text])
    time.sleep(RATE_LIMIT_DELAY)

    scores = [
        (obs["slug"], _cosine(article_emb, obs["embedding"]))
        for obs in observatory_embeddings
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    matched = [slug for slug, score in scores[:top_k] if score >= threshold]
    return {
        "observatories": matched,
        "scores": {slug: round(score, 4) for slug, score in scores[:top_k]},
        "all_scores": {slug: round(score, 4) for slug, score in scores},
    }


# [MOD-NUOVA] Dimension guard: estrae la dimensione di input su cui un classificatore
# sklearn è stato addestrato (n_features_in_), gestendo anche i modelli multi-output
# (macro_topics) che espongono la dimensione sui singoli estimators_.
def _classifier_dim(clf) -> Optional[int]:
    if clf is None:
        return None
    if hasattr(clf, "n_features_in_"):
        return int(clf.n_features_in_)
    estimators = getattr(clf, "estimators_", None)
    if estimators and hasattr(estimators[0], "n_features_in_"):
        return int(estimators[0].n_features_in_)
    return None


# [MOD-NUOVA] Verifica che il classificatore caricato sia compatibile con la
# dimensione dell'embedding model corrente. Se non lo è, blocca tutto: un mismatch
# significa che il pkl è stato addestrato con un altro modello di embedding.
def _assert_classifier_dim(model_dict: dict) -> None:
    dim = _classifier_dim(model_dict.get("model"))
    if dim is not None and dim != EMBEDDING_DIM:
        raise ValueError(
            f"Classifier was trained on dimension {dim} but current embedding "
            f"model returns {EMBEDDING_DIM}. "
            f"Delete classifier pkl files and retrain."
        )


def _load_model(field: str) -> Optional[dict]:
    path = MODELS_DIR / f"{field}_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model_dict = pickle.load(f)
    _assert_classifier_dim(model_dict)  # [MOD] dimension guard
    return model_dict


def predict_with_confidence(
    article_emb: np.ndarray,
    obs_scores: dict[str, float],
) -> dict:
    """
    Run all available local classifiers and return labels + per-field confidence.

    Returns:
        {
          "content_type": str,       "conf_content_type": float,
          "target_audience": str,    "conf_target_audience": float,
          "macro_topics": list[str], "conf_macro_topics": float,
          "conf_observatories": float,
        }
    """
    result = {}

    # observatories: max cosine score as confidence proxy
    result["conf_observatories"] = round(max(obs_scores.values()), 4) if obs_scores else 0.0

    X = article_emb.reshape(1, -1)

    for field in ["content_type", "target_audience"]:
        m = _load_model(field)
        if m is None:
            result[field] = None
            result[f"conf_{field}"] = 0.0
            continue
        clf = m["model"]
        le: pickle.ANY = m["label_encoder"]
        proba = clf.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        result[field] = le.inverse_transform([idx])[0]
        result[f"conf_{field}"] = round(float(proba[idx]), 4)

    m_mt = _load_model("macro_topics")
    if m_mt is None:
        result["macro_topics"] = None
        result["conf_macro_topics"] = 0.0
    else:
        clf = m_mt["model"]
        mlb = m_mt["label_encoder"]
        proba_matrix = np.array([est.predict_proba(X)[0][1] for est in clf.estimators_])
        threshold_mt = 0.40
        selected = [mlb.classes_[i] for i, p in enumerate(proba_matrix) if p >= threshold_mt]
        result["macro_topics"] = selected or [mlb.classes_[int(np.argmax(proba_matrix))]]
        result["conf_macro_topics"] = round(float(np.max(proba_matrix)), 4)

    return result

