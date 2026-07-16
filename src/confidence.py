"""
Confidence gate: decide per-field whether to use local classifier or fall back to LLM.

Logic:
- Field uses local model if: model exists AND loo_accuracy >= accuracy_gate AND conf >= threshold
- Otherwise: field is flagged for LLM fallback
- classify_with_gate() returns labels + _meta dict with per-field source and confidence
"""
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from openai import AzureOpenAI

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.75"))
ACCURACY_GATE = float(os.getenv("ACCURACY_GATE", "0.80"))


def _load_model(field: str) -> Optional[dict]:
    path = MODELS_DIR / f"{field}_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model_dict = pickle.load(f)
    # [MOD] dimension guard: stesso controllo di embed_classify._load_model, così
    # anche questo percorso di caricamento rifiuta classificatori di dimensione errata.
    from embed_classify import _assert_classifier_dim
    _assert_classifier_dim(model_dict)
    return model_dict


def _model_qualifies(m: Optional[dict], accuracy_gate: float) -> bool:
    return m is not None and m.get("loo_accuracy", 0.0) >= accuracy_gate


def classify_with_gate(
    article: dict,
    article_emb: np.ndarray,
    obs_result: dict,
    llm_client: Optional[AzureOpenAI],
    osservatori: Optional[list],
    llm_deployment: Optional[str],
    threshold: float = CONF_THRESHOLD,
    accuracy_gate: float = ACCURACY_GATE,
    no_llm: bool = False,
) -> dict:
    """
    Classify article using local models where confident, LLM elsewhere.

    obs_result: output from classify_article_embed() — already has observatories + scores
    article_emb: numpy array, 1D embedding of the article

    Returns merged classification dict + _meta with source per field.
    """
    from embed_classify import predict_with_confidence

    pred = predict_with_confidence(article_emb, obs_result["scores"])

    labels = {}
    meta = {}

    # --- observatories: always use embedding ---
    labels["observatories"] = obs_result["observatories"]
    meta["source_observatories"] = "embed"
    meta["conf_observatories"] = pred["conf_observatories"]

    # --- single-label fields ---
    for field in ["content_type", "target_audience"]:
        m = _load_model(field)
        conf = pred.get(f"conf_{field}", 0.0)

        if _model_qualifies(m, accuracy_gate) and conf >= threshold:
            labels[field] = pred[field]
            meta[f"source_{field}"] = "model"
            meta[f"conf_{field}"] = conf
        elif no_llm:
            # best effort from model even if below threshold
            labels[field] = pred[field] if pred.get(field) else ""
            meta[f"source_{field}"] = "model_forced"
            meta[f"conf_{field}"] = conf
        else:
            labels[field] = None  # signal: needs LLM
            meta[f"source_{field}"] = "llm"
            meta[f"conf_{field}"] = conf

    # --- macro_topics ---
    m_mt = _load_model("macro_topics")
    conf_mt = pred.get("conf_macro_topics", 0.0)

    if _model_qualifies(m_mt, accuracy_gate) and conf_mt >= threshold:
        labels["macro_topics"] = pred["macro_topics"] or []
        meta["source_macro_topics"] = "model"
        meta["conf_macro_topics"] = conf_mt
    elif no_llm:
        labels["macro_topics"] = pred["macro_topics"] or []
        meta["source_macro_topics"] = "model_forced"
        meta["conf_macro_topics"] = conf_mt
    else:
        labels["macro_topics"] = None  # signal: needs LLM
        meta["source_macro_topics"] = "llm"
        meta["conf_macro_topics"] = conf_mt

    # fields still needing LLM
    llm_needed = [f for f in ["content_type", "target_audience", "macro_topics"]
                  if labels.get(f) is None]

    if llm_needed and not no_llm:
        if llm_client is None:
            raise RuntimeError("LLM client required but not provided")
        from classify import classify_article
        llm_result = classify_article(article, llm_client, osservatori, llm_deployment)
        for f in llm_needed:
            labels[f] = llm_result.get(f, "" if f != "macro_topics" else [])
        if "reason" not in labels:
            labels["reason"] = llm_result.get("reason", "")
    else:
        labels.setdefault("reason", "")

    labels["_meta"] = meta
    return labels


def all_autonomous(accuracy_gate: float = ACCURACY_GATE, min_examples: int = 100) -> bool:
    """
    Returns True if all fields have models meeting the gate threshold.
    Used to decide whether --no-llm is safe.
    """
    from rlhf_store import get_training_set
    reviewed = get_training_set()
    n = len(reviewed)
    if n < min_examples:
        return False
    for field in ["content_type", "target_audience", "macro_topics"]:
        m = _load_model(field)
        if not _model_qualifies(m, accuracy_gate):
            return False
    return True
