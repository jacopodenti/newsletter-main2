"""
Funzioni condivise per la soluzione OCX relevance.
Isolata dalla produzione: non importa nulla da ../src.
"""
import os
import re
import html
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import trafilatura
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

# --- configurazione (unico punto dove cambiare modelli) ---
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
LLM_MODEL = "gpt-5.4"

DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)

# file sorgente (dati grezzi, in sola lettura)
XLSX_LABEL = ROOT / "LABEL-Newsletter.xlsx"          # foglio Dati (corpus) + Ranking (50 etichettati)
XLSX_EDITORIAL = ROOT / "OCX26_Newsletter.xlsx"      # articoli scelti per le newsletter
OSSERVATORI = ROOT / "config" / "osservatori.json"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


def log(msg):
    print(msg, flush=True)


def normalize_text(t: str) -> str:
    """Preprocessing unico: unescape HTML, togli tag, minuscolo, collassa spazi."""
    t = html.unescape(t or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def fetch_fulltext(url: str, fallback: str = "") -> tuple[str, str]:
    """Scarica il testo INTERO dall'URL. Ritorna (testo, sorgente) con sorgente in {'full','fallback'}."""
    if not url or not url.startswith("http"):
        return fallback, "fallback"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        body = trafilatura.extract(r.text, include_comments=False, include_tables=False,
                                   favor_precision=True) if r.ok else None
    except Exception:
        body = None
    if body and len(body) > len(fallback) + 50:
        return body[:8000], "full"
    return fallback, "fallback"


def _embed_client() -> AzureOpenAI:
    ep = (os.getenv("AZURE_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
    if ep.endswith("/openai/v1"):
        ep = ep[: -len("/openai/v1")]
    return AzureOpenAI(api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01"),
                       azure_endpoint=ep, api_key=os.getenv("AZURE_OPENAI_API_KEY"))


def embed_texts(texts: list[str], batch: int = 8, max_chars: int = 6000, workers: int = 4):
    """Embedding di una lista di testi -> array (n, EMBED_DIM). Traccia i token totali.
    Parallelo ma MODERATO (4 alla volta) per non sforare il limite Azure (429).
    Su 429/errore: aspetta 1,2,4,8,16,32 s e riprova (fino a 6 tentativi). Ordine preservato."""
    c = _embed_client()
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    results = [None] * len(chunks)
    state = {"tok": 0, "done": 0}
    lock = threading.Lock()

    def do(idx):
        chunk = [(normalize_text(t)[:max_chars] or "vuoto") for t in chunks[idx]]
        r = None
        for attempt in range(6):
            try:
                r = c.with_options(timeout=60.0).embeddings.create(
                    input=chunk, model=EMBED_MODEL, dimensions=EMBED_DIM)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)  # attesa crescente: 1,2,4,8,16 s
        results[idx] = [d.embedding for d in r.data]
        with lock:
            state["tok"] += r.usage.total_tokens
            state["done"] += 1
            if state["done"] % 10 == 0 or state["done"] == len(chunks):
                log(f"    embedding batch {state['done']}/{len(chunks)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, range(len(chunks))))
    out = [v for chunk in results for v in chunk]
    return np.array(out), state["tok"]


def llm_client() -> OpenAI:
    return OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                  base_url=os.environ["AZURE_OPENAI_ENDPOINT"])


def ocx_description() -> str:
    import json
    d = json.load(open(OSSERVATORI))
    o = next(x for x in d if x["slug"] == "omnichannel-customer-experience")
    return o.get("descrizione_enhanced") or o["descrizione"]
