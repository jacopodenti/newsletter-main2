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
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import numpy as np
import requests
import trafilatura
import yaml
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
TRACKING_PARAMS_PATH = ROOT / "corpus" / "tracking_params.yaml"  # config dedup URL, vedi canonicalize_url

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


def log(msg):
    print(msg, flush=True)


def detect_lang(text: str) -> str:
    """Lingua dominante del testo ('it', 'en', ...). 'it' se il testo è vuoto/troppo corto
    per essere rilevato (langdetect alza LangDetectException) — 'it' perché è la lingua
    maggioritaria negli esempi di riferimento, quindi il fallback più sicuro."""
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # langdetect non è deterministico senza un seed fisso
    text = (text or "").strip()
    if not text:
        return "it"
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "it"


def format_date(raw: str) -> str:
    """Da data RFC822 di un feed (es. 'Tue, 21 Jul 2026 08:47:09 +0000') a gg/mm/aaaa.
    Stringa vuota se non interpretabile."""
    from email.utils import parsedate_to_datetime
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return ""


def group_by_cluster(rows):
    """Riordina una lista di righe (dict con chiave 'cluster_id', es. da ranking.csv):
    i membri di uno stesso cluster (sospetto duplicato) vengono accostati subito dopo
    la prima occorrenza — che, siccome `rows` arriva già ordinata per punteggio, è
    sempre il membro più in alto in classifica. Gli articoli senza cluster (o i cluster
    già emessi) mantengono l'ordine originale. Usata per i fogli di revisione, dove
    vedere le fonti duplicate una accanto all'altra aiuta a scegliere quale tenere."""
    by_cluster = {}
    for r in rows:
        cid = r.get("cluster_id")
        if cid:
            by_cluster.setdefault(cid, []).append(r)
    seen, ordered = set(), []
    for r in rows:
        cid = r.get("cluster_id")
        if not cid:
            ordered.append(r)
        elif cid not in seen:
            seen.add(cid)
            ordered.extend(by_cluster[cid])
    return ordered


def normalize_text(t: str) -> str:
    """Preprocessing unico: unescape HTML, togli tag, minuscolo, collassa spazi."""
    t = html.unescape(t or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def load_tracking_params() -> set[str]:
    """Parametri di query da rimuovere in fase di canonicalizzazione URL (dedup) — lista in
    config (corpus/tracking_params.yaml), non hardcoded: se un feed nuovo porta un
    parametro di tracking non ancora noto, si aggiunge lì, non nel codice."""
    return set(yaml.safe_load(open(TRACKING_PARAMS_PATH, encoding="utf-8"))["tracking_params"])


def canonicalize_url(url: str, tracking_params: set[str]) -> str:
    """URL canonico per il dedup per link: host in minuscolo, senza i parametri di query in
    `tracking_params` (config), senza fragment, senza slash finale sul path. Stringa vuota
    invariata. NON risolve redirect (nessuna richiesta di rete): un dominio-ponte
    (news.google.com, bit.ly...) resta canonicalizzato ma non risolto — non osservato nei
    feed attuali (audit 13/08), ma è un limite dichiarato, non un caso silenzioso."""
    if not url:
        return url
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in tracking_params]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, urlencode(query), ""))


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
    # Margine EURISTICO, non calibrato su dati (a differenza di MIN_PER_CLASS/
    # CLUSTER_SIM_THRESHOLD in ranker.py): non basta che il testo scrapato sia più lungo
    # del fallback (il summary RSS) di un solo carattere — serve un margine minimo perché
    # un estratto di poco più lungo del summary è spesso il segno di uno scraping fallito
    # a metà (es. un frammento di paywall o cookie banner), non un vero guadagno di
    # informazione. 50 è un valore prudenziale, non testato sistematicamente: va
    # aggiustato se si osservano scelte sbagliate tra full/fallback.
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
