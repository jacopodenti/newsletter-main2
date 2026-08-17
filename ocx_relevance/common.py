"""
Funzioni condivise per la soluzione OCX relevance.
Isolata dalla produzione: non importa nulla da ../src.
"""
import os
import re
import html
import time
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
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

STORE_DIR = ROOT / "store"; STORE_DIR.mkdir(exist_ok=True)
ARTICLES_STORE_PATH = STORE_DIR / "articles.jsonl"
# Righe MAI selezionate per una newsletter vengono potate dallo store oltre questa età
# (da first_seen): non possono più ricomparire in un fetch futuro (finestra di 30gg in
# fetch_corpus.py) e a quella distanza il valore per "riconosci uno sviluppo successivo"
# è basso. Le righe con selected_for_newsletter=True NON vengono mai potate per età
# (vedi upsert_articles_store) — servono a tempo indeterminato per "non riproporre".
ARTICLES_RETENTION_DAYS = 365

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


def content_hash(article: dict) -> str | None:
    """Hash sha256 del contenuto normalizzato (titolo + full_text) — SOLO se
    text_source == 'full' (stesso limite di dedup_by_content_hash: l'hash del solo
    summary rischia falsi positivi tra articoli diversi con summary templato). None
    altrimenti. Estratta a parte così dedup_by_content_hash e lo store persistente
    (upsert_articles_store) usano esattamente la stessa formula, non due copie che
    possono disallinearsi."""
    if article.get("text_source") != "full":
        return None
    return hashlib.sha256(normalize_text(f"{article['title']} {article['full_text']}").encode("utf-8")).hexdigest()


def dedup_by_content_hash(articles: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicazione ESATTA per contenuto: hash sha256 del testo normalizzato
    (titolo + full_text), calcolato SOLO per gli articoli con text_source == 'full'. Se il
    testo è in fallback sul summary RSS (articolo non scaricabile), l'hash del solo summary
    rischierebbe di fondere articoli genuinamente diversi che condividono un summary
    templato — es. le rubriche ricorrenti tipo "Tech Life"/"digital health coffee time
    briefing" trovate nell'audit del 13/08, stesso summary-stile ma episodi diversi.
    Meglio un duplicato non catturato che due articoli distinti fusi: questi articoli non
    vengono toccati (né deduplicati né scartati), solo esclusi dal conteggio dell'hash.

    Ritorna (articoli_deduplicati, contatori) con contatori =
    {'saltati_fallback': articoli esclusi dall'hash perché non hanno full_text reale,
     'scartati_duplicato_hash': articoli scartati perché hash identico a uno già visto}."""
    seen_hashes = set()
    kept = []
    n_skipped_fallback = 0
    n_dropped_hash = 0
    for a in articles:
        h = content_hash(a)
        if h is None:
            n_skipped_fallback += 1
            kept.append(a)
            continue
        if h in seen_hashes:
            n_dropped_hash += 1
            continue
        seen_hashes.add(h)
        kept.append(a)
    return kept, {"saltati_fallback": n_skipped_fallback, "scartati_duplicato_hash": n_dropped_hash}


def load_articles_store() -> dict:
    """{(url_canonico, osservatorio): record} da store/articles.jsonl. Dict vuoto se lo
    store non esiste ancora (primo run)."""
    if not ARTICLES_STORE_PATH.exists():
        return {}
    store = {}
    for line in open(ARTICLES_STORE_PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        store[(r["url_canonico"], r["osservatorio"])] = r
    return store


# Campi che un upsert può aggiornare (oltre a first_seen/selected_for_newsletter, gestiti
# a parte con regole diverse). Un valore None in input NON sovrascrive un valore già
# presente nello store (serve al backfill del commit 3, che non conosce content_hash/
# text_source/last_rank per le righe storiche e non deve cancellarli se già noti da un
# run reale).
_ARTICLE_FIELDS = ("url_canonico", "osservatorio", "content_hash", "published_date", "text_source", "last_rank")


def upsert_articles_store(records: list[dict]) -> dict:
    """Read-merge-write su store/articles.jsonl (stesso pattern di dedup_stats.json).

    Ogni record in input: dict con le chiavi in _ARTICLE_FIELDS, più opzionalmente
    'first_seen' (solo per un inserimento nuovo con una data storica nota, es. il
    backfill del commit 3 — se assente su una riga nuova si usa la data di oggi) e
    'selected_for_newsletter' (bool, OR sticky: una volta True resta True, non si
    "de-seleziona" mai un articolo).

    RETENTION: dopo il merge, le righe con selected_for_newsletter falso e first_seen
    più vecchio di ARTICLES_RETENTION_DAYS vengono rimosse dallo store (non possono più
    ricomparire in un fetch futuro, vedi costante). Le righe selezionate non si potano mai.

    Ritorna {nuove, aggiornate, potate, totale_dopo}."""
    store = load_articles_store()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_new, n_updated = 0, 0

    for rec in records:
        key = (rec["url_canonico"], rec["osservatorio"])
        old = store.get(key)
        merged = dict(old) if old else {}
        for field in _ARTICLE_FIELDS:
            val = rec.get(field)
            if val is not None:
                merged[field] = val
            else:
                merged.setdefault(field, None)
        was_selected = bool(old.get("selected_for_newsletter")) if old else False
        merged["selected_for_newsletter"] = bool(rec.get("selected_for_newsletter")) or was_selected
        if old:
            merged["first_seen"] = old["first_seen"]
            n_updated += 1
        else:
            merged["first_seen"] = rec.get("first_seen") or today
            n_new += 1
        store[key] = merged

    cutoff = (datetime.now(timezone.utc) - timedelta(days=ARTICLES_RETENTION_DAYS)).strftime("%Y-%m-%d")
    kept, n_pruned = {}, 0
    for key, r in store.items():
        if not r.get("selected_for_newsletter") and r.get("first_seen", today) < cutoff:
            n_pruned += 1
            continue
        kept[key] = r

    with open(ARTICLES_STORE_PATH, "w", encoding="utf-8") as f:
        for r in kept.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"nuove": n_new, "aggiornate": n_updated, "potate": n_pruned, "totale_dopo": len(kept)}


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


EMBEDDINGS_STORE_PATH = STORE_DIR / "embeddings.npz"
# Retention degli embedding: doppio margine rispetto alla finestra di fetch (30gg in
# fetch_corpus.py) — un embedding più vecchio di così non può più essere riusato da
# nessun run futuro (l'articolo non può ricomparire nel corpus), quindi è solo peso
# morto nel file. Il "conto" è last_seen (ultima volta che è stato richiesto, hit o
# miss), non first_seen: un embedding riusato di continuo non scade mai.
EMBEDDINGS_RETENTION_DAYS = 60


def load_embeddings_store() -> tuple[list[str], "np.ndarray", list[str]]:
    """(urls, X, last_seen) da store/embeddings.npz. Liste/array vuoti se non esiste
    ancora. X è float32, shape (n, EMBED_DIM)."""
    if not EMBEDDINGS_STORE_PATH.exists():
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32), []
    d = np.load(EMBEDDINGS_STORE_PATH, allow_pickle=False)
    return list(d["urls"]), d["X"].astype(np.float32), list(d["last_seen"])


def embed_texts_cached(articles: list[dict], tracking_params: set[str], workers: int = 4) -> tuple["np.ndarray", int, dict]:
    """Come embed_texts, ma con cache PERSISTENTE CONDIVISA tra le 3 pipeline
    (store/embeddings.npz, chiave = URL canonicalizzato — il testo di un articolo non
    dipende da quale osservatorio lo sta valutando, quindi la cache si riusa tra
    pipeline oltre che tra run). Manda all'API SOLO gli URL non ancora in cache.

    `articles`: dict con almeno 'link', 'title', 'full_text'. Ritorna (X allineato
    all'ordine di `articles`, token EFFETTIVAMENTE spesi [0 se tutto cache-hit],
    {'n_cache_hit', 'n_nuovi', 'potati'}).

    CRITERIO DI CORRETTEZZA (verificato sui dati reali il 18/08, non solo dichiarato):
    l'embedding di Azure NON è bit-per-bit deterministico tra due chiamate indipendenti
    sullo stesso testo (deriva coseno misurata: 0,999992-0,999999 — vedi commit "check
    di determinismo embedding"). Questo significa che l'ORDINE del ranking non è mai
    esattamente riproducibile tra un run che riusa un embedding dalla cache e uno che lo
    ricalcolerebbe da zero — e non lo sarebbe nemmeno SENZA cache, tra due run indipendenti
    a distanza di tempo: non è un rischio introdotto da questa funzione, è una proprietà
    dell'API. "Ordine identico" (bit-per-bit o anche solo in una fascia come il top-50) è
    quindi un criterio SBAGLIATO da pretendere — verificato che fallisce anche in pratica
    (4 scambi nel top-50 su un campione reale, tutti tra posizioni adiacenti con differenza
    di score 0,00001-0,0005).

    Il criterio corretto è: nessun articolo deve attraversare una SOGLIA OPERATIVA — oggi
    rank 150 (taglio del gate LLM in 06_rank.py), in futuro le soglie di selezione della
    newsletter. Verificato sui dati reali (fascia di rank 100-200, 101 articoli, imbedding
    indipendente reale): 2 attraversamenti su 101 (i due articoli esattamente a cavallo
    della soglia, rank 150 e 151, hanno scambiato posizione — differenza di score 0,000064).
    Non è quindi un evento a probabilità zero: chi valuta una modifica a questa funzione
    dovrebbe confrontare il TASSO di attraversamento osservato (~2% in una fascia di 101
    articoli attorno a una soglia), non aspettarsi che scenda a zero."""
    urls, X, last_seen = load_embeddings_store()
    idx_by_url = {u: i for i, u in enumerate(urls)}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    canon = [canonicalize_url(a["link"], tracking_params) for a in articles]
    missing_pos = [i for i, c in enumerate(canon) if c not in idx_by_url]

    tok = 0
    if missing_pos:
        texts = [f"{articles[i]['title']}. {articles[i]['full_text']}" for i in missing_pos]
        X_new, tok = embed_texts(texts, workers=workers)
        X_new = X_new.astype(np.float32)
        for j, i in enumerate(missing_pos):
            c = canon[i]
            if c in idx_by_url:
                continue  # stesso URL ripetuto due volte in questo batch (non atteso, difesa)
            idx_by_url[c] = len(urls)
            urls.append(c)
            last_seen.append(today)
            X = np.vstack([X, X_new[j:j + 1]]) if X.shape[0] else X_new[j:j + 1].copy()

    X_out = np.zeros((len(articles), EMBED_DIM), dtype=np.float32)
    for i, c in enumerate(canon):
        pos = idx_by_url[c]
        X_out[i] = X[pos]
        last_seen[pos] = today  # riusato adesso: non scade finché continua a comparire nei fetch

    # Esenzione dalla potatura per età: un articolo selected_for_newsletter (in ALMENO
    # un osservatorio — la cache embedding è condivisa, non ha una dimensione per
    # osservatorio) serve a tempo indeterminato per confrontare articoli nuovi contro
    # quelli già usciti — stessa regola già applicata ai metadati in
    # upsert_articles_store, allineata qui per non avere due finestre di retention
    # scollegate (365gg metadati vs 60gg embedding: senza questa esenzione, dopo 2 mesi
    # l'embedding di un articolo già in newsletter sparirebbe mentre il suo metadato
    # resterebbe, rendendo il confronto impossibile proprio per i casi che contano di più).
    selected_urls = {url for (url, _obs), r in load_articles_store().items() if r.get("selected_for_newsletter")}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=EMBEDDINGS_RETENTION_DAYS)).strftime("%Y-%m-%d")
    keep_idx = [i for i, (u, d) in enumerate(zip(urls, last_seen)) if d >= cutoff or u in selected_urls]
    n_pruned = len(urls) - len(keep_idx)
    urls_kept = [urls[i] for i in keep_idx]
    X_kept = X[keep_idx] if keep_idx else np.zeros((0, EMBED_DIM), dtype=np.float32)
    last_seen_kept = [last_seen[i] for i in keep_idx]

    np.savez_compressed(EMBEDDINGS_STORE_PATH,
                         urls=np.array(urls_kept),  # dtype stringa nativo (<U...), niente pickle
                         X=X_kept.astype(np.float32),
                         last_seen=np.array(last_seen_kept))

    return X_out, tok, {"n_cache_hit": len(articles) - len(missing_pos), "n_nuovi": len(missing_pos), "potati": n_pruned}


def llm_client() -> OpenAI:
    return OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                  base_url=os.environ["AZURE_OPENAI_ENDPOINT"])


def ocx_description() -> str:
    import json
    d = json.load(open(OSSERVATORI))
    o = next(x for x in d if x["slug"] == "omnichannel-customer-experience")
    return o.get("descrizione_enhanced") or o["descrizione"]
