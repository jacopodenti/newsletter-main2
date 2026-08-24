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
from collections import Counter
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


def parse_date(raw: str):
    """Da data di un feed RSS a datetime timezone-aware, o None se vuota/non interpretabile.
    Estratta a parte così format_date, il calcolo dello spread temporale dei cluster
    (ranker/06_rank.py) e il filtro di freschezza (corpus/fetch_corpus.py) usano lo stesso
    parsing, non copie che possono disallinearsi.

    Prova RFC822 (es. 'Tue, 21 Jul 2026 08:47:09 +0000', il formato più comune), poi ISO8601
    (es. '2026-08-18T09:35:35-04:00', usato da The Verge/Business Insider/Fast Company —
    verificato 18/08: senza questo fallback, quei 3 feed perdevano ogni run tutti gli
    articoli come "data non interpretabile", indistinguibili da un feed davvero fermo).
    Un datetime senza timezone (visto su Fast Company) viene normalizzato a UTC: un mix di
    datetime aware e naive nello stesso confronto solleverebbe TypeError."""
    from email.utils import parsedate_to_datetime
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_date(raw: str) -> str:
    """Da data RFC822 di un feed a gg/mm/aaaa. Stringa vuota se non interpretabile."""
    dt = parse_date(raw)
    return dt.strftime("%d/%m/%Y") if dt else ""


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

    # Scrittura atomica: su file temporaneo poi rename (os.replace è atomico sullo
    # stesso filesystem) — un'interruzione a metà lasciava altrimenti articles.jsonl
    # troncato, con le righe non ricostruibili (audit 22/08).
    tmp_path = ARTICLES_STORE_PATH.with_name(ARTICLES_STORE_PATH.name + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in kept.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, ARTICLES_STORE_PATH)

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

    # Scrittura atomica: su file temporaneo (nome che finisce comunque in .npz, così
    # numpy non lo rinomina da solo) poi rename — un'interruzione a metà lasciava
    # altrimenti embeddings.npz troncato/corrotto, con embedding già pagati da
    # riscaricare (audit 22/08).
    tmp_path = EMBEDDINGS_STORE_PATH.with_name(f"{EMBEDDINGS_STORE_PATH.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp_path,
                         urls=np.array(urls_kept),  # dtype stringa nativo (<U...), niente pickle
                         X=X_kept.astype(np.float32),
                         last_seen=np.array(last_seen_kept))
    os.replace(tmp_path, EMBEDDINGS_STORE_PATH)

    return X_out, tok, {"n_cache_hit": len(articles) - len(missing_pos), "n_nuovi": len(missing_pos), "potati": n_pruned}


def llm_client() -> OpenAI:
    return OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                  base_url=os.environ["AZURE_OPENAI_ENDPOINT"])


# Tariffe gpt-5.4 (€ per milione di token) — confermate dall'utente, stesse usate in
# llm_only_rank.py. Finora il costo "di produzione" del gate viveva solo in un dizionario
# scritto a mano (PROD_COST in llm_only_rank.py) che dichiara di venire dai log senza che
# nessun log lo confermi — vedi ricognizione del 20/08. log_gate_cost() esiste per chiudere
# quel buco: da qui in avanti ogni chiamata del gate persiste il proprio costo reale.
GATE_PRICE_EUR_PER_1M = {"in": 2.50, "out": 15.00}
_GATE_COST_LOG_LOCK = threading.Lock()


def log_gate_cost(log_path, run_id, title, tokens_in, tokens_out):
    """Appende una riga jsonl con il costo REALE di una singola chiamata del gate. Non
    sovrascrive: il file accumula across run, ogni riga porta run_id per poterle raggruppare."""
    cost_eur = (tokens_in * GATE_PRICE_EUR_PER_1M["in"] + tokens_out * GATE_PRICE_EUR_PER_1M["out"]) / 1_000_000
    record = {
        "run_id": run_id,
        "title": title,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_eur": round(cost_eur, 6),
    }
    with _GATE_COST_LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cost_eur


def ocx_description() -> str:
    import json
    d = json.load(open(OSSERVATORI))
    o = next(x for x in d if x["slug"] == "omnichannel-customer-experience")
    return o.get("descrizione_enhanced") or o["descrizione"]


EDITORIAL_GROUPS_PATH = ROOT / "config" / "editorial_groups.json"


def load_editorial_groups() -> dict:
    """testata -> gruppo editoriale, da config/editorial_groups.json. Copertura verificata
    il 21/08: 4 gruppi su ~85 testate — una testata assente vale come gruppo di se stessa
    (vedi gruppo_editoriale_di), quindi max_per_gruppo_editoriale oggi agisce solo su queste
    testate mappate, non sulla concentrazione delle fonti in generale."""
    d = json.load(open(EDITORIAL_GROUPS_PATH, encoding="utf-8"))
    testata_to_gruppo = {}
    for gruppo, testate in d["gruppi"].items():
        for t in testate:
            testata_to_gruppo[t] = gruppo
    return testata_to_gruppo


def gruppo_editoriale_di(testata: str, groups_map: dict) -> str:
    return groups_map.get(testata, testata)


def select_for_newsletter(candidates: list[dict], config: dict, already_selected_urls: set) -> tuple:
    """
    Compositore: SELEZIONA soltanto, non riordina e non rivaluta la pertinenza — l'ordine
    interno resta sempre quello di 'rank' (kNN). Riceve TUTTI gli articoli visti dal gate
    (Si e No, non solo gli approvati): serve per il ripescaggio, il primo passo di recupero
    slot, prima di allentare qualunque vincolo.

    candidates: dict con almeno title, rank, pertinenza_llm ('Si'/'No'), fonte, content_type
    (None o 'non_classificato' ammessi), url_canonico, gate_confidence (opzionale — se assente
    il ripescaggio usa il rank come proxy di "bocciato per poco", meno preciso ma unico dato
    disponibile sui run storici senza confidence persistita).
    config: n_articoli_totali, max_per_testata, max_per_gruppo_editoriale,
    max_per_content_type (dict per slug, categorie assenti = illimitate), ordine_allentamento
    (lista, "ripescaggio" per primo se presente), max_round_allentamento.
    already_selected_urls: url_canonico già in store con selected_for_newsletter=true — MAI
    allentato, esclusione sempre valida.

    Ritorna (selezionati, esclusi, log_allentamento). Ogni selezionato ha "_via" (gate-si /
    ripescato / round1.../) e "_gruppo". Ogni escluso ha "motivo_esclusione".
    """
    groups_map = load_editorial_groups()
    n_tot = config["n_articoli_totali"]
    max_ct_base = config["max_per_content_type"]
    max_round = config.get("max_round_allentamento", 2)
    ordine = config.get("ordine_allentamento", ["ripescaggio", "max_per_testata",
                                                  "max_per_gruppo_editoriale", "max_per_content_type"])

    eligible = [c for c in candidates if c.get("url_canonico") not in already_selected_urls]
    esclusi_pubblicati = [c for c in candidates if c.get("url_canonico") in already_selected_urls]

    si_pool = sorted([c for c in eligible if c.get("pertinenza_llm") == "Si"], key=lambda c: c["rank"])
    # Ripescaggio ordinato per confidence crescente (il "No" meno sicuro per primo — bocciato
    # per poco); se gate_confidence manca, ricade sul rank come proxy meno preciso.
    no_pool = sorted([c for c in eligible if c.get("pertinenza_llm") != "Si"],
                      key=lambda c: c["gate_confidence"] if c.get("gate_confidence") is not None else c["rank"])

    selected: list[dict] = []
    reason_by_title: dict = {}
    cnt_testata, cnt_gruppo, cnt_ct = Counter(), Counter(), Counter()

    def try_fill(pool, cap_testata, cap_gruppo, cap_ct, via):
        for c in pool:
            if any(s["title"] == c["title"] for s in selected):
                continue
            if len(selected) >= n_tot:
                reason_by_title[c["title"]] = "slot pieni (nessun vincolo violato, semplicemente non è arrivato in tempo)"
                continue
            fonte = c.get("fonte", "")
            gruppo = gruppo_editoriale_di(fonte, groups_map)
            ct = c.get("content_type") or "non_classificato"
            if cnt_testata[fonte] >= cap_testata:
                reason_by_title[c["title"]] = f"max_per_testata raggiunto ({fonte}, {cnt_testata[fonte]}/{cap_testata})"
                continue
            if cnt_gruppo[gruppo] >= cap_gruppo:
                reason_by_title[c["title"]] = f"max_per_gruppo_editoriale raggiunto ({gruppo}, {cnt_gruppo[gruppo]}/{cap_gruppo})"
                continue
            if ct != "non_classificato" and cnt_ct[ct] >= cap_ct.get(ct, 10 ** 9):
                reason_by_title[c["title"]] = f"max_per_content_type raggiunto ({ct}, {cnt_ct[ct]}/{cap_ct.get(ct)})"
                continue
            selected.append({**c, "_via": via, "_gruppo": gruppo})
            reason_by_title.pop(c["title"], None)
            cnt_testata[fonte] += 1
            cnt_gruppo[gruppo] += 1
            cnt_ct[ct] += 1

    log: list[str] = []
    cur_testata = config["max_per_testata"]
    cur_gruppo = config["max_per_gruppo_editoriale"]
    cur_ct = dict(max_ct_base)

    try_fill(si_pool, cur_testata, cur_gruppo, cur_ct, "gate-si")
    if len(selected) >= n_tot:
        log.append("pieno senza allentare nulla")

    if len(selected) < n_tot and "ripescaggio" in ordine:
        before = len(selected)
        try_fill(no_pool, cur_testata, cur_gruppo, cur_ct, "ripescato")
        gained = len(selected) - before
        log.append(f"ripescaggio: {gained} articoli bocciati dal gate recuperati (nessun vincolo allentato)"
                    if gained else "ripescaggio tentato: nessun articolo bocciato recuperabile senza allentare vincoli")

    round_vincoli = [v for v in ordine if v != "ripescaggio"]
    for i, vincolo in enumerate(round_vincoli[:max_round], start=1):
        if len(selected) >= n_tot:
            break
        before = len(selected)
        if vincolo == "max_per_testata":
            cur_testata += 1
        elif vincolo == "max_per_gruppo_editoriale":
            cur_gruppo += 1
        elif vincolo == "max_per_content_type":
            cur_ct = {k: v + 1 for k, v in cur_ct.items()}
        try_fill(si_pool + no_pool, cur_testata, cur_gruppo, cur_ct, f"round{i}:{vincolo}")
        gained = len(selected) - before
        log.append(f"round {i} — allentato {vincolo} (+1): recuperati {gained} articoli" if gained
                   else f"round {i} — allentato {vincolo} (+1): nessun articolo recuperabile")

    if len(selected) < n_tot:
        log.append(f"BLOCCATO: {n_tot - len(selected)} slot restano vuoti dopo {max_round} round di "
                   f"allentamento — candidati eleggibili insufficienti in questo run, non un errore del compositore")

    selected_titles = {s["title"] for s in selected}
    esclusi = [{**c, "motivo_esclusione": reason_by_title.get(
                    c["title"], "mai valutato: gli slot erano già pieni prima del suo turno")}
               for c in eligible if c["title"] not in selected_titles]
    esclusi += [{**c, "motivo_esclusione": "già pubblicato (selected_for_newsletter in store)"}
                for c in esclusi_pubblicati]

    return selected, esclusi, log
