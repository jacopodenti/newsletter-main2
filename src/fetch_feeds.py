import feedparser
import requests
import yaml
from pathlib import Path
from datetime import datetime

# Stesso User-Agent del fetch del testo intero (ocx_relevance/common.py:UA) — prima il feed
# si presentava con lo user-agent di default di feedparser (identificabile come bot) mentre
# il fetch del testo si presenta come Chrome: un'incoerenza che può causare blocchi lato server
# su feed altrimenti raggiungibili.
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
TIMEOUT_S = 15
MAX_RETRIES = 1  # un solo retry sui fallimenti di rete, non parallelizzato (loop resta sequenziale)


def load_feeds(config_path: str = "config/feeds.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f)["feeds"]


def _fetch_parsed(url: str):
    """Scarica e parsa un feed con UA esplicito, timeout e un retry sui fallimenti di rete
    (connessione, DNS, timeout, status HTTP di errore). Se anche il retry fallisce, ritorna
    un feed vuoto con bozo=True e l'eccezione di rete come bozo_exception, così un fallimento
    di rete è indistinguibile da un feed malformato solo nel messaggio, non nel flag."""
    last_exc = None
    for _ in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=UA, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return feedparser.parse(resp.content)
        except Exception as e:
            last_exc = e
    parsed = feedparser.parse(b"")
    parsed["bozo"] = True
    parsed["bozo_exception"] = last_exc
    return parsed


def _parse_entries(parsed, name: str, max_per_feed: int):
    """Estrae gli articoli da un feed già scaricato/parsato e la diagnostica per quel feed:
    entry restituite, flag bozo (feed malformato/irraggiungibile per feedparser) e relativo
    errore, quante entry hanno link+titolo validi."""
    diag = {
        "feed_name": name,
        "n_entries_raw": len(parsed.entries),
        "bozo": bool(getattr(parsed, "bozo", 0)),
        "bozo_error": str(parsed.get("bozo_exception", "")) if getattr(parsed, "bozo", 0) else None,
        "n_link_titolo_validi": 0,
        "n_link_vuoto": 0,
    }

    articles = []
    for entry in parsed.entries[:max_per_feed]:
        link = entry.get("link", "")
        title = entry.get("title", "")
        if not link:
            diag["n_link_vuoto"] += 1
        elif title:
            diag["n_link_titolo_validi"] += 1

        published = entry.get("published") or entry.get("updated") or ""
        summary = (
            entry.get("summary")
            or entry.get("content", [{}])[0].get("value", "")
            or ""
        )

        articles.append({
            "title": title,
            "link": link,
            "published_date": published,
            "summary": summary,
            "feed_name": name,
        })

    return articles, diag


def fetch_all(config_path: str = "config/feeds.yaml", max_per_feed: int = 10) -> list[dict]:
    feeds = load_feeds(config_path)
    articles = []

    for feed_config in feeds:
        url = feed_config["url"]
        name = feed_config.get("name", url)
        parsed = _fetch_parsed(url)
        arts, _ = _parse_entries(parsed, name, max_per_feed)
        articles.extend(arts)

    return articles


def fetch_all_with_diagnostics(config_path: str = "config/feeds.yaml", max_per_feed: int = 10):
    """Come fetch_all, ma ritorna anche (articles, diagnostics): un dict di diagnostica per
    OGNI feed configurato, incluso un feed che restituisce zero entry — non sparisce dal
    conteggio, compare esplicitamente con n_entries_raw=0."""
    feeds = load_feeds(config_path)
    articles = []
    diagnostics = []

    for feed_config in feeds:
        url = feed_config["url"]
        name = feed_config.get("name", url)
        parsed = _fetch_parsed(url)
        arts, diag = _parse_entries(parsed, name, max_per_feed)
        articles.extend(arts)
        diagnostics.append(diag)

    return articles, diagnostics
