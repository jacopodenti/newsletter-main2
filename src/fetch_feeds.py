import feedparser
import yaml
from pathlib import Path
from datetime import datetime


def load_feeds(config_path: str = "config/feeds.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f)["feeds"]


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
        parsed = feedparser.parse(url)
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
        parsed = feedparser.parse(url)
        arts, diag = _parse_entries(parsed, name, max_per_feed)
        articles.extend(arts)
        diagnostics.append(diag)

    return articles, diagnostics
