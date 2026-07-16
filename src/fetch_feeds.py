import feedparser
import yaml
from pathlib import Path
from datetime import datetime


def load_feeds(config_path: str = "config/feeds.yaml") -> list[dict]:
    with open(config_path) as f:
        return yaml.safe_load(f)["feeds"]


def fetch_all(config_path: str = "config/feeds.yaml", max_per_feed: int = 10) -> list[dict]:
    feeds = load_feeds(config_path)
    articles = []

    for feed_config in feeds:
        url = feed_config["url"]
        name = feed_config.get("name", url)
        parsed = feedparser.parse(url)

        for entry in parsed.entries[:max_per_feed]:
            published = entry.get("published") or entry.get("updated") or ""
            summary = (
                entry.get("summary")
                or entry.get("content", [{}])[0].get("value", "")
                or ""
            )

            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published_date": published,
                "summary": summary,
                "feed_name": name,
            })

    return articles
