"""
CRUD operations on data/training_data.csv — the central RLHF dataset.

Key rules:
- article_id = sha1(link)[:12] — deduplication key
- LLM-generated rows land with reviewed=False, source="llm"
- Only reviewed=True rows are used for training
- Never overwrite a row that already has reviewed=True
"""
import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TRAINING_CSV = DATA_DIR / "training_data.csv"

SCHEMA = [
    "article_id",
    "title",
    "link",
    "published_date",
    "feed_name",
    "summary_clean",
    "text_for_embed",
    # labels
    "observatories",
    "content_type",
    "target_audience",
    "macro_topics",
    # rlhf metadata
    "source",
    "reviewed",
    "review_timestamp",
    "reviewer_notes",
    # confidence scores
    "conf_content_type",
    "conf_target_audience",
    "conf_observatories",
    "conf_macro_topics",
    "conf_overall",
    # infra
    "embedding_model",
]


def _article_id(link: str) -> str:
    return hashlib.sha1(link.encode()).hexdigest()[:12]


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _ensure_csv() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not TRAINING_CSV.exists():
        with open(TRAINING_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA)
            writer.writeheader()


def load_all() -> list[dict]:
    _ensure_csv()
    with open(TRAINING_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_index() -> dict[str, dict]:
    """Returns {article_id: row} map."""
    return {row["article_id"]: row for row in load_all()}


def append_article(
    article: dict,
    classification: dict,
    source: str = "llm",
    confidence: Optional[dict] = None,
    embedding_model: str = "text-embedding-3-small",
) -> bool:
    """
    Append article + classification to training_data.csv.
    Returns True if written, False if article_id already exists.
    Skips silently if existing row has reviewed=True (never overwrite human labels).
    """
    _ensure_csv()

    aid = _article_id(article["link"])
    index = _load_index()

    if aid in index:
        existing = index[aid]
        if existing.get("reviewed", "False") == "True":
            return False
        # unreviewed duplicate: skip (don't re-add)
        return False

    summary_clean = _strip_html(article.get("summary", ""))
    text_for_embed = f"{article['title']}. {summary_clean}"

    conf = confidence or {}
    conf_ct = conf.get("content_type", "")
    conf_ta = conf.get("target_audience", "")
    conf_obs = conf.get("observatories", "")
    conf_mt = conf.get("macro_topics", "")
    # overall = min of non-empty scores
    numeric = [v for v in [conf_ct, conf_ta, conf_obs, conf_mt] if isinstance(v, float)]
    conf_overall = round(min(numeric), 4) if numeric else ""

    obs = classification.get("observatories", [])
    mt = classification.get("macro_topics", [])

    row = {
        "article_id": aid,
        "title": article["title"],
        "link": article["link"],
        "published_date": article.get("published_date", ""),
        "feed_name": article.get("feed_name", ""),
        "summary_clean": summary_clean,
        "text_for_embed": text_for_embed,
        "observatories": "|".join(obs) if isinstance(obs, list) else (obs or ""),
        "content_type": classification.get("content_type", ""),
        "target_audience": classification.get("target_audience", ""),
        "macro_topics": "|".join(mt) if isinstance(mt, list) else (mt or ""),
        "source": source,
        "reviewed": "False",
        "review_timestamp": "",
        "reviewer_notes": "",
        "conf_content_type": conf_ct,
        "conf_target_audience": conf_ta,
        "conf_observatories": conf_obs,
        "conf_macro_topics": conf_mt,
        "conf_overall": conf_overall,
        "embedding_model": embedding_model,
    }

    with open(TRAINING_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writerow(row)

    return True


def mark_reviewed(
    article_id: str,
    labels: dict,
    reviewer_notes: str = "",
) -> bool:
    """
    Update labels for an article and mark as reviewed=True.
    labels: {observatories, content_type, target_audience, macro_topics}
    Returns False if article_id not found.
    """
    rows = load_all()
    found = False

    for row in rows:
        if row["article_id"] == article_id:
            found = True
            obs = labels.get("observatories", row["observatories"])
            mt = labels.get("macro_topics", row["macro_topics"])
            row["observatories"] = "|".join(obs) if isinstance(obs, list) else obs
            row["content_type"] = labels.get("content_type", row["content_type"])
            row["target_audience"] = labels.get("target_audience", row["target_audience"])
            row["macro_topics"] = "|".join(mt) if isinstance(mt, list) else mt
            # preserve original labels in source
            if row.get("source") == "llm":
                row["source"] = "human_corrected_llm"
            row["reviewed"] = "True"
            row["review_timestamp"] = datetime.now(timezone.utc).isoformat()
            row["reviewer_notes"] = reviewer_notes
            break

    if not found:
        return False

    _write_all(rows)
    return True


def confirm_reviewed(article_id: str) -> bool:
    """Mark article as reviewed without changing its labels (human confirms LLM labels)."""
    rows = load_all()
    found = False

    for row in rows:
        if row["article_id"] == article_id:
            found = True
            row["source"] = "human"
            row["reviewed"] = "True"
            row["review_timestamp"] = datetime.now(timezone.utc).isoformat()
            break

    if not found:
        return False

    _write_all(rows)
    return True


def bulk_confirm(min_conf: float = 0.0, source_filter: str = "llm", require_labels: bool = True) -> int:
    """Mark as reviewed all unreviewed articles from a given source.

    Args:
        min_conf: minimum conf_overall (skip if field is empty when > 0)
        source_filter: only confirm articles with this source
        require_labels: skip articles with empty content_type or target_audience

    Returns the number of articles confirmed.
    """
    rows = load_all()
    count = 0
    for row in rows:
        if row.get("reviewed", "False") == "True":
            continue
        if row.get("source", "") != source_filter:
            continue
        if require_labels and (not row.get("content_type", "").strip() or not row.get("target_audience", "").strip()):
            continue
        if min_conf > 0:
            try:
                conf = float(row.get("conf_overall", 0) or 0)
            except ValueError:
                conf = 0.0
            if conf < min_conf:
                continue
        row["reviewed"] = "True"
        row["source"] = "human"
        row["review_timestamp"] = datetime.now(timezone.utc).isoformat()
        count += 1
    if count:
        _write_all(rows)
    return count


def get_review_queue(
    feed_name: Optional[str] = None,
    date_from: Optional[str] = None,
    observatory: Optional[str] = None,
) -> list[dict]:
    """Return rows with reviewed=False, optionally filtered."""
    rows = load_all()
    queue = [r for r in rows if r.get("reviewed", "False") == "False"]
    if feed_name:
        queue = [r for r in queue if r.get("feed_name") == feed_name]
    if date_from:
        queue = [r for r in queue if r.get("published_date", "") >= date_from]
    if observatory:
        queue = [
            r for r in queue
            if observatory in [s.strip() for s in r.get("observatories", "").split("|") if s.strip()]
        ]
    return queue


def get_training_set(field: Optional[str] = None) -> list[dict]:
    """Return rows with reviewed=True. Optionally filter to rows with non-empty label for field."""
    rows = load_all()
    reviewed = [r for r in rows if r.get("reviewed", "False") == "True"]
    if field:
        reviewed = [r for r in reviewed if r.get(field, "").strip()]
    return reviewed


def stats() -> dict:
    rows = load_all()
    total = len(rows)
    reviewed = sum(1 for r in rows if r.get("reviewed") == "True")
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": total - reviewed,
    }


def _write_all(rows: list[dict]) -> None:
    with open(TRAINING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)
