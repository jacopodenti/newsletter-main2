"""
Scarica il corpus grezzo da tutti i feed di feeds.yaml — CONDIVISO tra tutti
gli osservatori (ocx, ai, startup-thinking, quelli futuri). Non è specifico di
un osservatorio: va rilanciato solo per aggiornare il corpus (es. una volta
ogni tanto), non ad ogni test di un nuovo osservatorio.

Filtra anche gli articoli troppo vecchi per una newsletter (fonti come ANSA
Tecnologia, MIT News, Engage Customer a volte restituiscono voci ferme a
mesi/anni fa — vedi controllo del 22/07): tiene solo quelli pubblicati negli
ultimi 30 giorni rispetto ad oggi. Le date non interpretabili vengono scartate
per sicurezza (meglio perdere un articolo che tenerne uno di età sconosciuta).

Dedup per URL CANONICALIZZATO (common.canonicalize_url), non per stringa esatta:
alcuni feed ripubblicano lo stesso articolo con parametri di tracking diversi
(utm_*, at_medium/at_campaign — vedi audit del 13/08) che una stringa esatta non
coglierebbe. Parametri da rimuovere in corpus/tracking_params.yaml, non hardcoded.

Riusa fetch_feeds.fetch_all() da ../src e canonicalize_url/load_tracking_params
da ../ocx_relevance/common.py (unico punto condiviso con le 3 pipeline).

Output: corpus_raw.jsonl (un articolo per riga: title, link, published_date, summary, feed_name)
        fetch_stats.json (contatori dell'ultimo run — persistiti, non solo loggati)
Uso:    python corpus/fetch_corpus.py
"""
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ocx_relevance"))

from fetch_feeds import fetch_all
from common import canonicalize_url, load_tracking_params

DATA = HERE

MAX_PER_FEED = 50
MAX_AGE_DAYS = 30


def parse_date(raw):
    """Ritorna un datetime timezone-aware, o None se non interpretabile."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def main():
    feeds_path = HERE / "feeds.yaml"
    print(f"STEP 1 — scraping di tutti i feed in {feeds_path.name} (max {MAX_PER_FEED}/feed)")

    articles = fetch_all(str(feeds_path), max_per_feed=MAX_PER_FEED)

    counts = Counter(a["feed_name"] for a in articles)
    n_feeds_total = len(counts)

    # dedup per URL CANONICALIZZATO (alcuni feed ripubblicano lo stesso articolo con
    # parametri di tracking diversi — una stringa esatta non li riconoscerebbe come
    # duplicati). Sostituisce il vecchio dedup per stringa esatta: canonicalizzare un link
    # già identico non lo cambia, quindi ogni duplicato esatto è comunque catturato.
    tracking_params = load_tracking_params()
    seen = set()
    deduped = []
    n_dropped_duplicate_url = 0
    for a in articles:
        canon = canonicalize_url(a["link"], tracking_params)
        if canon and canon in seen:
            n_dropped_duplicate_url += 1
            continue
        seen.add(canon)
        deduped.append(a)

    # --- CONTROLLO ETA' — stampa le statistiche prima di pulire ---
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MAX_AGE_DAYS)

    years = Counter()
    unparseable = 0
    for a in deduped:
        dt = parse_date(a.get("published_date", ""))
        if dt is None:
            unparseable += 1
        else:
            years[dt.year] += 1

    print(f"\n  [STATISTICHE ETA' — prima della pulizia]")
    print(f"  oggi: {now:%Y-%m-%d}  |  soglia (ultimi {MAX_AGE_DAYS} giorni): {cutoff:%Y-%m-%d}")
    print(f"  totale dopo dedup: {len(deduped)}")
    for y in sorted(years):
        print(f"    {y}: {years[y]}")
    print(f"    data non interpretabile: {unparseable}")

    # --- pulizia: tieni solo articoli con data valida, entro la soglia, anno corrente ---
    kept, dropped_old, dropped_unparseable = [], 0, 0
    dropped_by_feed = Counter()
    for a in deduped:
        dt = parse_date(a.get("published_date", ""))
        if dt is None:
            dropped_unparseable += 1
            dropped_by_feed[a["feed_name"]] += 1
            continue
        if dt < cutoff or dt.year != now.year:
            dropped_old += 1
            dropped_by_feed[a["feed_name"]] += 1
            continue
        kept.append(a)

    print(f"\n  [PULIZIA]")
    print(f"  scartati per data troppo vecchia (>{MAX_AGE_DAYS}gg o anno != {now.year}): {dropped_old}")
    print(f"  scartati per data non interpretabile: {dropped_unparseable}")
    print(f"  totale scartati: {dropped_old + dropped_unparseable}")
    print(f"  rimasti (freschi, ultimi {MAX_AGE_DAYS} giorni): {len(kept)}")
    if dropped_by_feed:
        print(f"\n  scarti per fonte (prime 10):")
        for name, n in dropped_by_feed.most_common(10):
            print(f"    {name:35s} {n}")

    out_path = DATA / "corpus_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for a in kept:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"\n  feed con almeno un articolo (prima della pulizia): {n_feeds_total}")
    print(f"  per feed, grezzo (primi 15 per volume):")
    for name, n in counts.most_common(15):
        print(f"    {name:35s} {n}")

    # contatori persistiti (non solo loggati a runtime) — anche la data non interpretabile,
    # che finora si perdeva a fine run.
    stats = {
        "feed_totali": n_feeds_total,
        "articoli_grezzi": len(articles),
        "scartati_duplicato_url": n_dropped_duplicate_url,
        "scartati_data_non_interpretabile": dropped_unparseable,
        "scartati_data_troppo_vecchia": dropped_old,
        "rimasti": len(kept),
    }
    stats_path = DATA / "fetch_stats.json"
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n  salvato: {out_path}  ({len(kept)} articoli freschi)")
    print(f"  salvato: {stats_path}")
    print("\nFATTO.")


if __name__ == "__main__":
    main()
