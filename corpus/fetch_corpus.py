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

Riusa fetch_feeds.fetch_all_with_diagnostics() da ../src e canonicalize_url/load_tracking_params
da ../ocx_relevance/common.py (unico punto condiviso con le 3 pipeline).

Output (nome derivato da --out, default corpus_raw.jsonl — il corpus di RIFERIMENTO):
    <out>.jsonl              un articolo per riga: title, link, published_date, summary, feed_name
    <out>_stats.json         contatori aggregati dell'ultimo run
    <out>_feed_report.json   {"riepilogo": feed problematici raggruppati per causa (zero entry
                              per errore HTTP, zero entry per XML non valido, feed sano ma
                              contenuto oltre la finestra temporale, feed sano ma full-text
                              sempre in fallback), "feeds": diagnostica per singolo feed}

Uso:
    python corpus/fetch_corpus.py                                  # aggiorna il corpus di riferimento
    python corpus/fetch_corpus.py --out corpus/corpus_raw_test.jsonl [--diagnose-fulltext]
        # run di test (es. per verificare un feed appena aggiunto): NON tocca il riferimento.
        # --diagnose-fulltext prova anche lo scraping del testo intero per ogni articolo
        # (una richiesta HTTP per articolo: più lento, ma serve a distinguere un feed rotto
        # da uno scraping del testo che va in fallback).
"""
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ocx_relevance"))

from fetch_feeds import fetch_all_with_diagnostics
from common import canonicalize_url, load_tracking_params, fetch_fulltext

DATA = HERE
REFERENCE_OUT = DATA / "corpus_raw.jsonl"

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


def _causa_zero_entry(bozo_error):
    """Distingue la causa di un feed a zero entry dal messaggio d'errore. bozo da solo non
    basta (vedi nota in fetch_feeds._parse_entries: bozo è un segnale, non un criterio di
    scarto): un HTTPError di requests.raise_for_status ha un formato riconoscibile ("403
    Client Error: ..."), un errore di parsing XML di feedparser (SAXParseException) inizia
    sempre con "<unknown>:"."""
    if not bozo_error:
        return "zero_entry_senza_errore"
    if bozo_error.startswith("<unknown>:"):
        return "zero_entry_xml_non_valido"
    if re.match(r"^\d{3} (Client|Server) Error", bozo_error):
        return "zero_entry_errore_http"
    return "zero_entry_altro_errore_rete"


def build_riepilogo(report, diagnose_fulltext):
    """Raggruppa i feed problematici per causa, così ad ogni run si vede subito cosa è
    cambiato senza scorrere tutte le voci del report. Le chiavi sono sempre presenti anche
    a lista vuota, per distinguere "controllato, nessun caso" da "categoria non applicabile"."""
    riepilogo = {
        "zero_entry_errore_http": [],
        "zero_entry_xml_non_valido": [],
        "zero_entry_altro_errore_rete": [],
        "zero_entry_senza_errore": [],
        "sano_ma_oltre_finestra_temporale": [],
        "sano_ma_fulltext_sempre_fallback": [],
    }
    for name, r in report.items():
        if r["n_entries_raw"] == 0:
            riepilogo[_causa_zero_entry(r["bozo_error"])].append(name)
            continue
        if r["n_link_titolo_validi"] > 0 and r["n_rimasti"] == 0 and r["n_scartati_data"] > 0:
            riepilogo["sano_ma_oltre_finestra_temporale"].append(name)
        if diagnose_fulltext and r["n_rimasti"] > 0 and r["n_full_text"] == 0:
            riepilogo["sano_ma_fulltext_sempre_fallback"].append(name)
    if not diagnose_fulltext:
        riepilogo["sano_ma_fulltext_sempre_fallback"] = "non testato in questo run (usa --diagnose-fulltext)"
    return riepilogo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                         help=f"Path di output per il corpus (default: {REFERENCE_OUT}, il corpus di RIFERIMENTO "
                              "su cui è costruita la baseline). Usa un path diverso per i run di test, così non "
                              "sovrascrivi mai il riferimento.")
    parser.add_argument("--diagnose-fulltext", action="store_true",
                         help="In più, prova lo scraping del testo intero per ogni articolo rimasto e conta "
                              "full/fallback per feed (lento: una richiesta HTTP per articolo).")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else REFERENCE_OUT
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    report_path = out_path.with_name(out_path.stem + "_feed_report.json")
    if out_path == REFERENCE_OUT:
        print(f"ATTENZIONE: nessun --out specificato, questo run sovrascriverà il corpus di RIFERIMENTO "
              f"({out_path}). Per un run di test usa --out <path>.")

    feeds_path = HERE / "feeds.yaml"
    print(f"STEP 1 — scraping di tutti i feed in {feeds_path.name} (max {MAX_PER_FEED}/feed)")

    articles, feed_diagnostics = fetch_all_with_diagnostics(str(feeds_path), max_per_feed=MAX_PER_FEED)

    # report per-feed: inizializzato da TUTTI i feed configurati (anche quelli a zero entry,
    # che altrimenti sparirebbero dal conteggio invece di comparire esplicitamente come zero).
    report = {}
    for diag in feed_diagnostics:
        report[diag["feed_name"]] = {
            "n_entries_raw": diag["n_entries_raw"],
            "bozo": diag["bozo"],
            "bozo_error": diag["bozo_error"],
            "n_link_titolo_validi": diag["n_link_titolo_validi"],
            "n_link_vuoto": diag["n_link_vuoto"],
            "n_scartati_duplicato_url": 0,
            "n_scartati_data": 0,
            "n_full_text": None,
            "n_fallback": None,
        }

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
            report[a["feed_name"]]["n_scartati_duplicato_url"] += 1
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
            report[a["feed_name"]]["n_scartati_data"] += 1
            continue
        if dt < cutoff or dt.year != now.year:
            dropped_old += 1
            dropped_by_feed[a["feed_name"]] += 1
            report[a["feed_name"]]["n_scartati_data"] += 1
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

    with open(out_path, "w", encoding="utf-8") as f:
        for a in kept:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"\n  feed con almeno un articolo (prima della pulizia): {n_feeds_total}")
    print(f"  per feed, grezzo (primi 15 per volume):")
    for name, n in counts.most_common(15):
        print(f"    {name:35s} {n}")

    # --- diagnostica full-text (opzionale: una richiesta HTTP per articolo, lento) ---
    # Distingue un feed rotto (zero entry/bozo, già visibile sopra) da uno scraping del
    # testo che va in fallback: il testo NON viene salvato in corpus_raw.jsonl (che resta
    # title/link/published_date/summary/feed_name), serve solo a contare full vs fallback.
    if args.diagnose_fulltext:
        from concurrent.futures import ThreadPoolExecutor
        print(f"\n  [DIAGNOSTICA FULL-TEXT] scraping di {len(kept)} articoli (16 worker)...")

        def _check(a):
            _, source = fetch_fulltext(a["link"], a["summary"])
            return a["feed_name"], source
        with ThreadPoolExecutor(max_workers=16) as ex:
            for feed_name, source in ex.map(_check, kept):
                if source == "full":
                    report[feed_name]["n_full_text"] = (report[feed_name]["n_full_text"] or 0) + 1
                else:
                    report[feed_name]["n_fallback"] = (report[feed_name]["n_fallback"] or 0) + 1
        # feed con almeno un articolo rimasto ma nessun "full": normalizza a 0, non a None,
        # per distinguerlo da "non testato in questo run".
        for feed_name, r in report.items():
            if r["n_fallback"] is not None and r["n_full_text"] is None:
                r["n_full_text"] = 0
            elif r["n_full_text"] is not None and r["n_fallback"] is None:
                r["n_fallback"] = 0

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
    for r in report.values():
        r["n_rimasti"] = r["n_link_titolo_validi"] - r["n_scartati_duplicato_url"] - r["n_scartati_data"]

    riepilogo = build_riepilogo(report, args.diagnose_fulltext)
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"riepilogo": riepilogo, "feeds": report}, open(report_path, "w", encoding="utf-8"),
               ensure_ascii=False, indent=2)

    n_zero = sum(1 for r in report.values() if r["n_entries_raw"] == 0)
    n_bozo = sum(1 for r in report.values() if r["bozo"])
    print(f"\n  [REPORT PER-FEED] {len(report)} feed configurati — {n_zero} a zero entry, {n_bozo} con bozo=True")
    print(f"  [RIEPILOGO] per causa (vedi {report_path.name} per l'elenco completo):")
    for causa, feed_list in riepilogo.items():
        n = len(feed_list) if isinstance(feed_list, list) else feed_list
        print(f"    {causa:38s} {n}")

    print(f"\n  salvato: {out_path}  ({len(kept)} articoli freschi)")
    print(f"  salvato: {stats_path}")
    print(f"  salvato: {report_path}")
    print("\nFATTO.")


if __name__ == "__main__":
    main()
