"""
STEP 6 — Pipeline completa sul corpus scrapato: dove trovi il ranking.

Flusso: corpus grezzo -> testo intero -> embedding -> ranker (somiglianza) -> ordina
        -> sospetti duplicati su TUTTO il corpus (stessa notizia, fonti diverse)
        -> gate LLM sui primi N candidati (informato dei sospetti; sotto N solo
           clusterizzazione, niente gate: quello ha un costo in token) -> output

Sorgente articoli: ../corpus/corpus_raw.jsonl — CONDIVISO tra tutti gli osservatori,
non viene rigenerato qui (vedi corpus/fetch_corpus.py per aggiornarlo).
Uso:
  python startup_thinking_relevance/06_rank.py                 # rank del corpus, LLM sui primi 150
  python startup_thinking_relevance/06_rank.py --topn 100
"""
import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import (log, embed_texts_cached, fetch_fulltext, format_date, detect_lang,
                    canonicalize_url, load_tracking_params, dedup_by_content_hash, content_hash,
                    upsert_articles_store)
from ranker import build_lang_pools, score_similarity_multilang, find_similar_clusters, CLUSTER_SIM_THRESHOLD
import importlib
sys.path.insert(0, str(HERE))
gate = importlib.import_module("05_llm_gate")

DATA = HERE / "data"
MODELS = HERE / "models"


def load_corpus():
    rows = [json.loads(l) for l in open(ROOT / "corpus" / "corpus_raw.jsonl", encoding="utf-8")]
    for r in rows:
        r["fonte"] = r.get("feed_name", "")

    # Esclude gli articoli GIA' nel dataset di riferimento: altrimenti si confronterebbero
    # con se stessi nel ranker (coseno ~1) e il punteggio verrebbe gonfiato (se "Si") o
    # sgonfiato (se "No") artificialmente. Confronto per URL CANONICALIZZATO, non stringa
    # esatta: un link con un tag di tracking diverso da quello salvato nel dataset di
    # riferimento altrimenti sfuggirebbe all'esclusione (vedi common.canonicalize_url).
    tracking_params = load_tracking_params()
    ref_rows = [json.loads(l) for l in open(DATA / "dataset.jsonl", encoding="utf-8")]
    ref_links = {canonicalize_url(r["link"], tracking_params) for r in ref_rows if r.get("link")}
    before = len(rows)
    rows = [r for r in rows if canonicalize_url(r["link"], tracking_params) not in ref_links]
    n_excluded = before - len(rows)
    log(f"  esclusi {n_excluded} articoli già nel dataset di riferimento (confronto con se stessi)")

    # contatore persistito (non solo loggato) — aggiornato in place da run successivi che
    # aggiungono altri contatori allo stesso file (vedi dedup_by_content_hash in main()).
    stats_path = DATA / "dedup_stats.json"
    stats = json.load(open(stats_path, encoding="utf-8")) if stats_path.exists() else {}
    stats["esclusi_gia_in_reference_set"] = n_excluded
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=150, help="quanti candidati passare all'LLM")
    args = ap.parse_args()

    log("STEP 6 — pipeline completa sul corpus startup-thinking")
    arts = load_corpus()
    log(f"  articoli in ingresso: {len(arts)}")

    # 1) testo intero — con CACHE su disco
    cache_path = MODELS / "corpus_fulltext_cache.json"
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    log(f"  1) testo intero — {len(cache)} già in cache, scarico solo i mancanti...")

    # Chiave della cache: URL canonicalizzato, non più il titolo — il titolo NON è una
    # chiave di identità affidabile (audit 13/08: 4 casi reali di stesso titolo + stessa
    # testata ma articoli diversi, es. rubriche ricorrenti con episodi diversi: con la
    # cache a titolo uno di questi riceveva silenziosamente il testo dell'altro).
    # MIGRAZIONE: le cache esistenti sono ancora chiavizzate per titolo — al primo run
    # dopo questa modifica nessuna voce fa match, quindi si riscarica il testo intero di
    # ogni articolo una volta sola (costo di rete, non di token: fetch_fulltext non
    # chiama nessuna API a pagamento).
    tracking_params = load_tracking_params()

    def f(a):
        key = canonicalize_url(a["link"], tracking_params)
        c = cache.get(key)
        if c:
            a["full_text"], a["text_source"] = c["text"], c["source"]
        else:
            a["full_text"], a["text_source"] = fetch_fulltext(a["link"], a["summary"])
        return a
    with ThreadPoolExecutor(max_workers=16) as ex:
        arts = list(ex.map(f, arts))
    for a in arts:
        cache[canonicalize_url(a["link"], tracking_params)] = {"text": a["full_text"], "source": a["text_source"]}
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    nfull = sum(1 for a in arts if a["text_source"] == "full")
    log(f"     testo intero: {nfull}/{len(arts)}  (cache: {cache_path.name})")

    # 1b) dedup ESATTA per hash del contenuto normalizzato, prima dell'embedding — solo
    # sugli articoli con testo intero reale (vedi dedup_by_content_hash: gli articoli in
    # fallback sul summary non vengono toccati, per non fondere rubriche ricorrenti con
    # summary templato ma contenuto diverso).
    before_hash = len(arts)
    arts, hash_stats = dedup_by_content_hash(arts)
    log(f"     dedup per hash contenuto: scartati {hash_stats['scartati_duplicato_hash']} duplicati esatti "
        f"su {before_hash} (saltati dall'hash {hash_stats['saltati_fallback']} in fallback summary, non toccati)")
    stats_path = DATA / "dedup_stats.json"
    stats = json.load(open(stats_path, encoding="utf-8")) if stats_path.exists() else {}
    stats.update(hash_stats)
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 2) embedding
    log("  2) embedding (cache condivisa store/embeddings.npz)...")
    X, tok, embed_stats = embed_texts_cached(arts, tracking_params)
    log(f"     vettori {X.shape}, token spesi {tok}  |  cache: {embed_stats['n_cache_hit']} hit, "
        f"{embed_stats['n_nuovi']} nuovi, {embed_stats['potati']} potati per età")

    # 3) ranker — somiglianza agli esempi (NIENTE addestramento), per lingua
    log("  3) punteggio per somiglianza agli esempi (reference_set.npz)...")
    ref = np.load(MODELS / "reference_set.npz", allow_pickle=True)
    pools = build_lang_pools(ref["X"], ref["y"], ref["langs"])
    log("     pool di riferimento per lingua: " + ", ".join(f"{l}={len(p[1])}" for l, p in pools.items()))
    arts_langs = np.array([detect_lang(f"{a['title']}. {a['full_text']}") for a in arts])
    for l, n in Counter(arts_langs).items():
        log(f"     articoli in '{l}': {n}")
    scores = score_similarity_multilang(X, arts_langs, pools, fallback_lang="it", k=3)
    order = np.argsort(-scores)

    # 4) sospetti duplicati — su TUTTO il corpus classificato, non solo sui candidati
    # inviati al gate: costa niente in più (stessi embedding del passo 2, solo aritmetica),
    # e così si vede anche se un candidato del gate ha altre fonti duplicate più giù in
    # classifica (mai gated per costo, ma comunque segnalate). Il gate LLM resta invece
    # limitato ai primi N (quello è il passo con un costo reale in token).
    cluster_ids = find_similar_clusters(X)
    for a, cid in zip(arts, cluster_ids):
        a["cluster_id"] = cid
    EXCERPT_LEN = 240  # breve estratto per peer nel prompt del gate — non il cap di 6000
    # usato per il corpo dell'articolo principale: qui basta capire "è la stessa notizia?",
    # e questo estratto si somma al costo di OGNI candidato gated con 2+ peer nello stesso
    # cluster, quindi resta corto apposta.
    by_cluster = {}
    for a in arts:
        if a["cluster_id"] is not None:
            by_cluster.setdefault(a["cluster_id"], []).append(a)
    for a in arts:
        a["cluster_peers"] = [
            {"title": p["title"], "excerpt": (p.get("full_text") or "")[:EXCERPT_LEN]}
            for p in by_cluster.get(a["cluster_id"], []) if p["title"] != a["title"]
        ]
    n_clustered = sum(1 for a in arts if a["cluster_id"] is not None)
    log(f"     sospetti duplicati: {n_clustered}/{len(arts)} articoli in {len(by_cluster)} cluster (soglia coseno {CLUSTER_SIM_THRESHOLD})")

    # 5) gate LLM sui primi N
    log(f"  5) gate LLM sui primi {args.topn}...")
    topn = [arts[i] for i in order[:args.topn]]
    topn, gtok = gate.run_gate(topn)
    llm_map = {a["title"]: a.get("llm") for a in topn}
    confidence_map = {a["title"]: a.get("llm_confidence") for a in topn}
    motivo_map = {a["title"]: a.get("llm_motivo", "") for a in topn}
    cluster_map = {a["title"]: a.get("cluster_id") for a in arts}
    si = sum(1 for a in topn if a.get("llm") == 1)
    log(f"     LLM-Si tra i primi {args.topn}: {si}  |  token {gtok['in']} in + {gtok['out']} out")

    # 6) output — CSV
    out_path = DATA / "ranking.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank", "score_ranker", "pertinenza_llm", "confidence_llm", "parere_revisore", "titolo", "fonte", "data", "link", "cluster_id"])
        for rank, i in enumerate(order, 1):
            a = arts[i]
            pl = llm_map.get(a["title"])
            conf = confidence_map.get(a["title"])
            cid = cluster_map.get(a["title"])
            w.writerow([rank, round(float(scores[i]), 4),
                        ("Si" if pl == 1 else "No") if pl is not None else "",
                        round(float(conf), 2) if conf is not None else "",
                        motivo_map.get(a["title"], ""),
                        a["title"], a["fonte"], format_date(a.get("published_date", "")), a["link"],
                        cid if cid is not None else ""])
    log(f"\n  salvato: {out_path}")

    # 7) store persistente — SOLO scrittura additiva a valle di ranking.csv (già salvato
    # sopra, con gli stessi identici arts/order/scores): non altera in nessun modo il
    # ranking prodotto, registra solo cosa è stato visto per i run futuri (riuso embedding,
    # "non riproporre", sviluppi successivi di una storia — vedi common.upsert_articles_store).
    records = []
    for rank, i in enumerate(order, 1):
        a = arts[i]
        records.append({
            "url_canonico": canonicalize_url(a["link"], tracking_params),
            "osservatorio": "startup",
            "content_hash": content_hash(a),
            "published_date": a.get("published_date", ""),
            "text_source": a.get("text_source"),
            "last_rank": rank,
        })
    store_stats = upsert_articles_store(records)
    log(f"  store articoli: {store_stats['nuove']} nuove, {store_stats['aggiornate']} aggiornate, "
        f"{store_stats['potate']} potate per età, {store_stats['totale_dopo']} totali in store/articles.jsonl")

    log(f"\n  ROSA FINALE (ranker top-{args.topn} confermati dall'LLM): {si} articoli")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
