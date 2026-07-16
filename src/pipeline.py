import argparse
import asyncio
import csv
import html
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# [MOD] logging INFO per registrare modello/dimensione di embedding allo startup
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("pipeline")

# allow running from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fetch_feeds import fetch_all
from classify import build_client, classify_article, load_osservatori
from embed_classify import build_observatory_embeddings, classify_article_embed, _get_client as get_embed_client, _embed, normalize_text, EMBEDDING_DIM, embedding_model_name
from rlhf_store import append_article
from confidence import classify_with_gate

CSV_COLUMNS = [
    "title",
    "link",
    "published_date",
    "summary",
    "feed_name",
    "observatories",
    "content_type",
    "target_audience",
    "macro_topics",
    "reason",
]

CSV_COLUMNS_IT = [
    "Titolo",
    "Link",
    "Data pubblicazione",
    "Sommario",
    "Fonte",
    "Osservatori",
    "Tipo contenuto",
    "Pubblico target",
    "Macro topic",
    "Motivazione",
]

_IT_MAP = dict(zip(CSV_COLUMNS, CSV_COLUMNS_IT))


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

CSV_COLUMNS_COMPARE = [
    "title",
    "link",
    "published_date",
    "feed_name",
    "llm_observatories",
    "embed_observatories",
    "embed_scores",
    "match",
]


def _compare(llm_obs: list[str], embed_obs: list[str]) -> dict:
    """
    Recall: how many LLM observatories did embed find?
    If LLM returned nothing, we check whether embed also returned nothing.
    """
    llm_set = set(llm_obs)
    embed_set = set(embed_obs)
    hits = llm_set & embed_set

    if not llm_set:
        # LLM found nothing — embed agreement means also nothing, disagreement means embed added
        label = "llm-empty/embed-empty" if not embed_set else "llm-empty/embed-added"
        return {"label": label, "recall": None, "hits": 0, "llm_total": 0}

    recall = len(hits) / len(llm_set)
    if recall == 1.0:
        label = "full"
    elif recall > 0:
        label = "partial"
    else:
        label = "miss"

    return {"label": label, "recall": round(recall, 2), "hits": len(hits), "llm_total": len(llm_set)}


MAX_CONCURRENT = 10  # max parallel requests in flight at once


def run_standard(articles, osservatori, deployment, output_path):
    asyncio.run(_run_standard_async(articles, osservatori, deployment, output_path))


async def _run_standard_async(articles, osservatori, deployment, output_path):
    client = build_client()
    template = None
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()
    total = len(articles)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS_IT)
        writer.writeheader()

        async def process(i, article):
            async with semaphore:
                loop = asyncio.get_event_loop()
                classification = await loop.run_in_executor(
                    None, classify_article, article, client, osservatori, deployment, template
                )
            row_en = {
                **article,
                "observatories": ", ".join(classification["observatories"]),
                "content_type": classification["content_type"] or "",
                "target_audience": ", ".join(classification["target_audience"]) if isinstance(classification["target_audience"], list) else (classification["target_audience"] or ""),
                "macro_topics": ", ".join(classification["macro_topics"]),
                "reason": classification["reason"] or "",
                "summary": _clean_html(article.get("summary", "")),
            }
            async with lock:
                print(f"[{i}/{total}] {article['title'][:80]}")
                writer.writerow({_IT_MAP[k]: v for k, v in row_en.items()})
                csvfile.flush()
                append_article(article, classification)

        await asyncio.gather(*[process(i, article) for i, article in enumerate(articles, 1)])


# [MOD-NUOVA] Step 1.1 + 1.4: legge la dimensione reale dal primo embedding
# osservatorio, la registra a livello INFO con il nome del modello, e verifica
# che coincida con EMBEDDING_DIM (su cui si basa il guard dei classificatori).
def _check_and_log_embedding_dim(obs_embeddings) -> int:
    embedding_dim = len(obs_embeddings[0]["embedding"])
    log.info("Embedding model=%s dimension=%d", embedding_model_name(), embedding_dim)
    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model returned dimension {embedding_dim} but EMBEDDING_DIM={EMBEDDING_DIM}. "
            f"Update EMBEDDING_DIM and delete cached pkl files."
        )
    return embedding_dim


def run_gate(articles, osservatori, deployment, output_path, no_llm=False):
    """Pipeline with confidence gate: uses local models when confident, LLM as fallback."""
    import json
    import numpy as np
    embed_client = get_embed_client()
    llm_client = build_client() if not no_llm else None

    print("Pre-calcolo embedding osservatori...")
    obs_embeddings = build_observatory_embeddings(osservatori, embed_client)
    _check_and_log_embedding_dim(obs_embeddings)  # [MOD] dimension guard + log INFO

    llm_call_count = 0
    model_call_count = 0

    jsonl_path = output_path.with_suffix(".jsonl")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile, \
         open(jsonl_path, "w", encoding="utf-8") as jsonlfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS_IT)
        writer.writeheader()

        for i, article in enumerate(articles, 1):
            print(f"[{i}/{len(articles)}] {article['title'][:80]}")

            # embed article once
            # [MOD] stessa normalize_text() di embed_classify: prima qui si usava
            # _strip_html (niente unescape/lowercasing) -> articolo embeddato due volte
            # con preprocessing divergente. Ora i due embedding coincidono.
            text = normalize_text(f"{article['title']}. {article.get('summary', '')}")
            [article_emb_list] = _embed(embed_client, [text])
            article_emb = np.array(article_emb_list)

            obs_result = classify_article_embed(article, embed_client, obs_embeddings)

            classification = classify_with_gate(
                article=article,
                article_emb=article_emb,
                obs_result=obs_result,
                llm_client=llm_client,
                osservatori=osservatori,
                llm_deployment=deployment,
                no_llm=no_llm,
            )

            meta = classification.get("_meta", {})
            used_llm = any(v == "llm" for v in meta.values())
            if used_llm:
                llm_call_count += 1
            else:
                model_call_count += 1

            source_summary = {k.replace("source_", ""): v for k, v in meta.items() if k.startswith("source_")}
            print(f"  sources: {source_summary}")

            obs = classification.get("observatories", [])
            mt = classification.get("macro_topics") or []
            ta = classification.get("target_audience") or ""

            row_en = {
                **article,
                "observatories": ", ".join(obs),
                "content_type": classification.get("content_type") or "",
                "target_audience": ", ".join(ta) if isinstance(ta, list) else ta,
                "macro_topics": ", ".join(mt),
                "reason": classification.get("reason") or "",
                "summary": _clean_html(article.get("summary", "")),
            }
            writer.writerow({_IT_MAP[k]: v for k, v in row_en.items()})
            csvfile.flush()

            conf = {
                "content_type": meta.get("conf_content_type", ""),
                "target_audience": meta.get("conf_target_audience", ""),
                "observatories": meta.get("conf_observatories", ""),
                "macro_topics": meta.get("conf_macro_topics", ""),
            }

            debug_record = {
                "title": article["title"],
                "link": article["link"],
                "published_date": article["published_date"],
                "feed_name": article["feed_name"],
                "observatories": obs,
                "content_type": classification.get("content_type") or "",
                "target_audience": ta if isinstance(ta, list) else ([ta] if ta else []),
                "macro_topics": mt,
                "embed_scores_top": obs_result.get("scores", {}),
                "embed_scores_all": obs_result.get("all_scores", {}),
                "confidence": conf,
                "sources": source_summary,
                "used_llm": used_llm,
            }
            jsonlfile.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            jsonlfile.flush()

            append_article(article, classification, source="llm" if used_llm else "model", confidence=conf)

    total = len(articles)
    print(f"\nLLM calls: {llm_call_count}/{total}  |  Model: {model_call_count}/{total}")
    print(f"Debug JSONL salvato in: {jsonl_path}")


def run_compare(articles, osservatori, deployment, output_path):
    llm_client = build_client()
    embed_client = get_embed_client()

    print("Pre-calcolo embedding osservatori...")
    obs_embeddings = build_observatory_embeddings(osservatori, embed_client)
    _check_and_log_embedding_dim(obs_embeddings)  # [MOD] dimension guard + log INFO

    recall_scores = []
    label_counts = {"full": 0, "partial": 0, "miss": 0, "llm-empty/embed-empty": 0, "llm-empty/embed-added": 0}

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS_COMPARE)
        writer.writeheader()

        for i, article in enumerate(articles, 1):
            print(f"[{i}/{len(articles)}] {article['title'][:80]}")

            llm_result = classify_article(article, llm_client, osservatori, deployment)
            embed_result = classify_article_embed(article, embed_client, obs_embeddings)

            cmp = _compare(llm_result["observatories"], embed_result["observatories"])
            label_counts[cmp["label"]] = label_counts.get(cmp["label"], 0) + 1
            if cmp["recall"] is not None:
                recall_scores.append(cmp["recall"])

            print(f"  LLM:   {llm_result['observatories']}")
            print(f"  Embed: {embed_result['observatories']}  scores: {embed_result['scores']}")
            print(f"  Recall: {cmp['recall']}  ({cmp['label']}  {cmp['hits']}/{cmp['llm_total']})\n")

            row = {
                "title": article["title"],
                "link": article["link"],
                "published_date": article["published_date"],
                "feed_name": article["feed_name"],
                "llm_observatories": "|".join(llm_result["observatories"]),
                "embed_observatories": "|".join(embed_result["observatories"]),
                "embed_scores": str(embed_result["scores"]),
                "match": f"{cmp['label']} recall={cmp['recall']}",
            }
            writer.writerow(row)
            csvfile.flush()

    total = len(articles)
    avg_recall = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0
    print(f"\nRisultati confronto ({total} articoli):")
    print(f"  Avg recall (su articoli con LLM non vuoto): {avg_recall}")
    print(f"  Full recall (1.0):  {label_counts['full']}")
    print(f"  Partial recall:     {label_counts['partial']}")
    print(f"  Miss (0.0):         {label_counts['miss']}")
    print(f"  LLM vuoto:          {label_counts.get('llm-empty/embed-empty',0) + label_counts.get('llm-empty/embed-added',0)}")


def main():
    parser = argparse.ArgumentParser(description="Newsletter classification pipeline")
    parser.add_argument("--compare", action="store_true",
                        help="Run both LLM and embedding classification and compare results")
    parser.add_argument("--gate", action="store_true",
                        help="Use confidence gate: local models when confident, LLM fallback")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM entirely — use only local models (requires --gate)")
    parser.add_argument("--max-per-feed", type=int, default=10,
                        help="Max articles per feed (default: 10)")
    args = parser.parse_args()

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")

    print("Caricamento osservatori...")
    osservatori = load_osservatori(ROOT / "config" / "osservatori.json")

    print("Recupero articoli dai feed RSS...")
    articles = fetch_all(ROOT / "config" / "feeds.yaml", max_per_feed=args.max_per_feed)
    print(f"Trovati {len(articles)} articoli.\n")

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare:
        output_path = output_dir / f"compare_{timestamp}.csv"
        run_compare(articles, osservatori, deployment, output_path)
    elif args.gate or args.no_llm:
        output_path = output_dir / f"results_{timestamp}.csv"
        run_gate(articles, osservatori, deployment, output_path, no_llm=args.no_llm)
    else:
        output_path = output_dir / f"results_{timestamp}.csv"
        run_standard(articles, osservatori, deployment, output_path)

    print(f"Output salvato in: {output_path}")


if __name__ == "__main__":
    main()
