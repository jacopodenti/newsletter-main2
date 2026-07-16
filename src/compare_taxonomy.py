"""
Compare LLM taxonomy classification (content_type, target_audience, macro_topics)
against embedding-based classification using pre-computed results CSV.

Usage:
    python src/compare_taxonomy.py --results output/results_20260429_125950.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

TAXONOMY_PATH = ROOT / "config" / "classification_taxonomy.json"
DELAY = 0.05


def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[: -len("/openai/v1")]
    return AzureOpenAI(
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01"),
        azure_endpoint=endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def _embed(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    response = client.embeddings.create(input=texts, model=deployment)
    return [item.embedding for item in response.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
   
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def build_vocab_embeddings(client: AzureOpenAI, taxonomy: dict) -> dict:
    """Pre-compute embeddings for all taxonomy values. Returns {field: [{slug, embedding}]}"""
    result = {}
    for field, items in taxonomy.items():
        texts = [f"{it['slug']}: {it['descrizione']}" for it in items]
        print(f"  Embedding {len(texts)} valori per '{field}'...")
        embeddings = _embed(client, texts)
        result[field] = [
            {"slug": it["slug"], "embedding": emb}
            for it, emb in zip(items, embeddings)
        ]
    return result


def classify_field(article_emb: list[float], vocab: list[dict], top_k: int, threshold: float) -> list[str]:
    scores = [(v["slug"], _cosine(article_emb, v["embedding"])) for v in vocab]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [slug for slug, score in scores[:top_k] if score >= threshold]


def recall(llm: list[str], embed: list[str]) -> dict:
    if not llm:
        return {"label": "llm-empty", "recall": None, "hits": 0, "total": 0}
    hits = set(llm) & set(embed)
    r = len(hits) / len(llm)
    return {
        "label": "full" if r == 1.0 else ("partial" if r > 0 else "miss"),
        "recall": round(r, 2),
        "hits": len(hits),
        "total": len(llm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results CSV with LLM output")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()

    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    client = _get_client()

    print("Pre-calcolo embedding tassonomia...")
    vocab_embs = build_vocab_embeddings(client, taxonomy)

    results_path = Path(args.results)
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    total = len(rows)
    print(f"Articoli da confrontare: {total}\n")

    recall_sums = {"content_type": [], "target_audience": [], "macro_topics": []}
    output_rows = []

    for i, row in enumerate(rows, 1):
        title = row["title"]
        summary = row.get("summary", "")
        # strip HTML from summary
        import re
        summary_clean = re.sub(r"<[^>]+>", " ", summary).strip()
        text = f"{title}. {summary_clean}"

        print(f"[{i}/{total}] {title[:70]}")

        [article_emb] = _embed(client, [text])
        time.sleep(DELAY)

        # --- content_type: single value from LLM, top-1 from embed
        llm_ct = [row["content_type"]] if row.get("content_type") else []
        embed_ct = classify_field(article_emb, vocab_embs["content_type"], top_k=1, threshold=0.0)
        cmp_ct = recall(llm_ct, embed_ct)

        # --- target_audience: single value from LLM, top-1 from embed
        llm_ta = [row["target_audience"]] if row.get("target_audience") else []
        embed_ta = classify_field(article_emb, vocab_embs["target_audience"], top_k=1, threshold=0.0)
        cmp_ta = recall(llm_ta, embed_ta)

        # --- macro_topics: multi-value from LLM, top-3 from embed above threshold
        llm_mt = [t.strip() for t in row.get("macro_topics", "").split("|") if t.strip()]
        embed_mt = classify_field(article_emb, vocab_embs["macro_topics"], top_k=3, threshold=args.threshold)
        cmp_mt = recall(llm_mt, embed_mt)

        for field, cmp in [("content_type", cmp_ct), ("target_audience", cmp_ta), ("macro_topics", cmp_mt)]:
            if cmp["recall"] is not None:
                recall_sums[field].append(cmp["recall"])

        print(f"  content_type   LLM:{llm_ct} → Embed:{embed_ct}  {cmp_ct['label']} ({cmp_ct['recall']})")
        print(f"  target_audience LLM:{llm_ta} → Embed:{embed_ta}  {cmp_ta['label']} ({cmp_ta['recall']})")
        print(f"  macro_topics   LLM:{llm_mt} → Embed:{embed_mt}  {cmp_mt['label']} ({cmp_mt['recall']})\n")

        output_rows.append({
            "title": title,
            "llm_content_type": row.get("content_type", ""),
            "embed_content_type": embed_ct[0] if embed_ct else "",
            "ct_match": cmp_ct["label"],
            "llm_target_audience": row.get("target_audience", ""),
            "embed_target_audience": embed_ta[0] if embed_ta else "",
            "ta_match": cmp_ta["label"],
            "llm_macro_topics": row.get("macro_topics", ""),
            "embed_macro_topics": "|".join(embed_mt),
            "mt_recall": cmp_mt["recall"],
            "mt_label": cmp_mt["label"],
        })

    # summary
    print("\n=== RIEPILOGO ===")
    for field, scores in recall_sums.items():
        if scores:
            avg = round(sum(scores) / len(scores), 2)
            full = scores.count(1.0)
            miss = scores.count(0.0)
            print(f"  {field}: avg recall={avg}  full={full}/{len(scores)}  miss={miss}/{len(scores)}")

    # save output
    out_path = results_path.parent / f"taxonomy_compare_{results_path.stem}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()

