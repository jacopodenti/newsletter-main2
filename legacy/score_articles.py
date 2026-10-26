"""
Sperimenta pesi di scoring su un JSONL esistente.

Uso:
    python3 score_articles.py --obs omnichannel-customer-experience --w-obs 0.5 --w-ct 0.3 --w-ta 0.2
    python3 score_articles.py --obs artificial-intelligence --w-obs 1.0 --w-ct 0 --w-ta 0 --top 15
    python3 score_articles.py --list-obs   # mostra tutti gli osservatori disponibili

Formula:
    Score = w_obs * norm(obs_cosine)
          + w_ct  * norm(ct_prob  * ct_multiplier)
          + w_ta  * norm(ta_prob  * ta_multiplier)
    dove norm = min-max sull'intero batch
"""
import argparse
import json
import sys
from pathlib import Path

# --- moltiplicatori content type (stessi del foglio Config) ---
CT_MUL = {
    "analysis":            1.0,
    "trend_article":       1.0,
    "opinion_analysis":    1.0,
    "interview":           1.0,
    "research_report":     1.5,
    "regulation":          0.8,
    "vendor_announcement": 0.4,
    "case_study":          1.2,
    "partnership_event":   0.6,
}

# --- moltiplicatori target audience ---
TA_MUL = {
    "executive":       1.3,
    "marketing":       1.0,
    "sales":           1.0,
    "customer_care":   0.9,
    "it_data":         0.8,
    "cross_functional": 1.2,
    "research":        1.1,
}


def load_jsonl(path: Path) -> list[dict]:
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def score_batch(articles: list[dict], obs_slug: str, w_obs: float, w_ct: float, w_ta: float) -> list[dict]:
    rows = []
    for a in articles:
        obs_score = a.get("embed_scores_all", {}).get(obs_slug, 0.0)

        ct_pred = a.get("content_type", "")
        ct_probs = a.get("proba_content_type_embed", {})
        ct_prob = ct_probs.get(ct_pred, max(ct_probs.values()) if ct_probs else 0.0)
        ct_raw = ct_prob * CT_MUL.get(ct_pred, 1.0)

        ta_pred = a.get("target_audience", "")
        ta_probs = a.get("proba_target_audience_embed", {})
        ta_prob = ta_probs.get(ta_pred, max(ta_probs.values()) if ta_probs else 0.0)
        ta_raw = ta_prob * TA_MUL.get(ta_pred, 1.0)

        rows.append({
            "title": a.get("title", ""),
            "feed": a.get("feed_name", ""),
            "date": a.get("published_date", "")[:16],
            "obs_assigned": ", ".join(a.get("observatories", [])),
            "ct": ct_pred,
            "ta": ta_pred,
            "obs_score": obs_score,
            "ct_raw": ct_raw,
            "ta_raw": ta_raw,
        })

    # min-max normalization
    def norm(values):
        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.5] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    obs_norm  = norm([r["obs_score"] for r in rows])
    ct_norm   = norm([r["ct_raw"]    for r in rows])
    ta_norm   = norm([r["ta_raw"]    for r in rows])

    for i, r in enumerate(rows):
        r["score"] = round(w_obs * obs_norm[i] + w_ct * ct_norm[i] + w_ta * ta_norm[i], 4)
        r["obs_norm"] = round(obs_norm[i], 3)
        r["ct_norm"]  = round(ct_norm[i],  3)
        r["ta_norm"]  = round(ta_norm[i],  3)

    return sorted(rows, key=lambda x: x["score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Sperimenta pesi di scoring su JSONL esistente")
    parser.add_argument("--jsonl", default=None, help="Path al file JSONL (auto-detect se non specificato)")
    parser.add_argument("--obs", default="omnichannel-customer-experience", help="Slug osservatorio target")
    parser.add_argument("--w-obs", type=float, default=0.5, help="Peso osservatorio (default 0.5)")
    parser.add_argument("--w-ct",  type=float, default=0.3, help="Peso content type (default 0.3)")
    parser.add_argument("--w-ta",  type=float, default=0.2, help="Peso target audience (default 0.2)")
    parser.add_argument("--top",   type=int,   default=15,  help="Quanti articoli mostrare (default 15)")
    parser.add_argument("--list-obs", action="store_true", help="Elenca osservatori disponibili nel JSONL")
    args = parser.parse_args()

    # trova JSONL — spostato in legacy/ il 26/10: root ora un livello sopra
    root = Path(__file__).resolve().parent.parent
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        candidates = sorted(root.glob("*.jsonl")) + sorted((root / "output").glob("*.jsonl"))
        if not candidates:
            print("Nessun file .jsonl trovato. Passa --jsonl <path>.")
            sys.exit(1)
        jsonl_path = candidates[-1]  # più recente

    print(f"Carico: {jsonl_path}")
    articles = load_jsonl(jsonl_path)
    print(f"Articoli: {len(articles)}\n")

    if args.list_obs:
        slugs = set()
        for a in articles:
            slugs.update(a.get("embed_scores_all", {}).keys())
        for s in sorted(slugs):
            print(s)
        return

    total = args.w_obs + args.w_ct + args.w_ta
    if abs(total - 1.0) > 0.01:
        print(f"⚠  I pesi sommano a {total:.2f} invece di 1.0 — li normalizzo automaticamente.")
        args.w_obs /= total
        args.w_ct  /= total
        args.w_ta  /= total

    print(f"Osservatorio : {args.obs}")
    print(f"Pesi         : obs={args.w_obs:.2f}  ct={args.w_ct:.2f}  ta={args.w_ta:.2f}")
    print(f"Top          : {args.top}")
    print("─" * 100)

    ranked = score_batch(articles, args.obs, args.w_obs, args.w_ct, args.w_ta)

    for i, r in enumerate(ranked[:args.top], 1):
        assigned_marker = "✓" if args.obs in r["obs_assigned"] else " "
        print(f"#{i:2}  [{r['score']:.3f}]  {assigned_marker}  {r['title'][:70]}")
        print(f"       Feed: {r['feed']:<25}  CT: {r['ct']:<22}  TA: {r['ta']}")
        print(f"       obs_norm={r['obs_norm']:.2f}  ct_norm={r['ct_norm']:.2f}  ta_norm={r['ta_norm']:.2f}  | assegnato a: {r['obs_assigned'] or '—'}")
        print()


if __name__ == "__main__":
    main()
