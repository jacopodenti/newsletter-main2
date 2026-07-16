import csv
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classify import build_client, classify_article, load_osservatori

FALLBACK_REASON = "Output del modello non parsabile in JSON."

IT_TO_EN = {
    "Titolo": "title",
    "Link": "link",
    "Data pubblicazione": "published_date",
    "Sommario": "summary",
    "Fonte": "feed_name",
    "Osservatori": "observatories",
    "Tipo contenuto": "content_type",
    "Pubblico target": "target_audience",
    "Macro topic": "macro_topics",
    "Motivazione": "reason",
}

EN_TO_IT = {v: k for k, v in IT_TO_EN.items()}


def main(csv_path: Path):
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")
    osservatori = load_osservatori(ROOT / "config" / "osservatori.json")
    client = build_client()

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    failed = [i for i, r in enumerate(rows) if r.get("Motivazione") == FALLBACK_REASON]
    print(f"Articoli da riclassificare: {len(failed)}")

    for count, i in enumerate(failed, 1):
        row = rows[i]
        article = {
            "title": row["Titolo"],
            "summary": row["Sommario"],
            "link": row["Link"],
            "published_date": row["Data pubblicazione"],
            "feed_name": row["Fonte"],
        }
        print(f"[{count}/{len(failed)}] {article['title'][:80]}")

        result = classify_article(article, client, osservatori, deployment)

        rows[i]["Osservatori"] = ", ".join(result["observatories"])
        rows[i]["Tipo contenuto"] = result["content_type"] or ""
        rows[i]["Pubblico target"] = result["target_audience"] or ""
        rows[i]["Macro topic"] = ", ".join(result["macro_topics"])
        rows[i]["Motivazione"] = result["reason"] or ""

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    still_failed = sum(1 for r in rows if r.get("Motivazione") == FALLBACK_REASON)
    print(f"\nDone. Ancora non parsabili: {still_failed}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default: ultimo file results_
        files = sorted((ROOT / "output").glob("results_*.csv"))
        if not files:
            print("Nessun file results_*.csv trovato.")
            sys.exit(1)
        csv_path = files[-1]
    else:
        csv_path = Path(sys.argv[1])

    print(f"File: {csv_path}")
    main(csv_path)
