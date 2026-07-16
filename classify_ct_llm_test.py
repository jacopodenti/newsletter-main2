"""
Test: confronta CT predetto dall'embedding vs CT predetto da Claude LLM
sui primi 10 articoli dell'ultimo JSONL disponibile.

Uso:
    python3 classify_ct_llm_test.py
    python3 classify_ct_llm_test.py --jsonl path/al/file.jsonl
    python3 classify_ct_llm_test.py --n 20
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import os
from openai import OpenAI
from classify_ct_llm import classify_ct


def load_jsonl(path: Path) -> list[dict]:
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=None, help="Path al file JSONL (auto-detect se omesso)")
    parser.add_argument("--n", type=int, default=10, help="Quanti articoli testare (default 10)")
    args = parser.parse_args()

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        candidates = sorted(ROOT.glob("*.jsonl")) + sorted((ROOT / "output").glob("*.jsonl"))
        if not candidates:
            print("Nessun file .jsonl trovato. Usa --jsonl <path>.")
            sys.exit(1)
        jsonl_path = candidates[-1]

    print(f"JSONL : {jsonl_path}")
    articles = load_jsonl(jsonl_path)[: args.n]
    print(f"Articoli da testare: {len(articles)}\n")

    client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

    col_t = 52
    col_e = 24
    col_l = 24
    header = (
        f"{'#':<3}  {'Titolo':<{col_t}}  {'CT embed':<{col_e}}  {'CT LLM':<{col_l}}  {'Conf':>6}  Match"
    )
    print(header)
    print("─" * len(header))

    matches = 0
    uncertain = 0

    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        summary = a.get("summary", "")
        ct_embed = a.get("content_type", "—")

        result = classify_ct(title, summary, client=client)
        ct_llm = result["label"]
        conf = result["confidence"]
        raw = result["raw_label"]

        is_match = ct_embed == raw
        if is_match:
            matches += 1
        if ct_llm == "uncertain":
            uncertain += 1

        match_marker = "✓" if is_match else "✗"
        title_short = (title[:col_t - 2] + "..") if len(title) > col_t else title
        ct_llm_display = f"{ct_llm}" + (f" [{raw}]" if ct_llm == "uncertain" else "")

        print(
            f"{i:<3}  {title_short:<{col_t}}  {ct_embed:<{col_e}}  {ct_llm_display:<{col_l}}  {conf:>6.2f}  {match_marker}"
        )
        print(f"       → {result['reason']}")
        print()

    print("─" * len(header))
    n = len(articles)
    print(f"Risultato: {matches}/{n} match embed↔LLM  |  {uncertain}/{n} uncertain (conf < 0.7)")


if __name__ == "__main__":
    main()
