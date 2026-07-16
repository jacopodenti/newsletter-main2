"""
STEP 5 — Pipeline completa su un insieme di articoli.

Flusso:  articoli -> testo intero -> embedding -> ranker (LogReg) -> ordina
         -> gate LLM sui primi N candidati -> output

Sorgente articoli (default): foglio 'Dati' di LABEL-Newsletter.xlsx (il corpus).
Uso:
  python ocx_relevance/05_rank.py                 # rank del corpus, LLM sui primi 150
  python ocx_relevance/05_rank.py --topn 100

Output: data/ranking.csv  (rank, score_ranker, pertinenza_llm, titolo, fonte, link)
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openpyxl
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, embed_texts, fetch_fulltext, llm_client, ocx_description, MODELS, DATA, XLSX_LABEL, LLM_MODEL
from ranker import score_similarity
import importlib
gate = importlib.import_module("04_llm_gate")


def load_corpus():
    dt = openpyxl.load_workbook(XLSX_LABEL, data_only=True)["Dati"]
    arts = []
    for r in range(2, dt.max_row + 1):
        t = dt.cell(row=r, column=1).value
        if not t:
            continue
        arts.append({"title": str(t).strip(), "summary": str(dt.cell(row=r, column=17).value or ""),
                     "fonte": str(dt.cell(row=r, column=4).value or ""), "link": str(dt.cell(row=r, column=2).value or "").strip()})
    return arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=150, help="quanti candidati passare all'LLM")
    args = ap.parse_args()

    log("STEP 5 — pipeline completa")
    arts = load_corpus()
    log(f"  articoli in ingresso: {len(arts)}")

    # 1) testo intero — con CACHE su disco (non ri-scarica gli articoli già presi)
    cache_path = MODELS / "corpus_fulltext_cache.json"
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    log(f"  1) testo intero — {len(cache)} già in cache, scarico solo i mancanti...")

    def f(a):
        c = cache.get(a["title"])
        if c:
            a["full_text"], a["text_source"] = c["text"], c["source"]
        else:
            a["full_text"], a["text_source"] = fetch_fulltext(a["link"], a["summary"])
        return a
    with ThreadPoolExecutor(max_workers=16) as ex:
        arts = list(ex.map(f, arts))
    for a in arts:
        cache[a["title"]] = {"text": a["full_text"], "source": a["text_source"]}
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    nfull = sum(1 for a in arts if a["text_source"] == "full")
    log(f"     testo intero: {nfull}/{len(arts)}  (cache: {cache_path.name})")

    # 2) embedding
    log("  2) embedding...")
    X, tok = embed_texts([f"{a['title']}. {a['full_text']}" for a in arts])
    log(f"     vettori {X.shape}, token {tok}")

    # 3) ranker — somiglianza agli esempi OCX (NIENTE addestramento)
    log("  3) punteggio per somiglianza agli esempi (reference_set.npz)...")
    ref = np.load(MODELS / "reference_set.npz")
    scores = score_similarity(X, ref["X"], ref["y"], k=3)
    order = np.argsort(-scores)

    # 4) gate LLM sui primi N
    log(f"  4) gate LLM ({LLM_MODEL}) sui primi {args.topn}...")
    topn = [arts[i] for i in order[:args.topn]]
    topn, gtok = gate.run_gate(topn)
    llm_map = {a["title"]: a.get("llm") for a in topn}
    si = sum(1 for a in topn if a.get("llm") == 1)
    log(f"     LLM-Si tra i primi {args.topn}: {si}  |  token {gtok['in']} in + {gtok['out']} out")

    # 5) output — CSV
    with open(DATA / "ranking.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank", "score_ranker", "pertinenza_llm", "titolo", "fonte", "link"])
        for rank, i in enumerate(order, 1):
            a = arts[i]
            pl = llm_map.get(a["title"])
            w.writerow([rank, round(float(scores[i]), 4),
                        ("Si" if pl == 1 else "No") if pl is not None else "",
                        a["title"], a["fonte"], a["link"]])
    log(f"\n  salvato: {DATA/'ranking.csv'}")

    # 5b) output — foglio nuovo dentro LABEL-Newsletter.xlsx
    from openpyxl.styles import Font, PatternFill
    # label note (dal dataset stabile, NON dal foglio Ranking le cui formule perdono la cache)
    lab = {}
    try:
        for line in open(DATA / "dataset.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("origin") == "set50":
                lab[r["title"].strip()] = "si" if r["label"] == 1 else "no"
    except Exception:
        pass

    SHEET = "Ranking_OCX_nuovo"
    wb = openpyxl.load_workbook(XLSX_LABEL)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)
    hf = Font(bold=True, color="FFFFFF"); hd = PatternFill("solid", fgColor="1F6F3E")
    green = PatternFill("solid", fgColor="C6EFCE"); red = PatternFill("solid", fgColor="FFC7CE")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    ws["A1"] = (f"Ranking OCX (nuovo: ranker + gate LLM su top-{args.topn}). Rosa finale: {si}.  "
                f"Label nota: verde=Si / rosso=No.  Testo: giallo = senza testo completo (solo sommario).")
    ws["A1"].font = Font(bold=True, size=12)
    heads = ["Rank", "Score ranker", "Pertinenza LLM", "Label nota", "Testo", "Titolo", "Fonte", "Link"]
    for c, h in enumerate(heads, 1):
        cell = ws.cell(row=3, column=c, value=h); cell.font = hf; cell.fill = hd
    for rank, i in enumerate(order, 1):
        a = arts[i]; row = 3 + rank
        pl = llm_map.get(a["title"]); L = lab.get(a["title"]); full = a.get("text_source") == "full"
        ws.cell(row=row, column=1, value=rank)
        ws.cell(row=row, column=2, value=round(float(scores[i]), 4))
        ws.cell(row=row, column=3, value=("Si" if pl == 1 else "No") if pl is not None else "")
        c4 = ws.cell(row=row, column=4, value=("Si" if L == "si" else "No") if L else "")
        if L == "si": c4.fill = green
        elif L == "no": c4.fill = red
        c5 = ws.cell(row=row, column=5, value="completo" if full else "solo sommario")
        if not full: c5.fill = yellow
        ws.cell(row=row, column=6, value=a["title"])
        ws.cell(row=row, column=7, value=a["fonte"])
        ws.cell(row=row, column=8, value=a["link"])
    for col, wd in {"A": 6, "B": 12, "C": 14, "D": 10, "E": 13, "F": 60, "G": 20, "H": 30}.items():
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "A4"
    wb.save(XLSX_LABEL)
    log(f"  salvato foglio '{SHEET}' in {XLSX_LABEL.name}")

    log(f"\n  ROSA FINALE (ranker top-{args.topn} confermati dall'LLM): {si} articoli")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
