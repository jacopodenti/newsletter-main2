"""
STEP 7 — Scrive il ranking come nuovo foglio DENTRO ExportNewsLetters.xlsx.

STESSO SCHEMA di "Ranking_OCX_nuovo" (ocx_relevance/06_rank.py) — stesse colonne,
stessa logica, stessi colori. Aggiunge solo filtro attivo, che mancava in
entrambi i fogli (vedi anche fix_ocx_sheet_formatting.py per allineare quello OCX).

Non rifà nessuna chiamata API: legge solo i file già prodotti (ranking.csv,
corpus_fulltext_cache.json, dataset.jsonl).

Output: ExportNewsLetters.xlsx -> nuovo foglio "Ranking_AI_nuovo"
Uso:    python ai_relevance/07_export_excel.py
"""
import argparse
import csv
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"
XLSX_EDITORIAL = HERE / "ExportNewsLetters.xlsx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="ranking.csv", help="file ranking da usare (dentro data/)")
    ap.add_argument("--sheet", default="Ranking_AI_nuovo", help="nome del foglio da scrivere")
    ap.add_argument("--note", default="", help="testo extra da aggiungere all'intestazione")
    args = ap.parse_args()

    # 1) input — legge ranking.csv e la cache del testo (nessuna chiamata API qui)
    ranking = list(csv.DictReader(open(DATA / args.ranking, encoding="utf-8")))
    cache = json.load(open(MODELS / "corpus_fulltext_cache.json", encoding="utf-8"))

    si = sum(1 for r in ranking if r["pertinenza_llm"] == "Si")
    topn = sum(1 for r in ranking if r["pertinenza_llm"] in ("Si", "No"))
    n = len(ranking)

    # 2) crea il foglio (sostituendo un eventuale foglio con lo stesso nome)
    SHEET = args.sheet
    wb = openpyxl.load_workbook(XLSX_EDITORIAL)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    # Titolo colorato = sospetto duplicato (embedding molto simile a un altro candidato
    # dello stesso run, es. stessa notizia da fonti diverse): stesso colore = stesso cluster.
    CLUSTER_COLORS = ["FFD9D9", "D9E8FF", "D9FFE0", "F0D9FF", "D9FFF7", "FFE0F0", "E8FFD9", "E0E0FF"]

    def cluster_fill(cluster_id):
        if not cluster_id:
            return None
        return PatternFill("solid", fgColor=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)])

    # 3) titolo + intestazione colonne
    ws["A1"] = (f"Ranking AI (ranker + gate LLM su top-{topn}). Rosa finale: {si}.  "
                f"Parere ricercatore: da compilare a mano (Si/No).  Testo: giallo = senza testo completo (solo sommario).  "
                f"Titolo colorato: sospetto duplicato (stessa notizia da fonti diverse) — vedi motivo del revisore LLM.{args.note}")
    ws["A1"].font = Font(bold=True, size=12)

    heads = ["Rank", "Score ranker", "Pertinenza LLM", "Parere ricercatore", "Testo", "Titolo", "Fonte", "Data", "Link", "Parere revisore LLM", "Confidence LLM"]
    for c, h in enumerate(heads, 1):
        cell = ws.cell(row=3, column=c, value=h)
        cell.font = hf
        cell.fill = hd

    # 4) righe dati — una per articolo (giallo = senza testo completo, colore = cluster duplicati)
    for i, r in enumerate(ranking):
        row = 3 + i + 1
        title = r["titolo"]
        full = cache.get(title, {}).get("source") == "full"

        ws.cell(row=row, column=1, value=int(r["rank"]))
        ws.cell(row=row, column=2, value=float(r["score_ranker"]))
        ws.cell(row=row, column=3, value=r["pertinenza_llm"])
        # colonna 4 "Parere ricercatore" lasciata vuota apposta: da compilare a mano (dropdown Si/No)
        c5 = ws.cell(row=row, column=5, value="completo" if full else "solo sommario")
        if not full:
            c5.fill = yellow
        c6 = ws.cell(row=row, column=6, value=title)
        cf = cluster_fill(r.get("cluster_id"))
        if cf:
            c6.fill = cf
        ws.cell(row=row, column=7, value=r["fonte"])
        ws.cell(row=row, column=8, value=r.get("data", ""))
        c9 = ws.cell(row=row, column=9, value="apri articolo" if r["link"] else "")
        if r["link"]:
            c9.hyperlink = r["link"]
            c9.font = Font(color="0563C1", underline="single")
        ws.cell(row=row, column=10, value=r.get("motivo_llm", ""))
        conf = r.get("confidence_llm", "")
        ws.cell(row=row, column=11, value=float(conf) if conf not in ("", None) else "")

    # 5) dropdown Si/No + rifiniture foglio (larghezze, filtro, freeze) e salvataggio
    last_row = 3 + n
    dv = DataValidation(type="list", formula1='"Si,No"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D4:D{last_row}")

    widths = {"A": 6, "B": 12, "C": 14, "D": 16, "E": 13, "F": 60, "G": 20, "H": 22, "I": 14, "J": 55, "K": 12}
    for col, wd in widths.items():
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:K{2 + n + 1}"

    wb.save(XLSX_EDITORIAL)
    print(f"salvato foglio '{SHEET}' in {XLSX_EDITORIAL.name}  ({n} righe)")


if __name__ == "__main__":
    main()
