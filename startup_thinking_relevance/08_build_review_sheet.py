"""
STEP 8 — Costruisce un secondo foglio "Da_revisionare": una selezione curata
(non tutta la classifica) per la revisione umana, con due scopi:
  1. Validare la classificazione (i "Si" del gate)
  2. Minare negativi difficili genuini (i "No" del gate + borderline appena
     fuori dai primi N, mai controllati da un umano finora)

Non rifà nessuna chiamata API: legge solo ranking.csv e corpus_fulltext_cache.json.

Output: SUT_Elenco news.xlsx -> nuovo foglio "Da_revisionare"
Uso:    python startup_thinking_relevance/08_build_review_sheet.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import group_by_cluster

DATA = HERE / "data"
MODELS = HERE / "models"
XLSX_EDITORIAL = HERE / "SUT_Elenco news.xlsx"

MIN_NEGATIVI = 30  # se i "No" del gate sono meno, integra con borderline oltre il cutoff
N_BORDERLINE_EXTRA = 30


def contiguous_ranges(row_nums):
    """[3,4,5,9,10] -> [(3,5),(9,10)] — per applicare la dropdown solo alle righe dati,
    saltando le intestazioni di cluster inserite in mezzo."""
    if not row_nums:
        return []
    ranges, start = [], row_nums[0]
    prev = start
    for n in row_nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append((start, prev))
            start = prev = n
    ranges.append((start, prev))
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="Da_revisionare")
    args = ap.parse_args()

    # 1) carica ranking + cache e seleziona le righe delle due sezioni (Si da validare;
    # No/borderline da minare, integrati con qualche borderline se i "No" del gate sono pochi)
    ranking = list(csv.DictReader(open(DATA / "ranking.csv", encoding="utf-8")))
    cache = json.load(open(MODELS / "corpus_fulltext_cache.json", encoding="utf-8"))

    si_rows = [r for r in ranking if r["pertinenza_llm"] == "Si"]
    no_rows = [r for r in ranking if r["pertinenza_llm"] == "No"]
    ungated = [r for r in ranking if r["pertinenza_llm"] == ""]

    borderline = list(no_rows)
    if len(no_rows) < MIN_NEGATIVI:
        extra = ungated[:N_BORDERLINE_EXTRA]  # ungated e' gia' ordinato per rank (score decrescente)
        borderline = no_rows + extra
        print(f"  solo {len(no_rows)} 'No' dal gate — integrati con {len(extra)} borderline appena fuori dai primi N (mai valutati)")

    print(f"validazione (Si): {len(si_rows)}  |  negativi da minare (No + borderline): {len(borderline)}")

    # 2) crea (o sostituisce) il foglio
    SHEET = args.sheet
    wb = openpyxl.load_workbook(XLSX_EDITORIAL)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    section_fill = PatternFill("solid", fgColor="D9D9D9")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    # Titolo colorato = sospetto duplicato (embedding molto simile a un altro candidato
    # dello stesso run, es. stessa notizia da fonti diverse): stesso colore = stesso cluster.
    CLUSTER_COLORS = ["FFD9D9", "D9E8FF", "D9FFE0", "F0D9FF", "D9FFF7", "FFE0F0", "E8FFD9", "E0E0FF"]

    def cluster_fill(cluster_id):
        if not cluster_id:
            return None
        return PatternFill("solid", fgColor=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)])

    ws["A1"] = ("Selezione curata per revisione — NON tutta la classifica. "
                "Sezione 1: valida i \"Si\" del gate. Sezione 2: giudica i \"No\"/borderline — qui stanno i negativi genuinamente difficili, mai controllati da una persona. "
                "Righe con la barra grigia: sospetti duplicati accostati al più in alto in classifica del gruppo — vedi motivo del revisore LLM.")
    ws["A1"].font = Font(bold=True, size=12)

    heads = ["Rank", "Score ranker", "Pertinenza LLM", "Parere ricercatore", "Testo", "Titolo", "Fonte", "Data", "Link", "Parere revisore LLM", "Voto LLM"]

    def write_header(row):
        for c, h in enumerate(heads, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = hf
            cell.fill = hd

    def write_section_title(row, text):
        ws.cell(row=row, column=1, value=text).font = Font(bold=True)
        for c in range(1, len(heads) + 1):
            ws.cell(row=row, column=c).fill = section_fill

    def write_rows(start_row, rows):
        # Cluster (sospetti duplicati) accostati al membro più in alto in classifica,
        # non più sparsi alle loro posizioni individuali — così si vedono a colpo d'occhio.
        rows = group_by_cluster(rows)
        r = start_row
        data_rows = []
        prev_cluster = None
        for row in rows:
            cid = row.get("cluster_id")
            n_members = sum(1 for x in rows if x.get("cluster_id") == cid) if cid else 0
            if cid and cid != prev_cluster and n_members >= 2:
                # intestazione solo se in QUESTA sezione sono visibili almeno 2 membri
                # dello stesso cluster — un solo membro qui non è un "gruppo" da mostrare insieme.
                hcell = ws.cell(row=r, column=1, value=f"↳ sospetto duplicato — {n_members} fonti sulla stessa notizia")
                hcell.font = Font(italic=True, size=9, color="6B6F76")
                cf = cluster_fill(cid)
                for c in range(1, len(heads) + 1):
                    ws.cell(row=r, column=c).fill = cf
                ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(heads))
                r += 1
            prev_cluster = cid

            title = row["titolo"]
            full = cache.get(title, {}).get("source") == "full"
            ws.cell(row=r, column=1, value=int(row["rank"]))
            ws.cell(row=r, column=2, value=float(row["score_ranker"]))
            ws.cell(row=r, column=3, value=row["pertinenza_llm"] or "(non valutato dal gate)")
            c5 = ws.cell(row=r, column=5, value="completo" if full else "solo sommario")
            if not full:
                c5.fill = yellow
            c6 = ws.cell(row=r, column=6, value=title)
            cf = cluster_fill(cid)
            if cf:
                c6.fill = cf
            ws.cell(row=r, column=7, value=row["fonte"])
            ws.cell(row=r, column=8, value=row.get("data", ""))
            c9 = ws.cell(row=r, column=9, value="apri articolo" if row["link"] else "")
            if row["link"]:
                c9.hyperlink = row["link"]
                c9.font = Font(color="0563C1", underline="single")
            ws.cell(row=r, column=10, value=row.get("motivo_llm", ""))
            voto = row.get("voto_llm", "")
            ws.cell(row=r, column=11, value=int(float(voto)) if voto not in ("", None) else "")
            data_rows.append(r)
            r += 1
        return r, data_rows

    # 3) scrive le due sezioni — intestazioni + righe (write_rows raggruppa i sospetti
    # duplicati accanto al membro più in alto in classifica)
    row = 3
    write_section_title(row, f"SEZIONE 1 — Valida i confermati Sì ({len(si_rows)} righe)")
    row += 1
    write_header(row)
    row += 1
    row, si_data_rows = write_rows(row, si_rows)

    row += 1  # riga vuota di separazione
    write_section_title(row, f"SEZIONE 2 — Giudica No/borderline, negativi da minare ({len(borderline)} righe)")
    row += 1
    write_header(row)
    row += 1
    row, neg_data_rows = write_rows(row, borderline)

    # 4) dropdown "Parere ricercatore" (una per intervallo contiguo di righe dati, saltando
    # le intestazioni di sezione/cluster) + formattazione finale e salvataggio
    dv = DataValidation(type="list", formula1='"Si,No"', allow_blank=True)
    ws.add_data_validation(dv)
    for a, b in contiguous_ranges(si_data_rows) + contiguous_ranges(neg_data_rows):
        dv.add(f"D{a}:D{b}")

    widths = {"A": 6, "B": 12, "C": 18, "D": 16, "E": 13, "F": 60, "G": 20, "H": 22, "I": 14, "J": 55, "K": 12}
    for col, wd in widths.items():
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "A4"

    wb.save(XLSX_EDITORIAL)
    print(f"salvato foglio '{SHEET}' in {XLSX_EDITORIAL.name}  ({len(si_data_rows) + len(neg_data_rows)} righe totali)")


if __name__ == "__main__":
    main()
