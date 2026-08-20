"""
Costruisce un file Excel A SE' (non dentro il file sorgente) con due fogli:
  "Ranking"        — la classifica completa
  "Da_revisionare" — selezione curata per la revisione (Si da validare + No/borderline da minare)

Uso:
  python build_review_file.py <data_dir> <models_dir> <output.xlsx> <etichetta>
Esempio:
  python build_review_file.py ocx_relevance/data ocx_relevance/models OCX_ranking_revisione.xlsx OCX
"""
import csv
import json
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent / "ocx_relevance"))
from common import group_by_cluster

MIN_NEGATIVI = 30
N_BORDERLINE_EXTRA = 30

HEADS = ["Rank", "Score ranker", "Pertinenza LLM", "Parere ricercatore", "Testo", "Titolo", "Fonte", "Data", "Link", "Parere revisore LLM", "Confidence LLM"]

# Titolo colorato = sospetto duplicato (embedding molto simile a un altro candidato dello
# stesso run, es. stessa notizia da fonti diverse): stesso colore = stesso cluster.
CLUSTER_COLORS = ["FFD9D9", "D9E8FF", "D9FFE0", "F0D9FF", "D9FFF7", "FFE0F0", "E8FFD9", "E0E0FF"]


def cluster_fill(cluster_id):
    if not cluster_id:
        return None
    return PatternFill("solid", fgColor=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)])


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


def write_row(ws, r_idx, row, cache, yellow):
    title = row["titolo"]
    full = cache.get(title, {}).get("source") == "full"
    ws.cell(row=r_idx, column=1, value=int(row["rank"]))
    ws.cell(row=r_idx, column=2, value=float(row["score_ranker"]))
    ws.cell(row=r_idx, column=3, value=row["pertinenza_llm"] or "(non valutato dal gate)")
    c5 = ws.cell(row=r_idx, column=5, value="completo" if full else "solo sommario")
    if not full:
        c5.fill = yellow
    c6 = ws.cell(row=r_idx, column=6, value=title)
    cf = cluster_fill(row.get("cluster_id"))
    if cf:
        c6.fill = cf
    ws.cell(row=r_idx, column=7, value=row["fonte"])
    ws.cell(row=r_idx, column=8, value=row.get("data", ""))
    c9 = ws.cell(row=r_idx, column=9, value="apri articolo" if row["link"] else "")
    if row["link"]:
        c9.hyperlink = row["link"]
        c9.font = Font(color="0563C1", underline="single")
    ws.cell(row=r_idx, column=10, value=row.get("motivo_llm", ""))
    conf = row.get("confidence_llm", "")
    ws.cell(row=r_idx, column=11, value=float(conf) if conf not in ("", None) else "")


def write_rows_grouped(ws, start_row, rows, cache, yellow):
    """Come write_row, ma accosta i membri di uno stesso cluster (sospetto duplicato)
    al più in alto in classifica del gruppo, con una riga di intestazione leggera —
    usata solo nel foglio Da_revisionare, dove la revisione umana ne trae vantaggio."""
    rows = group_by_cluster(rows)
    r = start_row
    data_rows = []
    prev_cluster = None
    for row in rows:
        cid = row.get("cluster_id")
        n_members = sum(1 for x in rows if x.get("cluster_id") == cid) if cid else 0
        if cid and cid != prev_cluster and n_members >= 2:
            hcell = ws.cell(row=r, column=1, value=f"↳ sospetto duplicato — {n_members} fonti sulla stessa notizia")
            hcell.font = Font(italic=True, size=9, color="6B6F76")
            cf = cluster_fill(cid)
            for c in range(1, len(HEADS) + 1):
                ws.cell(row=r, column=c).fill = cf
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(HEADS))
            r += 1
        prev_cluster = cid
        write_row(ws, r, row, cache, yellow)
        data_rows.append(r)
        r += 1
    return r, data_rows


def main():
    data_dir, models_dir, out_path, label = sys.argv[1:5]
    data_dir, models_dir, out_path = Path(data_dir), Path(models_dir), Path(out_path)

    ranking = list(csv.DictReader(open(data_dir / "ranking.csv", encoding="utf-8")))
    cache = json.load(open(models_dir / "corpus_fulltext_cache.json", encoding="utf-8"))

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    section_fill = PatternFill("solid", fgColor="D9D9D9")
    yellow = PatternFill("solid", fgColor="FFF2CC")

    wb = openpyxl.Workbook()

    # --- foglio 1: Ranking completo ---
    ws1 = wb.active
    ws1.title = "Ranking"
    si_tot = sum(1 for r in ranking if r["pertinenza_llm"] == "Si")
    topn = sum(1 for r in ranking if r["pertinenza_llm"] in ("Si", "No"))
    ws1["A1"] = f"Ranking {label} — ranker + gate LLM su top-{topn}. Rosa finale: {si_tot}."
    ws1["A1"].font = Font(bold=True, size=12)
    for c, h in enumerate(HEADS, 1):
        cell = ws1.cell(row=3, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    for i, row in enumerate(ranking):
        write_row(ws1, 4 + i, row, cache, yellow)
    last1 = 3 + len(ranking)
    dv1 = DataValidation(type="list", formula1='"Si,No"', allow_blank=True)
    ws1.add_data_validation(dv1)
    dv1.add(f"D4:D{last1}")
    widths = {"A": 6, "B": 12, "C": 18, "D": 16, "E": 13, "F": 60, "G": 20, "H": 22, "I": 14, "J": 55, "K": 12}
    for col, wd in widths.items():
        ws1.column_dimensions[col].width = wd
    ws1.freeze_panes = "A4"
    ws1.auto_filter.ref = f"A3:K{last1}"

    # --- foglio 2: Da_revisionare (selezione curata) ---
    si_rows = [r for r in ranking if r["pertinenza_llm"] == "Si"]
    no_rows = [r for r in ranking if r["pertinenza_llm"] == "No"]
    ungated = [r for r in ranking if r["pertinenza_llm"] == ""]
    borderline = list(no_rows)
    if len(no_rows) < MIN_NEGATIVI:
        borderline = no_rows + ungated[:N_BORDERLINE_EXTRA]

    ws2 = wb.create_sheet("Da_revisionare")
    ws2["A1"] = ("Selezione curata per revisione — NON tutta la classifica. "
                 "Sezione 1: valida i \"Si\" del gate. Sezione 2: giudica i \"No\"/borderline — negativi genuinamente difficili, mai controllati da una persona. "
                 "Righe con la barra grigia: sospetti duplicati accostati al più in alto in classifica del gruppo — vedi motivo del revisore LLM.")
    ws2["A1"].font = Font(bold=True, size=12)

    row = 3
    ws2.cell(row=row, column=1, value=f"SEZIONE 1 — Valida i confermati Sì ({len(si_rows)} righe)").font = Font(bold=True)
    for c in range(1, len(HEADS) + 1):
        ws2.cell(row=row, column=c).fill = section_fill
    row += 1
    for c, h in enumerate(HEADS, 1):
        cell = ws2.cell(row=row, column=c, value=h); cell.font = hf; cell.fill = hd
    row += 1
    row, si_data_rows = write_rows_grouped(ws2, row, si_rows, cache, yellow)

    row += 1
    ws2.cell(row=row, column=1, value=f"SEZIONE 2 — Giudica No/borderline, negativi da minare ({len(borderline)} righe)").font = Font(bold=True)
    for c in range(1, len(HEADS) + 1):
        ws2.cell(row=row, column=c).fill = section_fill
    row += 1
    for c, h in enumerate(HEADS, 1):
        cell = ws2.cell(row=row, column=c, value=h); cell.font = hf; cell.fill = hd
    row += 1
    row, neg_data_rows = write_rows_grouped(ws2, row, borderline, cache, yellow)

    dv2 = DataValidation(type="list", formula1='"Si,No"', allow_blank=True)
    ws2.add_data_validation(dv2)
    for a, b in contiguous_ranges(si_data_rows) + contiguous_ranges(neg_data_rows):
        dv2.add(f"D{a}:D{b}")
    for col, wd in widths.items():
        ws2.column_dimensions[col].width = wd
    ws2.freeze_panes = "A4"

    wb.save(out_path)
    print(f"salvato: {out_path}  (Ranking: {len(ranking)} righe, Da_revisionare: {len(si_data_rows) + len(neg_data_rows)} righe)")


if __name__ == "__main__":
    main()
