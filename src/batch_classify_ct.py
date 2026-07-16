"""
Classifica CT degli articoli in Book.xlsx foglio Foglio1.
Uso:
    python src/batch_classify_ct.py
    
    python src/batch_classify_ct.py path/al/file.xlsx
"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import openpyxl
from classify_ct_llm import classify_ct, _build_client

DEFAULT_EXCEL = ROOT / "Book-2.xlsx"
SHEET_NAME = "Foglio1"
HEADER_ROW = 1  # header in riga 1, dati da riga 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("excel", nargs="?", default=str(DEFAULT_EXCEL))
    args = parser.parse_args()

    path = Path(args.excel)
    if not path.exists():
        print(f"File non trovato: {path}")
        sys.exit(1)

    print(f"File: {path}")
    wb = openpyxl.load_workbook(path)
    ws = wb[SHEET_NAME]

    # trova colonne dall'header in riga 1
    headers = {cell.value: cell.column for cell in ws[HEADER_ROW] if cell.value}
    col_title = headers.get("Titolo")
    col_summary = headers.get("Sommario")

    if not col_title or not col_summary:
        print(f"Errore: colonne Titolo/Sommario non trovate. Header trovati: {list(headers)}")
        sys.exit(1)

    # crea colonna CT LLM se non esiste
    col_ct_llm = headers.get("CT LLM")
    if col_ct_llm is None:
        col_ct_llm = max(headers.values()) + 1
        ws.cell(row=HEADER_ROW, column=col_ct_llm, value="CT LLM")
        print(f"Colonna 'CT LLM' creata (colonna {col_ct_llm})")
    else:
        print(f"Colonna 'CT LLM' già presente (colonna {col_ct_llm})")

    client = _build_client()

    i = 0
    for row_idx in range(HEADER_ROW + 1, ws.max_row + 1):
        title = ws.cell(row=row_idx, column=col_title).value
        if not title or not str(title).strip():
            break  # stop alla prima riga vuota

        title = str(title).strip()
        summary = str(ws.cell(row=row_idx, column=col_summary).value or "").strip()

        result = classify_ct(title, summary, client=client)
        label = result["label"]
        conf = result["confidence"]

        ws.cell(row=row_idx, column=col_ct_llm, value=label)
        i += 1
        print(f"[{i}] {label:<24} conf={conf:.2f}  {title[:60]}")
        print(f"     → {result['reason']}")

        time.sleep(0.5)

    wb.save(path)
    print(f"\nSalvato: {path.name}  ({i} righe elaborate)")


if __name__ == "__main__":
    main()