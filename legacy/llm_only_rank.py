"""
Esperimento diagnostico: "come classificherebbe un LLM da solo, senza il ranker kNN?"

Rivaluta gli stessi candidati già nel foglio "Da_revisionare" di una pipeline (i "Sì"
del gate + i "No"/borderline — la stessa identica selezione già scritta nel foglio),
con un punteggio LLM indipendente (0-100), e scrive DIRETTAMENTE nel foglio già
esistente due colonne in coda: "Rank LLM" e "Δ posizione" (colonne L/M — K è
"Confidence LLM" del gate di produzione) (rank kNN - rank LLM, entrambi ricalcolati
1..N dentro il solo sottoinsieme dei candidati, per un confronto onesto — non il rank
globale sul corpus intero).

NON tocca ranking.csv, NON tocca le colonne A-K già scritte, NON tocca il gate/ranker
di produzione: è additivo, si limita ad aggiungere 2 colonne in coda (L/M) alle righe
già presenti (abbinate per link).

Uso:
  python llm_only_rank.py --pipeline ocx                    # gpt-5.4 (default)
  python llm_only_rank.py --pipeline ai --model gpt-5.6-sol  # prova il modello storico
  python llm_only_rank.py --pipeline all                    # tutte e 3 di fila
  python llm_only_rank.py --pipeline ocx --dry-run           # conta candidati, zero chiamate
  python llm_only_rank.py --pipeline ocx --limit 15          # lotto pilota

Modalità --full-corpus-study: ranking LLM sull'INTERO corpus (non solo i candidati gated) a
confronto col ranking kNN, scritto in revisione/ (mai nei file di produzione) — solo per le
pipeline in FULL_CORPUS_STUDY_PIPELINES (oggi: ocx, startup, ai). Costo reale, non un
sottoinsieme: vedi log per la stima token/€ prima di lanciare il giro completo.
  python llm_only_rank.py --pipeline ocx --full-corpus-study --dry-run
  python llm_only_rank.py --pipeline ocx --full-corpus-study --limit 20      # lotto pilota
  python llm_only_rank.py --pipeline ocx --full-corpus-study                 # giro pieno, modelli default: gpt-5.4,gpt-5.6-sol
  python llm_only_rank.py --pipeline startup --full-corpus-study --models gpt-5.4,gpt-5.6-sol

Modalità --add-model: aggiunge UN modello nuovo a un foglio 'LLM_vs_kNN_corpus_intero' già
scritto da --full-corpus-study, SENZA rispendere sui modelli già presenti (rilegge i loro
rank dal foglio, chiamate reali solo per il modello nuovo). --models qui indica i modelli
GIÀ nel foglio, non quelli da aggiungere.
  python llm_only_rank.py --pipeline ocx --add-model gpt-5.6-luna --models gpt-5.4,gpt-5.6-sol

Modalità --cross-summary: NESSUNA chiamata LLM — rilegge i fogli 'LLM_vs_kNN_corpus_intero'
già scritti (per le pipeline in --pipelines) e li affianca in un unico file nuovo dentro
revisione/ (Confronto_kNN_vs_LLM.xlsx di default), utile per confrontare gli osservatori
senza aprire un file a testa.
  python llm_only_rank.py --cross-summary --pipelines ocx,startup,ai --models gpt-5.4,gpt-5.6-sol,gpt-5.6-luna
"""
import argparse
import csv
import importlib.util
import itertools
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parent.parent  # spostato in legacy/ il 26/10
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import log, llm_client, ocx_description

MIN_NEGATIVI = 30       # stessa soglia di 0X_build_review_sheet.py
N_BORDERLINE_EXTRA = 30

# Stessa tavolozza di ranker.py/0X_export_excel.py: stesso cluster = stesso colore
# ovunque nei fogli, anche qui in Analisi_Cluster.
CLUSTER_COLORS = ["FFD9D9", "D9E8FF", "D9FFE0", "F0D9FF", "D9FFF7", "FFE0F0", "E8FFD9", "E0E0FF"]


def cluster_fill(cluster_id):
    if not cluster_id:
        return None
    return PatternFill("solid", fgColor=CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)])

# Costo dell'ultimo giro DI PRODUZIONE (embedding sull'intero corpus + gate LLM sui soli
# top-150) — letto dai log reali del 23-24/07, non stimato. Se rilanci 0X_rank.py con un
# corpus diverso questi numeri vanno aggiornati a mano (non c'è un posto dove la pipeline
# di produzione li salva da sola oggi).
PROD_COST = {
    "ocx":     dict(corpus_size=1412, embed_tokens=1252667, gate_in=230948, gate_out=10491),
    "ai":      dict(corpus_size=1384, embed_tokens=1233504, gate_in=235175, gate_out=12092),
    "startup": dict(corpus_size=1378, embed_tokens=1202541, gate_in=250493, gate_out=12002),
}

# Tariffe (€ per milione di token). gpt-5.4: confermate dall'utente. Embedding: prezzo
# pubblico noto di text-embedding-3-large, NON confermato sul contratto Azure specifico —
# va corretto se diverso.
PRICE_EUR_PER_1M = {
    "embed_in": 0.13,
    "gpt54_in": 2.50,
    "gpt54_out": 15.00,
}

# Costo REALE già misurato del ranking-solo-LLM (gpt-5.4) sui candidati di Da_revisionare
# (giro completo del 24/07 — vedi log). Serve solo come riserva per --cluster-sheet-only,
# che non richiama l'LLM e quindi non ha un `tok` fresco da un run appena fatto.
LLM_ONLY_COST_LAST_RUN = {
    "ocx":     dict(n=148, tok_in=220618, tok_out=11864),
    "ai":      dict(n=179, tok_in=260106, tok_out=14860),
    "startup": dict(n=148, tok_in=244813, tok_out=11141),
}

PIPELINES = {
    "ocx": dict(
        data_dir=ROOT / "ocx_relevance" / "data",
        xlsx=ROOT / "LABEL-Newsletter.xlsx",
        xlsx_revisione=ROOT / "revisione" / "OCX_ranking_revisione.xlsx",
        full_ranking_sheet="Ranking_OCX_nuovo",
        gate_dir=ROOT / "ocx_relevance", gate_module="05_llm_gate", desc_attr=None,
        criteria_attr="OCX_CRITERIA",
        obs_name="Omnichannel Customer Experience",
    ),
    "ai": dict(
        data_dir=ROOT / "ai_relevance" / "data",
        xlsx=ROOT / "ai_relevance" / "ExportNewsLetters.xlsx",
        xlsx_revisione=ROOT / "revisione" / "AI_ranking_revisione.xlsx",
        full_ranking_sheet="Ranking_AI_nuovo",
        gate_dir=ROOT / "ai_relevance", gate_module="05_llm_gate", desc_attr="AI_DESCRIPTION",
        criteria_attr="AI_CRITERIA",
        obs_name="Artificial Intelligence",
    ),
    "startup": dict(
        data_dir=ROOT / "startup_thinking_relevance" / "data",
        xlsx=ROOT / "startup_thinking_relevance" / "SUT_Elenco news.xlsx",
        xlsx_revisione=ROOT / "revisione" / "StartupThinking_ranking_revisione.xlsx",
        full_ranking_sheet="Ranking_ST_nuovo",
        gate_dir=ROOT / "startup_thinking_relevance", gate_module="05_llm_gate", desc_attr="STARTUP_DESCRIPTION",
        criteria_attr="STARTUP_CRITERIA",
        obs_name="Startup Thinking",
    ),
}

# Pipeline supportate dallo studio "LLM sull'intero corpus vs kNN" (quelle con criteria_attr
# valorizzato).
FULL_CORPUS_STUDY_PIPELINES = ("ocx", "startup", "ai")

# Prompt "editoriale": non solo pertinenza (0/100 tutto ciò che tocca il tema), ma quanto
# l'articolo meriterebbe DAVVERO la selezione in newsletter — stessi criteri RIENTRA/NON
# RIENTRA del gate di produzione (via get_criteria), non un metro diverso.
PROMPT = """Sei un valutatore editoriale per l'Osservatorio "{obs_name}" del Politecnico di Milano.

DESCRIZIONE OSSERVATORIO:
{desc}

CRITERI DI PERTINENZA:
{criteria}

COMPITO: dato un articolo, assegna un punteggio da 0 a 100 di quanto MERITEREBBE DAVVERO
la selezione per la prossima newsletter di questo Osservatorio — non solo "è pertinente al
tema" (condizione necessaria, vedi criteri sopra), ma "è abbastanza rilevante, specifico e
di valore editoriale da essere una delle notizie scelte quella settimana", come farebbe un
ricercatore che seleziona a mano tra decine di candidati. 0 = fuori scope o banale, 100 =
sceglieresti sicuramente questo articolo. Valuta l'articolo da solo, confrontandolo solo
con i criteri sopra — non stai confrontando con altri articoli, ogni valutazione è
indipendente.

Rispondi SOLO con JSON: {{"punteggio": 0-100, "motivo": "una frase"}}

ARTICOLO:
Titolo: {title}
Testo: {body}

JSON:"""


def _load_gate_attr(cfg, attr_name):
    """Carica una costante dal modulo gate della pipeline, per percorso esplicito (non
    import_module) perché ai_relevance e startup_thinking_relevance hanno ENTRAMBI un file
    chiamato "05_llm_gate.py" — con import_module il secondo ripescherebbe dalla cache di
    sys.modules il modulo del primo (stesso nome)."""
    path = cfg["gate_dir"] / f"{cfg['gate_module']}.py"
    spec = importlib.util.spec_from_file_location(f"_gate_{cfg['gate_dir'].name}_{attr_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


def get_description(cfg):
    """Descrizione osservatorio: OCX la legge da config/osservatori.json (common.py),
    AI e Startup Thinking la tengono come costante nel loro modulo gate."""
    if cfg["desc_attr"] is None:
        return ocx_description()
    return _load_gate_attr(cfg, cfg["desc_attr"])


def get_criteria(cfg):
    """Criteri RIENTRA/NON RIENTRA (o soglia di pertinenza) — stessa costante usata dal gate
    di produzione, così il ranking-LLM-sul-corpus-intero giudica con lo stesso metro del
    Sì/No di produzione, non un criterio diverso inventato qui. Stringa vuota se la pipeline
    non ha ancora una costante criteri a sé (oggi: AI, fuori scope per questo giro)."""
    if cfg["criteria_attr"] is None:
        return ""
    return _load_gate_attr(cfg, cfg["criteria_attr"])


def load_candidates(cfg):
    """Stessa identica selezione di 0X_build_review_sheet.py: i "Sì" del gate +
    i "No"/borderline (con l'integrazione di ungated se i "No" sono pochi)."""
    ranking = list(csv.DictReader(open(cfg["data_dir"] / "ranking.csv", encoding="utf-8")))
    cache = json.load(open(cfg["data_dir"].parent / "models" / "corpus_fulltext_cache.json", encoding="utf-8"))

    si_rows = [r for r in ranking if r["pertinenza_llm"] == "Si"]
    no_rows = [r for r in ranking if r["pertinenza_llm"] == "No"]
    ungated = [r for r in ranking if r["pertinenza_llm"] == ""]
    borderline = list(no_rows)
    if len(no_rows) < MIN_NEGATIVI:
        borderline = no_rows + ungated[:N_BORDERLINE_EXTRA]

    candidates = si_rows + borderline
    for r in candidates:
        c = cache.get(r["titolo"], {})
        r["full_text"] = c.get("text", "")
    return candidates


def load_full_corpus(cfg):
    """Come load_candidates(), ma SENZA filtro: ogni riga di ranking.csv (l'intero corpus
    classificato dal kNN, non solo i ~150 già passati dal gate). Serve per lo studio
    "LLM sull'intero corpus vs kNN": rivalutare solo il sottoinsieme già scelto dal kNN non
    scoprirebbe mai un articolo buono che il kNN ha escluso a monte. Stessa cache testo,
    nessuno scraping nuovo."""
    ranking = list(csv.DictReader(open(cfg["data_dir"] / "ranking.csv", encoding="utf-8")))
    cache = json.load(open(cfg["data_dir"].parent / "models" / "corpus_fulltext_cache.json", encoding="utf-8"))
    for r in ranking:
        c = cache.get(r["titolo"], {})
        r["full_text"] = c.get("text", "")
    return ranking


def score_candidates(candidates, desc, criteria, obs_name, model, limit=None):
    """Punteggio LLM indipendente (0-100) per ogni candidato. Stesso schema del gate
    esistente: ThreadPoolExecutor + retry con attesa crescente su errore/429."""
    client = llm_client()
    rows = candidates[:limit] if limit else candidates
    lock = threading.Lock()
    tok = {"in": 0, "out": 0}

    def one(r):
        body = (r.get("full_text") or r.get("summary") or "")[:6000]
        prompt = PROMPT.format(obs_name=obs_name, desc=desc, criteria=criteria, title=r["titolo"], body=body)
        for attempt in range(5):
            try:
                resp = client.responses.create(model=model, input=prompt)
                txt = resp.output_text.strip()
                if txt.startswith("```"):
                    txt = txt.split("```")[1].removeprefix("json").strip()
                res = json.loads(txt)
                with lock:
                    tok["in"] += resp.usage.input_tokens
                    tok["out"] += resp.usage.output_tokens
                return {"punteggio": float(res.get("punteggio", 0)), "motivo": res.get("motivo", "")}
            except Exception as e:
                if attempt == 4:
                    log(f"    errore su '{r['titolo'][:50]}': {e}")
                    return {"punteggio": None, "motivo": ""}
                time.sleep(2 ** attempt)
        return {"punteggio": None, "motivo": ""}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, rows))
    for r, res in zip(rows, results):
        r["llm_score"] = res["punteggio"]
        r["llm_motivo"] = res["motivo"]
    return rows, tok


def compute_knn_ranks(rows):
    """Rank kNN ricalcolato 1..N dentro il solo sottoinsieme dei candidati passati (non
    il rank globale sul corpus intero) — ignora la divisione visiva Sì/No del foglio
    (quella è una decisione del gate, non del ranker) e riordina per score_ranker puro."""
    by_knn = sorted(rows, key=lambda r: -float(r["score_ranker"]))
    for i, r in enumerate(by_knn, 1):
        r["rank_knn_local"] = i
    return rows


def compute_llm_ranks(rows):
    """Rank LLM (1..N nello stesso sottoinsieme) e delta, da `llm_score` appena calcolato.
    Va chiamata dopo compute_knn_ranks."""
    scored = [r for r in rows if r.get("llm_score") is not None]
    by_llm = sorted(scored, key=lambda r: -r["llm_score"])
    for i, r in enumerate(by_llm, 1):
        r["rank_llm_local"] = i
    for r in rows:
        if r.get("llm_score") is None:
            r["rank_llm_local"] = None
        r["delta_rank"] = (r["rank_knn_local"] - r["rank_llm_local"]) if r.get("rank_llm_local") else None
    return rows


def read_existing_llm_ranks(xlsx_path, rows):
    """Legge Rank LLM / Δ posizione già scritti in un giro precedente (colonne L/M del
    foglio Da_revisionare — K è "Confidence LLM" del gate di produzione), abbinando per
    link — NESSUNA chiamata LLM. Usata da --cluster-sheet-only per costruire Analisi_Cluster
    senza rispendere token."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Da_revisionare"]
    by_link = {}
    for r in range(1, ws.max_row + 1):
        c9 = ws.cell(row=r, column=9)
        if not c9.hyperlink:
            continue
        by_link[c9.hyperlink.target] = (ws.cell(row=r, column=12).value, ws.cell(row=r, column=13).value)

    found = 0
    for r in rows:
        vals = by_link.get(r["link"])
        if vals and vals[0] not in (None, ""):
            r["rank_llm_local"] = int(vals[0])
            r["delta_rank"] = int(vals[1]) if vals[1] not in (None, "") else None
            found += 1
        else:
            r["rank_llm_local"] = None
            r["delta_rank"] = None
    return found


def write_llm_columns_into_sheet(xlsx_path, sheet_name, rows, model, all_candidates_expected=True):
    """Scrive Rank LLM / Δ posizione in coda (colonne L, M — K è "Confidence LLM" del gate
    di produzione) in un foglio già esistente (Da_revisionare O il ranking completo
    Ranking_X_nuovo), abbinando per link (colonna I, dove il link e' un hyperlink — non
    testo grezzo). Non tocca nessun'altra colonna o riga.

    In Da_revisionare tutte le righe hanno un punteggio LLM (stessa popolazione). Nel
    ranking completo NO — lì ci sono ~1400 righe ma l'LLM ne ha valutate solo ~150-180
    (quelle già passate dal gate): per le altre le due colonne restano vuote apposta,
    non è un errore."""
    by_link = {r["link"]: r for r in rows}

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb[sheet_name]

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")

    matched, unmatched, header_rows = 0, 0, []
    for r in range(1, ws.max_row + 1):
        c1 = ws.cell(row=r, column=1).value
        if c1 == "Rank":  # riga di intestazione colonne (una o più, a seconda del foglio)
            header_rows.append(r)
            continue
        c9 = ws.cell(row=r, column=9)
        if not c9.hyperlink:
            continue  # riga di nota/sezione/cluster-header, non un candidato
        link = c9.hyperlink.target
        row = by_link.get(link)
        if not row:
            # normale nel ranking completo (l'LLM ha valutato solo i candidati del gate);
            # pulisce eventuali valori residui da un giro precedente con altri candidati.
            ws.cell(row=r, column=12, value="")
            ws.cell(row=r, column=13, value="")
            unmatched += 1
            continue
        ws.cell(row=r, column=12, value=row["rank_llm_local"] if row["rank_llm_local"] else "")
        ws.cell(row=r, column=13, value=row["delta_rank"] if row["delta_rank"] is not None else "")
        matched += 1

    if all_candidates_expected and unmatched:
        log(f"  ATTENZIONE: {unmatched} righe di {sheet_name} non abbinate a un punteggio LLM (inatteso qui)")

    for hr in header_rows:
        for col, label in ((12, "Rank LLM"), (13, "Δ posizione")):
            cell = ws.cell(row=hr, column=col, value=label)
            cell.font = hf
            cell.fill = hd

    ws.column_dimensions["L"].width = 12
    ws.column_dimensions["M"].width = 12

    scope = ("sugli stessi candidati" if all_candidates_expected else
             f"sui {matched} candidati già passati dal gate — vuoto per gli altri, mai valutati dall'LLM")
    note = f"  [Esperimento: colonne L/M = ranking indipendente ({model}) {scope} (positivo = LLM lo valuta più in alto del kNN)."
    a1 = ws["A1"].value or ""
    if "[Esperimento:" not in a1:
        ws["A1"] = a1 + note

    wb.save(xlsx_path)
    return matched


def write_cluster_sheet(xlsx_path, rows, model):
    """Foglio dedicato 'Analisi_Cluster': SOLO i sospetti duplicati (cluster con 2+ membri
    tra i candidati), con le posizioni secondo le due classifiche affiancate — apposta
    FUORI da Da_revisionare, che deve restare il foglio pulito dove il ricercatore giudica
    solo Sì/No. Qui invece si guarda come kNN e LLM ordinano ogni cluster e se sono
    d'accordo o no."""
    clustered = [r for r in rows if r.get("cluster_id")]
    by_cluster = {}
    for r in clustered:
        by_cluster.setdefault(r["cluster_id"], []).append(r)
    by_cluster = {cid: m for cid, m in by_cluster.items() if len(m) >= 2}

    wb = openpyxl.load_workbook(xlsx_path)
    SHEET = "Analisi_Cluster"
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    section_fill = PatternFill("solid", fgColor="D9D9D9")

    ws["A1"] = (f"Sospetti duplicati: come li ordinano kNN e LLM ({model}) indipendentemente, "
                f"posizioni dentro i {len(rows)} candidati del foglio Da_revisionare. "
                f"Δ = rank kNN − rank LLM. Non serve giudicare Sì/No qui — se ne occupa Da_revisionare.")
    ws["A1"].font = Font(bold=True, size=12)

    heads = ["Titolo", "Fonte", "Rank kNN", "Rank LLM", "Δ posizione", "Link"]
    row = 3
    for cid in sorted(by_cluster, key=lambda c: min(m["rank_knn_local"] for m in by_cluster[c])):
        members = sorted(by_cluster[cid], key=lambda r: r["rank_knn_local"])
        deltas = [abs(m["delta_rank"]) for m in members if m.get("delta_rank") is not None]
        if not deltas:
            accordo = "Rank LLM non disponibile per questo cluster"
        elif max(deltas) <= 10:
            accordo = "kNN e LLM concordano"
        elif max(deltas) >= 40:
            accordo = "forte disaccordo tra kNN e LLM"
        else:
            accordo = "disaccordo moderato"

        title_cell = ws.cell(row=row, column=1, value=f"Cluster {cid} — {len(members)} fonti — {accordo}")
        title_cell.font = Font(bold=True)
        for c in range(1, len(heads) + 1):
            ws.cell(row=row, column=c).fill = section_fill
        row += 1
        for c, h in enumerate(heads, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = hf
            cell.fill = hd
        row += 1
        cf = cluster_fill(cid)
        for m in members:
            t = ws.cell(row=row, column=1, value=m["titolo"])
            t.fill = cf
            ws.cell(row=row, column=2, value=m["fonte"])
            ws.cell(row=row, column=3, value=m["rank_knn_local"])
            ws.cell(row=row, column=4, value=m.get("rank_llm_local") or "")
            ws.cell(row=row, column=5, value=m.get("delta_rank") if m.get("delta_rank") is not None else "")
            link_cell = ws.cell(row=row, column=6, value="apri articolo")
            link_cell.hyperlink = m["link"]
            link_cell.font = Font(color="0563C1", underline="single")
            row += 1
        row += 1  # riga vuota tra un cluster e l'altro

    widths = {"A": 55, "B": 22, "C": 10, "D": 10, "E": 12, "F": 14}
    for col, wd in widths.items():
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "A3"

    wb.save(xlsx_path)
    return len(by_cluster)


def write_cost_sheet(xlsx_path, name, model, llm_only_tok, n_candidates):
    """Foglio 'Costi': confronta il costo REALE della pipeline di produzione (embedding
    sull'intero corpus + gate LLM sui soli top-150) col costo REALE misurato qui
    (LLM indipendente sui candidati) e una PROIEZIONE (non eseguita) di quanto
    costerebbe rimpiazzare tutta la pipeline con un ranking-solo-LLM sull'intero corpus.
    Nessuna chiamata nuova: usa solo numeri già misurati."""
    prod = PROD_COST[name]
    corpus_size = prod["corpus_size"]

    # proiezione: costo medio per articolo misurato sui candidati -> esteso a tutto il corpus
    proj_in = round(llm_only_tok["in"] / n_candidates * corpus_size)
    proj_out = round(llm_only_tok["out"] / n_candidates * corpus_size)

    prod_total = prod["embed_tokens"] + prod["gate_in"] + prod["gate_out"]
    proj_total = proj_in + proj_out
    delta_pct = (proj_total - prod_total) / prod_total * 100

    wb = openpyxl.load_workbook(xlsx_path)
    SHEET = "Costi"
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    tot_fill = PatternFill("solid", fgColor="D9D9D9")
    warn_fill = PatternFill("solid", fgColor="FFF2CC")

    def eur(tin, tout, is_embed):
        if is_embed:
            return tin * PRICE_EUR_PER_1M["embed_in"] / 1_000_000
        return (tin * PRICE_EUR_PER_1M["gpt54_in"] + tout * PRICE_EUR_PER_1M["gpt54_out"]) / 1_000_000

    prod_eur = eur(prod["embed_tokens"], 0, True) + eur(prod["gate_in"], prod["gate_out"], False)
    proj_eur = eur(proj_in, proj_out, False)  # la proiezione sostituisce l'embedding col solo-LLM

    ws["A1"] = (f"Costo della pipeline attuale vs un ipotetico ranking-solo-LLM ({model}) su tutto il "
                f"corpus ({corpus_size} articoli). Righe 1-2 = costo REALE di produzione. Riga 3 = costo "
                f"REALE misurato qui su {n_candidates} candidati. Riga 4 = PROIEZIONE, non eseguita per "
                f"davvero — estrapolata dal costo medio/articolo di riga 3. Tariffe: gpt-5.4 €2,50/€15,00 "
                f"per milione di token in/out (confermate); embedding €0,13/milione (prezzo pubblico "
                f"text-embedding-3-large, NON verificato sul tuo contratto Azure specifico).")
    ws["A1"].font = Font(bold=True, size=12)
    ws["A1"].alignment = openpyxl.styles.Alignment(wrap_text=True)
    ws.row_dimensions[1].height = 60

    heads = ["Fase", "Token input", "Token output", "Totale token", "Costo (€)", "Nota"]
    for c, h in enumerate(heads, 1):
        cell = ws.cell(row=3, column=c, value=h)
        cell.font = hf
        cell.fill = hd

    data = [
        ("1. Embedding intero corpus (ranker kNN)", prod["embed_tokens"], 0, True,
         f"text-embedding-3-large, {corpus_size} articoli — reale"),
        ("2. Gate LLM (solo i migliori 150)", prod["gate_in"], prod["gate_out"], False,
         "gpt-5.4, pipeline di produzione attuale — reale"),
        ("3. Rank LLM indipendente sui candidati già passati dal gate", llm_only_tok["in"], llm_only_tok["out"], False,
         f"{model}, {n_candidates} articoli — reale, quello di questa analisi"),
        ("4. PROIEZIONE: rank LLM su TUTTO il corpus, al posto di 1+2", proj_in, proj_out, False,
         f"{model}, {corpus_size} articoli — NON eseguito, stimato da riga 3"),
    ]
    row = 4
    for label, tin, tout, is_embed, note in data:
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=tin)
        ws.cell(row=row, column=3, value=tout)
        ws.cell(row=row, column=4, value=tin + tout)
        ws.cell(row=row, column=5, value=round(eur(tin, tout, is_embed), 4))
        ws.cell(row=row, column=6, value=note)
        if label.startswith("4."):
            for c in range(1, 7):
                ws.cell(row=row, column=c).fill = warn_fill
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Totale pipeline attuale (1+2)").font = Font(bold=True)
    ws.cell(row=row, column=4, value=prod_total).font = Font(bold=True)
    ws.cell(row=row, column=5, value=round(prod_eur, 2)).font = Font(bold=True)
    for c in range(1, 7):
        ws.cell(row=row, column=c).fill = tot_fill
    row += 1
    ws.cell(row=row, column=1, value="Totale proiezione solo-LLM su tutto (4)").font = Font(bold=True)
    ws.cell(row=row, column=4, value=proj_total).font = Font(bold=True)
    ws.cell(row=row, column=5, value=round(proj_eur, 2)).font = Font(bold=True)
    for c in range(1, 7):
        ws.cell(row=row, column=c).fill = tot_fill
    row += 1
    eur_delta_pct = (proj_eur - prod_eur) / prod_eur * 100
    ws.cell(row=row, column=1,
            value=(f"Differenza: la proiezione costerebbe {proj_eur - prod_eur:+.2f}€ in più "
                   f"({eur_delta_pct:+.0f}% in €, contro {delta_pct:+.0f}% se si guardano solo i token — "
                   f"il divario in € è più marcato perché l'LLM costa molto di più dell'embedding per token)"
                   )).font = Font(bold=True, italic=True)

    row += 2
    ws.cell(row=row, column=1,
            value=("Tariffe usate: gpt-5.4 = €2,50/milione token input, €15,00/milione output (confermate "
                   "dall'utente). Embedding text-embedding-3-large = €0,13/milione — prezzo pubblico di listino, "
                   "NON confrontato con il contratto Azure effettivo: se il tuo prezzo è diverso, correggilo in "
                   "PRICE_EUR_PER_1M dentro llm_only_rank.py e rilancia con --cluster-sheet-only (zero costo)."))
    ws.cell(row=row, column=1).font = Font(italic=True, size=10, color="6B6F76")
    ws.cell(row=row, column=1).alignment = openpyxl.styles.Alignment(wrap_text=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    ws.row_dimensions[row].height = 60

    widths = {"A": 55, "B": 14, "C": 14, "D": 14, "E": 12, "F": 50}
    for col, wd in widths.items():
        ws.column_dimensions[col].width = wd

    wb.save(xlsx_path)


def summarize_pair(rows, key_a, key_b):
    """Statistiche aggregate sul confronto tra due ranking (rank_* già calcolati su rows,
    es. rank_knn_local vs rank_llm__gpt-5.4): media/mediana |Δ|, % concordanti (|Δ|≤10) e
    in forte disaccordo (|Δ|≥40) — stesse soglie già usate in write_cluster_sheet() per
    l'etichetta "accordo" per cluster, qui sull'intero corpus. `corr` è Pearson sui RANK
    (equivale a Spearman sui punteggi originali, senza bisogno di scipy); `tau` è Kendall's
    tau-b (scipy) — concordanza ordinale a coppie, più robusta di Spearman quando i Δ grandi
    sono pochi ma vistosi (Spearman li pesa quadraticamente, Kendall linearmente)."""
    from scipy.stats import kendalltau
    pairs = [(r[key_a], r[key_b]) for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    n = len(pairs)
    if n < 2:
        return None
    a_arr = np.array([a for a, _ in pairs], dtype=float)
    b_arr = np.array([b for _, b in pairs], dtype=float)
    deltas = np.abs(a_arr - b_arr)
    tau, tau_p = kendalltau(a_arr, b_arr)
    return dict(
        n=n,
        mean_abs_delta=float(deltas.mean()),
        median_abs_delta=float(np.median(deltas)),
        pct_agree=float((deltas <= 10).mean() * 100),
        pct_disagree=float((deltas >= 40).mean() * 100),
        corr=float(np.corrcoef(a_arr, b_arr)[0, 1]),
        tau=float(tau),
        tau_pvalue=float(tau_p),
    )


def summarize_top_union_tau(rows, key_a, key_b, n=300):
    """Kendall's tau-b ristretto all'unione dei top-n di CIASCUNA delle due classifiche (non
    solo il top-n di una delle due, per non scegliere aprioristicamente un ranking come
    riferimento) — più informativo di summarize_pair() sull'intero corpus, dove l'accordo
    banale sulla lunga coda di articoli chiaramente irrilevanti (entrambi i metodi li
    piazzano in fondo) diluisce il segnale che conta davvero: l'accordo vicino al taglio
    operativo. Tau è invariante a trasformazioni monotone del rank, quindi non serve
    ricalcolare rank locali dentro il sottoinsieme: i rank originali bastano per l'ordine
    relativo a coppie."""
    from scipy.stats import kendalltau
    pairs = [(r[key_a], r[key_b]) for r in rows
             if r.get(key_a) is not None and r.get(key_b) is not None
             and (r[key_a] <= n or r[key_b] <= n)]
    if len(pairs) < 2:
        return None
    a_arr = np.array([a for a, _ in pairs], dtype=float)
    b_arr = np.array([b for _, b in pairs], dtype=float)
    tau, tau_p = kendalltau(a_arr, b_arr)
    return dict(n=len(pairs), n_top=n, tau=float(tau), tau_pvalue=float(tau_p))


def summarize_top_n(rows, model, pool_size=150, knn_key="rank_knn_local", agree_thresh=10, disagree_thresh=40):
    """Confronto ristretto ai primi `pool_size` per kNN (oggi: il pool che arriva davvero al
    gate di produzione) — più rilevante del confronto sull'intero corpus (summarize_pair),
    perché è la popolazione su cui la pipeline decide realmente. Due numeri:
    - sovrapposizione: quanti dei top-`pool_size` kNN sono ANCHE nei top-`pool_size` GLOBALI
      del modello (stesso corpus intero, non ricalcolato).
    - Δ locale: il modello viene riordinato SOLO dentro questi articoli (rank locale
      ricalcolato dal rank globale già noto — riordinare per rank globale dentro un
      sottoinsieme preserva l'ordine relativo, nessun nuovo punteggio necessario) e
      confrontato col rank kNN, che è già 1..`pool_size` per costruzione su questo
      sottoinsieme. Il dict restituito usa "n" per il conteggio effettivo (coerente con
      summarize_pair) e "n_pool" per la soglia richiesta — tenerli distinti perché possono
      differire se qualche articolo non ha un punteggio del modello."""
    top = [r for r in rows if r.get(knn_key) and r[knn_key] <= pool_size]
    if not top:
        return None
    key = f"rank_llm__{model}"
    withval = [r for r in top if r.get(key)]
    if not withval:
        return None
    overlap_pct = sum(1 for r in top if r.get(key) and r[key] <= pool_size) / len(top) * 100
    ranked = sorted(withval, key=lambda r: r[key])
    local_rank = {id(r): i + 1 for i, r in enumerate(ranked)}
    deltas = np.array([abs(r[knn_key] - local_rank[id(r)]) for r in withval], dtype=float)
    return dict(
        n=len(deltas), n_pool=pool_size, overlap_pct=overlap_pct,
        mean_abs_delta=float(deltas.mean()),
        median_abs_delta=float(np.median(deltas)),
        pct_agree=float((deltas <= agree_thresh).mean() * 100),
        pct_disagree=float((deltas >= disagree_thresh).mean() * 100),
    )


def sweep_weight(rows, llm_score_key, ground_truth_key="pertinenza_llm"):
    """Cerca il peso w che massimizza l'AUC di combined = w*score_ranker_norm +
    (1-w)*llm_score_norm contro il Sì/No del gate di produzione — SOLO sui candidati che
    hanno un giudizio del gate (gli altri non hanno un riferimento binario). Normalizzazione
    min-max dentro l'insieme valutato (stessa idea dei rank "locali" già usati altrove).
    LIMITE dichiarato: questo è l'unico riferimento binario disponibile oggi — nessuna
    revisione umana ("Parere ricercatore") è ancora stata compilata in nessuna pipeline, va
    trattato come un controllo parziale, non una validazione indipendente."""
    from sklearn.metrics import roc_auc_score
    labeled = [r for r in rows if r.get(ground_truth_key) in ("Si", "No") and r.get(llm_score_key) is not None]
    if len(labeled) < 10:
        return None
    y = np.array([1 if r[ground_truth_key] == "Si" else 0 for r in labeled])
    if len(set(y.tolist())) < 2:
        return None
    knn = np.array([float(r["score_ranker"]) for r in labeled])
    llm = np.array([float(r[llm_score_key]) for r in labeled])

    def norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    knn_n, llm_n = norm(knn), norm(llm)
    curve = []
    for i in range(21):
        w = i / 20
        auc = roc_auc_score(y, w * knn_n + (1 - w) * llm_n)
        curve.append((w, auc))
    best_w, best_auc = max(curve, key=lambda t: t[1])
    return dict(n=len(labeled), best_w=best_w, best_auc=best_auc,
                auc_knn_only=curve[-1][1], auc_llm_only=curve[0][1])


def write_full_corpus_comparison_sheet(xlsx_path, pipeline_name, rows, models, precomputed_weights=None):
    """Foglio dedicato 'LLM_vs_kNN_corpus_intero': confronto riga per riga tra il rank kNN
    e il rank di ciascun modello LLM valutato sull'INTERO corpus (non sul sottoinsieme già
    gated) — a differenza di write_llm_columns_into_sheet(), pensata per i soli candidati
    già passati dal gate. In cima, quattro blocchi in ordine: (1) statistiche aggregate per
    coppia sull'intero corpus (media/mediana |Δ|, % concordi/discordi, correlazione), (2) lo
    stesso confronto ristretto ai primi 150 per kNN — il pool che arriva davvero al gate
    oggi, più rilevante del confronto (1) — con la sovrapposizione tra i due top-150
    (summarize_top_n), (3) Kendall's tau-b per coppia, sia sull'intero corpus sia ristretto
    all'unione dei primi 300 per kNN o per LLM (summarize_top_union_tau) — complementare a
    (1)/(2), non li sostituisce, (4) il peso ottimo kNN+LLM (solo sui candidati con un
    giudizio gate, vedi limite in sweep_weight). `precomputed_weights` (opzionale): se dato
    (dict {modello: risultato di sweep_weight}), usa questi valori per il blocco (4) invece
    di richiamare sweep_weight() da zero — serve a refresh_full_corpus_sheet() per rigenerare
    il foglio con blocchi nuovi (es. il tau) SENZA richiamare l'LLM, dato che il punteggio
    grezzo 0-100 necessario a sweep_weight() non è ripersistito da nessuna parte."""
    wb = openpyxl.load_workbook(xlsx_path)
    SHEET = "LLM_vs_kNN_corpus_intero"
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    note_fill = PatternFill("solid", fgColor="FFF2CC")
    # Confronto completo a coppie (kNN vs ciascun modello + ogni modello contro ogni altro),
    # non solo il caso speciale a 2 modelli — regge con qualsiasi N.
    pairs = [("kNN", m) for m in models] + list(itertools.combinations(models, 2))
    heads = ["Titolo", "Fonte", "Rank kNN"] + [f"Rank LLM ({m})" for m in models] \
        + [f"Δ {a}-{b}" for a, b in pairs] + ["Link"]
    n_cols = len(heads)

    row = 1
    ws.cell(row=row, column=1, value=(
        f"Ranking LLM sull'INTERO corpus ({pipeline_name}, {len(rows)} articoli) vs ranking "
        f"kNN attuale. Modelli: {', '.join(models)}. LIMITE: il \"peso ottimo\" sotto è "
        f"calcolato solo sui candidati già passati dal gate di produzione (unico "
        f"riferimento Sì/No disponibile oggi — zero righe \"Parere ricercatore\" compilate "
        f"a mano finora in nessuna pipeline) — è un controllo parziale, non una validazione "
        f"indipendente."
    )).font = Font(bold=True, size=11)
    ws.cell(row=row, column=1).alignment = openpyxl.styles.Alignment(wrap_text=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)
    ws.row_dimensions[row].height = 60
    row += 2

    def rank_key(name):
        return "rank_knn_local" if name == "kNN" else f"rank_llm__{name}"

    stats_by_pair = {}
    for a, b in pairs:
        stats = summarize_pair(rows, rank_key(a), rank_key(b))
        stats_by_pair[(a, b)] = stats
        if not stats:
            continue
        ws.cell(row=row, column=1, value=f"{a} vs {b}  (n={stats['n']})").font = Font(bold=True)
        ws.cell(row=row, column=2, value=f"media |Δ|={stats['mean_abs_delta']:.1f}")
        ws.cell(row=row, column=3, value=f"mediana |Δ|={stats['median_abs_delta']:.1f}")
        ws.cell(row=row, column=4, value=f"concordi (|Δ|≤10)={stats['pct_agree']:.0f}%")
        ws.cell(row=row, column=5, value=f"forte disaccordo (|Δ|≥40)={stats['pct_disagree']:.0f}%")
        ws.cell(row=row, column=6, value=f"correlazione={stats['corr']:.2f}")
        row += 1

    row += 1
    ws.cell(row=row, column=1, value=(
        "Confronto ristretto ai primi 150 per kNN — il pool che arriva DAVVERO al gate di "
        "produzione oggi (più rilevante del confronto sull'intero corpus qui sopra)."
    )).font = Font(bold=True, size=12)
    row += 1
    for model in models:
        t = summarize_top_n(rows, model, pool_size=150)
        if not t:
            continue
        ws.cell(row=row, column=1, value=f"kNN vs {model}  (n={t['n']} su {t['n_pool']})").font = Font(bold=True)
        ov_cell = ws.cell(row=row, column=2, value=f"sovrapposizione top-150 = {t['overlap_pct']:.0f}%")
        ov_cell.font = Font(bold=True, color="B8623E")
        ws.cell(row=row, column=3, value=f"media |Δ| locale={t['mean_abs_delta']:.1f}")
        ws.cell(row=row, column=4, value=f"mediana={t['median_abs_delta']:.1f}")
        ws.cell(row=row, column=5, value=f"concordi(≤10)={t['pct_agree']:.0f}%")
        ws.cell(row=row, column=6, value=f"forte disacc.(≥40)={t['pct_disagree']:.0f}%")
        row += 1

    row += 1
    ws.cell(row=row, column=1, value=(
        "Concordanza ordinale — Kendall's tau-b (1 = ordine identico, 0 = nessuna relazione, "
        "-1 = ordine invertito). Colonna 2: sull'intero corpus (attenzione: qui l'accordo "
        "banale sulla lunga coda di articoli chiaramente irrilevanti a entrambi i metodi "
        "gonfia il numero). Colonna 3: ristretto all'unione dei primi 300 per kNN O per LLM "
        "(non solo uno dei due) — più rilevante perché guarda solo dove la decisione conta."
    )).font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).alignment = openpyxl.styles.Alignment(wrap_text=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)
    ws.row_dimensions[row].height = 45
    row += 1
    for a, b in pairs:
        stats = stats_by_pair.get((a, b))
        tu = summarize_top_union_tau(rows, rank_key(a), rank_key(b), n=300)
        if not stats:
            continue
        ws.cell(row=row, column=1, value=f"{a} vs {b}").font = Font(bold=True)
        ws.cell(row=row, column=2, value=f"tau intero corpus={stats['tau']:.2f}  (n={stats['n']})")
        if tu:
            ws.cell(row=row, column=3, value=f"tau top-300 unione={tu['tau']:.2f}  (n={tu['n']})")
        row += 1

    row += 1
    for model in models:
        # se un peso già calcolato è stato passato (refresh_full_corpus_sheet, zero
        # chiamate LLM: il punteggio grezzo 0-100 non è ripersistito da nessuna parte, quindi
        # senza questo non ci sarebbe modo di ricalcolarlo senza rispendere), usalo; altrimenti
        # ricalcola da rows (giro fresco, llm_score__<model> presente sui dict appena scorati).
        w = (precomputed_weights or {}).get(model) if precomputed_weights is not None else sweep_weight(rows, f"llm_score__{model}")
        if not w:
            ws.cell(row=row, column=1,
                    value=f"Peso ottimo kNN+{model}: dati insufficienti (nessun giudizio gate su questo corpus)")
        else:
            ws.cell(row=row, column=1, value=(
                f"Peso ottimo kNN+{model} (su {w['n']} candidati con giudizio gate): "
                f"w={w['best_w']:.2f} → combinato AUC={w['best_auc']:.3f}  |  "
                f"solo kNN (w=1) AUC={w['auc_knn_only']:.3f}  |  "
                f"solo LLM (w=0) AUC={w['auc_llm_only']:.3f}"
            ))
        ws.cell(row=row, column=1).font = Font(italic=True)
        ws.cell(row=row, column=1).fill = note_fill
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=n_cols)
        row += 1

    row += 1
    for c, h in enumerate(heads, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    header_row = row
    row += 1

    for r in sorted(rows, key=lambda r: r["rank_knn_local"]):
        c = 1
        ws.cell(row=row, column=c, value=r["titolo"]); c += 1
        ws.cell(row=row, column=c, value=r["fonte"]); c += 1
        ws.cell(row=row, column=c, value=r["rank_knn_local"]); c += 1
        for m in models:
            rk = r.get(f"rank_llm__{m}")
            ws.cell(row=row, column=c, value=rk if rk else "")
            c += 1
        for a, b in pairs:
            va, vb = r.get(rank_key(a)), r.get(rank_key(b))
            ws.cell(row=row, column=c, value=(va - vb) if (va and vb) else "")
            c += 1
        link_cell = ws.cell(row=row, column=c, value="apri articolo" if r["link"] else "")
        if r["link"]:
            link_cell.hyperlink = r["link"]
            link_cell.font = Font(color="0563C1", underline="single")
        row += 1

    ws.freeze_panes = f"A{header_row + 1}"
    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 22

    wb.save(xlsx_path)
    return len(rows)


# Riconosce le righe "Peso ottimo kNN+<modello> (...): w=... → combinato AUC=... | solo
# kNN (w=1) AUC=... | solo LLM (w=0) AUC=..." scritte da write_full_corpus_comparison_sheet.
WEIGHT_LINE_RE = re.compile(
    r"Peso ottimo kNN\+(?P<model>.+?) \(su (?P<n>\d+) candidati.*?\): "
    r"w=(?P<w>[\d.]+) → combinato AUC=(?P<auc>[\d.]+)\s*\|\s*"
    r"solo kNN \(w=1\) AUC=(?P<auc_knn>[\d.]+)\s*\|\s*solo LLM \(w=0\) AUC=(?P<auc_llm>[\d.]+)"
)


def read_existing_rank_table(xlsx_path, models):
    """Rilegge la tabella riga-per-articolo già scritta da write_full_corpus_comparison_sheet
    dentro il foglio 'LLM_vs_kNN_corpus_intero' — SENZA richiamare l'LLM. Ogni riga: titolo,
    fonte, link, rank_knn_local, rank_llm__<modello> per ciascun modello in `models` (chiave
    assente/None se quel modello non è (ancora) nel foglio). Riusata sia per rileggere le
    statistiche aggregate (read_pipeline_study_stats) sia per aggiungere un modello nuovo a
    un foglio già scritto (run_add_model_study) senza rispendere sui modelli già fatti."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["LLM_vs_kNN_corpus_intero"]

    header_row = None
    for r in range(1, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == "Titolo":
            header_row = r
            break
    if header_row is None:
        raise ValueError(f"foglio 'LLM_vs_kNN_corpus_intero' in {xlsx_path.name} senza intestazione tabella")
    heads = [ws.cell(row=header_row, column=c).value for c in range(1, ws.max_column + 1)]
    link_col = heads.index("Link") + 1 if "Link" in heads else None

    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        if not ws.cell(row=r, column=1).value:
            continue
        vals = {h: ws.cell(row=r, column=c).value for c, h in enumerate(heads, 1)}
        row = {
            "titolo": vals.get("Titolo"),
            "fonte": vals.get("Fonte"),
            "link": ws.cell(row=r, column=link_col).hyperlink.target if link_col and ws.cell(row=r, column=link_col).hyperlink else "",
            "rank_knn_local": vals.get("Rank kNN"),
        }
        for m in models:
            row[f"rank_llm__{m}"] = vals.get(f"Rank LLM ({m})")
        rows.append(row)
    return rows


def read_existing_weights(xlsx_path):
    """Rilegge le righe testuali "Peso ottimo kNN+<modello> (...)" già scritte in cima al
    foglio 'LLM_vs_kNN_corpus_intero' da write_full_corpus_comparison_sheet (WEIGHT_LINE_RE)
    — SENZA richiamare l'LLM. Il punteggio grezzo 0-100 di sweep_weight() non è ripersistito
    da nessuna parte, quindi rileggere questo testo deterministico è l'UNICO modo di
    recuperare peso/AUC di un modello già valutato senza rispendere da zero. Ritorna
    {modello: dict(n, best_w, best_auc, auc_knn_only, auc_llm_only)}."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["LLM_vs_kNN_corpus_intero"]
    header_row = next(r for r in range(1, ws.max_row + 1) if ws.cell(row=r, column=1).value == "Titolo")
    weights = {}
    for r in range(1, header_row):
        val = ws.cell(row=r, column=1).value
        if not val or "Peso ottimo" not in str(val):
            continue
        m = WEIGHT_LINE_RE.search(str(val))
        if m:
            weights[m.group("model")] = dict(
                n=int(m.group("n")), best_w=float(m.group("w")), best_auc=float(m.group("auc")),
                auc_knn_only=float(m.group("auc_knn")), auc_llm_only=float(m.group("auc_llm")),
            )
    return weights


def refresh_full_corpus_sheet(name, models):
    """Rigenera il foglio 'LLM_vs_kNN_corpus_intero' di una pipeline con l'ultima versione
    dei blocchi riassuntivi scritti da write_full_corpus_comparison_sheet (es. il blocco tau,
    aggiunto DOPO l'ultimo giro reale con l'LLM) — SENZA richiamare l'LLM: rilegge i rank già
    a foglio (read_existing_rank_table) e il peso ottimo già scritto (read_existing_weights,
    testo deterministico, il punteggio grezzo non è ripersistito), poi riscrive lo stesso
    foglio. Zero chiamate, zero nuovo costo — serve esattamente per far apparire una metrica
    nuova nei file già prodotti senza rilanciare lo studio completo."""
    cfg = PIPELINES[name]
    rows = read_existing_rank_table(cfg["xlsx_revisione"], models)
    weights = read_existing_weights(cfg["xlsx_revisione"])
    n_written = write_full_corpus_comparison_sheet(cfg["xlsx_revisione"], name, rows, models,
                                                     precomputed_weights=weights)
    log(f"  foglio 'LLM_vs_kNN_corpus_intero' rigenerato in {cfg['xlsx_revisione'].name} "
        f"(zero chiamate LLM): {n_written} righe")
    return n_written


def read_pipeline_study_stats(xlsx_path, models):
    """Rilegge le statistiche di un foglio 'LLM_vs_kNN_corpus_intero' già scritto da
    write_full_corpus_comparison_sheet(), SENZA richiamare l'LLM. Ritorna
    {n_corpus, pairs, top150, top_union_tau, weights}. `pairs` (confronto sull'intero corpus,
    incluso Kendall's tau) e `top150` (confronto ristretto ai primi 150 per kNN, vedi
    summarize_top_n) sono RICALCOLATI dalle colonne numeriche Rank kNN/Rank LLM della tabella
    (robusto, stessa aritmetica delle funzioni omonime); `top_union_tau` allo stesso modo, via
    summarize_top_union_tau (unione dei primi 300 per kNN o per LLM). `weights` (peso
    ottimo/AUC) invece non è ripersistito in forma numerica pulita (servirebbe rilanciare
    l'LLM per ricalcolarlo da zero): si rilegge il testo deterministico che questo stesso
    script ha scritto nelle righe "Peso ottimo ..." in cima al foglio."""
    rows = read_existing_rank_table(xlsx_path, models)

    def rank_key(name):
        return "rank_knn_local" if name == "kNN" else f"rank_llm__{name}"

    pairs = {}
    for a, b in [("kNN", m) for m in models] + list(itertools.combinations(models, 2)):
        pairs[(a, b)] = summarize_pair(rows, rank_key(a), rank_key(b))

    # confronto ristretto ai primi 150 per kNN — ricalcolato dagli stessi rank numerici già
    # letti in `rows`, nessun parsing di testo necessario (a differenza del peso ottimo sotto).
    top150 = {}
    for m in models:
        t = summarize_top_n(rows, m, pool_size=150)
        if t:
            top150[m] = t

    # tau ristretto all'unione dei primi 300 per kNN o per LLM — stessa idea di top150, ma
    # con Kendall's tau invece di sovrapposizione/Δ (vedi summarize_top_union_tau).
    top_union_tau = {}
    for a, b in [("kNN", m) for m in models] + list(itertools.combinations(models, 2)):
        tu = summarize_top_union_tau(rows, rank_key(a), rank_key(b), n=300)
        if tu:
            top_union_tau[(a, b)] = tu

    weights = read_existing_weights(xlsx_path)

    return dict(n_corpus=len(rows), pairs=pairs, top150=top150, top_union_tau=top_union_tau, weights=weights)


def write_cross_pipeline_summary(out_path, stats_by_pipeline, models):
    """Nuovo file (non un foglio dentro uno esistente) in revisione/ con le metriche di
    TUTTE le pipeline passate affiancate — pensato per la presentazione: una tabella sola,
    non un foglio a testa da confrontare a mano. `stats_by_pipeline` = {nome_pipeline:
    risultato di read_pipeline_study_stats()}."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Confronto"

    hf = Font(bold=True, color="FFFFFF")
    hd = PatternFill("solid", fgColor="1F6F3E")
    note_fill = PatternFill("solid", fgColor="FFF2CC")

    row = 1
    ws.cell(row=row, column=1, value=(
        "Confronto kNN vs LLM (ranking sull'intero corpus) tra osservatori. "
        "CAUTELE da leggere prima di trarre conclusioni: "
        "(1) la soglia \"forte disaccordo\" è ≥40 posizioni — su un corpus di ~1400 articoli "
        "è solo il 2-3% della lista, quindi una percentuale alta di \"disaccordo\" è meno "
        "drammatica di quanto sembri a prima vista; "
        "(2) l'AUC di gpt-5.4 è quasi tautologica: misura se il punteggio 0-100 di gpt-5.4 "
        "predice la decisione Sì/No... di gpt-5.4 stesso (stesso modello, prompt diverso) — "
        "non è una prova indipendente, a differenza dell'AUC di un modello diverso; "
        "(3) l'AUC del kNN è calcolata SOLO sui candidati che il kNN stesso ha già scelto per "
        "il gate — un confronto non ad armi pari con l'LLM, che giudica tutto il corpus; "
        "(4) zero righe \"Parere ricercatore\" sono compilate a mano finora in nessuna "
        "pipeline: nessuna verifica umana indipendente ancora disponibile."
    )).font = Font(bold=True, size=11)
    ws.cell(row=row, column=1).alignment = openpyxl.styles.Alignment(wrap_text=True)
    ws.cell(row=row, column=1).fill = note_fill
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.row_dimensions[row].height = 140
    row += 2

    ws.cell(row=row, column=1, value="Confronto posizioni (kNN vs LLM, per coppia)").font = Font(bold=True, size=12)
    row += 1
    heads1 = ["Osservatorio", "Confronto", "n", "media |Δ|", "mediana |Δ|", "concordi (≤10)", "forte disaccordo (≥40)", "correlazione", "tau"]
    for c, h in enumerate(heads1, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    row += 1
    for pipeline_name, stats in stats_by_pipeline.items():
        for (a, b), s in stats["pairs"].items():
            if not s:
                continue
            ws.cell(row=row, column=1, value=pipeline_name)
            ws.cell(row=row, column=2, value=f"{a} vs {b}")
            ws.cell(row=row, column=3, value=s["n"])
            ws.cell(row=row, column=4, value=round(s["mean_abs_delta"], 1))
            ws.cell(row=row, column=5, value=round(s["median_abs_delta"], 1))
            ws.cell(row=row, column=6, value=f"{s['pct_agree']:.0f}%")
            ws.cell(row=row, column=7, value=f"{s['pct_disagree']:.0f}%")
            ws.cell(row=row, column=8, value=round(s["corr"], 2))
            ws.cell(row=row, column=9, value=round(s["tau"], 2))
            row += 1

    row += 1
    ws.cell(row=row, column=1, value=(
        "Confronto ristretto ai primi 150 per kNN — il pool che arriva DAVVERO al gate "
        "oggi (più rilevante della tabella sopra, che guarda tutto il corpus)"
    )).font = Font(bold=True, size=12)
    row += 1
    heads_t150 = ["Osservatorio", "kNN vs", "n", "sovrapposizione top-150", "media |Δ| locale", "mediana", "concordi (≤10)", "forte disaccordo (≥40)"]
    for c, h in enumerate(heads_t150, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    row += 1
    for pipeline_name, stats in stats_by_pipeline.items():
        for model, t in stats.get("top150", {}).items():
            ws.cell(row=row, column=1, value=pipeline_name)
            ws.cell(row=row, column=2, value=model)
            ws.cell(row=row, column=3, value=t["n"])
            ov_cell = ws.cell(row=row, column=4, value=f"{t['overlap_pct']:.0f}%")
            ov_cell.font = Font(bold=True, color="B8623E")
            ws.cell(row=row, column=5, value=round(t["mean_abs_delta"], 1))
            ws.cell(row=row, column=6, value=round(t["median_abs_delta"], 1))
            ws.cell(row=row, column=7, value=f"{t['pct_agree']:.0f}%")
            ws.cell(row=row, column=8, value=f"{t['pct_disagree']:.0f}%")
            row += 1

    row += 1
    ws.cell(row=row, column=1, value=(
        "Kendall's tau ristretto all'unione dei primi 300 per kNN o per LLM — complementare "
        "alla tabella sopra (corpus intero) e a quella dei top-150 (overlap/Δ): guarda solo "
        "dove la decisione conta, senza scegliere aprioristicamente un ranking come riferimento."
    )).font = Font(bold=True, size=12)
    row += 1
    heads_tau300 = ["Osservatorio", "Confronto", "n", "tau top-300 (unione)"]
    for c, h in enumerate(heads_tau300, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    row += 1
    for pipeline_name, stats in stats_by_pipeline.items():
        for (a, b), tu in stats.get("top_union_tau", {}).items():
            ws.cell(row=row, column=1, value=pipeline_name)
            ws.cell(row=row, column=2, value=f"{a} vs {b}")
            ws.cell(row=row, column=3, value=tu["n"])
            ws.cell(row=row, column=4, value=round(tu["tau"], 2))
            row += 1

    row += 1
    ws.cell(row=row, column=1, value="Peso ottimo kNN+LLM (AUC contro il Sì/No del gate)").font = Font(bold=True, size=12)
    row += 1
    heads2 = ["Osservatorio", "Modello LLM", "n (con giudizio gate)", "peso ottimo (w)", "AUC combinato", "AUC solo kNN", "AUC solo LLM"]
    for c, h in enumerate(heads2, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = hf
        cell.fill = hd
    row += 1
    for pipeline_name, stats in stats_by_pipeline.items():
        for model, w in stats["weights"].items():
            ws.cell(row=row, column=1, value=pipeline_name)
            ws.cell(row=row, column=2, value=model)
            ws.cell(row=row, column=3, value=w["n"])
            ws.cell(row=row, column=4, value=w["best_w"])
            ws.cell(row=row, column=5, value=w["best_auc"])
            ws.cell(row=row, column=6, value=w["auc_knn_only"])
            ws.cell(row=row, column=7, value=w["auc_llm_only"])
            row += 1

    widths = {"A": 16, "B": 26, "C": 22, "D": 16, "E": 16, "F": 16, "G": 16, "H": 14, "I": 10}
    for col, wd in widths.items():
        ws.column_dimensions[col].width = wd

    wb.save(out_path)
    return out_path


def run_full_corpus_study(name, models, dry_run=False, limit=None):
    """Orchestratore dello studio 'LLM sull'intero corpus vs kNN'. Punteggia OGNI riga di
    ranking.csv (non solo si_rows+borderline) con ciascun modello in `models`, in sequenza
    (stessi dict di riga riusati — il punteggio di un modello va copiato in una chiave
    dedicata prima di passare al modello successivo, altrimenti verrebbe sovrascritto).
    Scrive il risultato in revisione/ (xlsx_revisione), mai nei file di produzione."""
    if name not in FULL_CORPUS_STUDY_PIPELINES:
        log(f"  ATTENZIONE: pipeline '{name}' non ancora supportata da questo studio "
            f"(serve criteria_attr in PIPELINES) — salto.")
        return {"in": 0, "out": 0}

    cfg = PIPELINES[name]
    log(f"\n=== STUDIO CORPUS INTERO: {name} — modelli: {', '.join(models)} ===")
    rows = load_full_corpus(cfg)
    log(f"  articoli nel corpus: {len(rows)}")
    rows = compute_knn_ranks(rows)

    desc = get_description(cfg)
    criteria = get_criteria(cfg)

    if dry_run:
        r0 = rows[0]
        prompt = PROMPT.format(obs_name=cfg["obs_name"], desc=desc, criteria=criteria,
                                title=r0["titolo"], body=(r0.get("full_text") or "")[:6000])
        log(f"  [dry-run] esempio di prompt (primo articolo, '{r0['titolo'][:50]}'):\n{prompt[:900]}...")
        return None

    totals = {"in": 0, "out": 0}
    for model in models:
        log(f"  --- modello {model} ---")
        scored, tok = score_candidates(rows, desc, criteria, cfg["obs_name"], model, limit=limit)
        totals["in"] += tok["in"]
        totals["out"] += tok["out"]
        ok = sum(1 for r in scored if r.get("llm_score") is not None)
        log(f"    punteggiati: {ok}/{len(scored)}  |  token: {tok['in']} in + {tok['out']} out")
        # copia il punteggio in una chiave dedicata PRIMA del prossimo modello (altrimenti
        # score_candidates() del modello successivo sovrascrive "llm_score" sugli stessi dict)
        for r in scored:
            r[f"llm_score__{model}"] = r.get("llm_score")

    if limit:
        log(f"  [lotto pilota, {limit} righe] nessuna scrittura sul foglio.")
        return totals

    for model in models:
        for r in rows:
            r["llm_score"] = r.get(f"llm_score__{model}")
        rows = compute_llm_ranks(rows)
        for r in rows:
            r[f"rank_llm__{model}"] = r.get("rank_llm_local")

    n_written = write_full_corpus_comparison_sheet(cfg["xlsx_revisione"], name, rows, models)
    log(f"  foglio 'LLM_vs_kNN_corpus_intero' scritto in {cfg['xlsx_revisione'].name}: {n_written} righe")
    return totals


def run_add_model_study(name, existing_models, new_model, dry_run=False, limit=None):
    """Aggiunge UN modello nuovo a un foglio 'LLM_vs_kNN_corpus_intero' già scritto da
    run_full_corpus_study(), SENZA rispendere sui modelli già fatti (`existing_models`):
    rilegge i loro rank già a foglio (read_existing_rank_table, zero chiamate), punteggia
    sul corpus intero SOLO `new_model` (chiamate reali), abbina per link, riscrive il foglio
    con tutti i modelli insieme (write_full_corpus_comparison_sheet ora regge N modelli).
    Il peso ottimo/AUC per `existing_models` non è ricalcolabile (i punteggi 0-100 grezzi non
    sono ripersistiti, solo i rank) — resta quello già mostrato altrove per quei modelli;
    qui si vede correttamente solo il peso ottimo del modello nuovo."""
    if name not in FULL_CORPUS_STUDY_PIPELINES:
        log(f"  ATTENZIONE: pipeline '{name}' non ancora supportata da questo studio — salto.")
        return {"in": 0, "out": 0}

    cfg = PIPELINES[name]
    log(f"\n=== AGGIUNGO MODELLO: {name} — nuovo: {new_model}  (già presenti: {', '.join(existing_models)}) ===")

    existing_by_link = {r["link"]: r for r in read_existing_rank_table(cfg["xlsx_revisione"], existing_models) if r["link"]}
    log(f"  righe già a foglio (con i rank di {', '.join(existing_models)}): {len(existing_by_link)}")

    rows = load_full_corpus(cfg)
    log(f"  articoli nel corpus (freschi da ranking.csv): {len(rows)}")
    rows = compute_knn_ranks(rows)

    desc = get_description(cfg)
    criteria = get_criteria(cfg)

    if dry_run:
        r0 = rows[0]
        prompt = PROMPT.format(obs_name=cfg["obs_name"], desc=desc, criteria=criteria,
                                title=r0["titolo"], body=(r0.get("full_text") or "")[:6000])
        log(f"  [dry-run] esempio di prompt (primo articolo, '{r0['titolo'][:50]}'):\n{prompt[:900]}...")
        return None

    scored, tok = score_candidates(rows, desc, criteria, cfg["obs_name"], new_model, limit=limit)
    ok = sum(1 for r in scored if r.get("llm_score") is not None)
    log(f"  punteggiati ({new_model}): {ok}/{len(scored)}  |  token: {tok['in']} in + {tok['out']} out")
    # copia il punteggio grezzo in una chiave dedicata: sweep_weight() dentro
    # write_full_corpus_comparison_sheet() cerca "llm_score__<modello>", non "llm_score" —
    # senza questa riga il peso ottimo del modello nuovo risulterebbe "dati insufficienti"
    # anche se i punteggi ci sono (bug reale, trovato e corretto).
    for r in scored:
        r[f"llm_score__{new_model}"] = r.get("llm_score")

    if limit:
        log(f"  [lotto pilota, {limit} righe] nessuna scrittura sul foglio.")
        return tok

    rows = compute_llm_ranks(rows)
    for r in rows:
        r[f"rank_llm__{new_model}"] = r.get("rank_llm_local")
        old = existing_by_link.get(r["link"])
        for m in existing_models:
            r[f"rank_llm__{m}"] = old.get(f"rank_llm__{m}") if old else None

    matched = sum(1 for r in rows if r["link"] in existing_by_link)
    log(f"  abbinati per link ai rank già a foglio: {matched}/{len(rows)}")

    all_models = list(existing_models) + [new_model]
    n_written = write_full_corpus_comparison_sheet(cfg["xlsx_revisione"], name, rows, all_models)
    log(f"  foglio 'LLM_vs_kNN_corpus_intero' riscritto in {cfg['xlsx_revisione'].name} con {len(all_models)} modelli: {n_written} righe")
    return tok


def run_pipeline(name, model, dry_run=False, limit=None, cluster_sheet_only=False):
    cfg = PIPELINES[name]
    log(f"\n=== {name} ===")
    candidates = load_candidates(cfg)
    log(f"  candidati (stessa selezione del foglio Da_revisionare): {len(candidates)}")

    if dry_run:
        r0 = candidates[0]
        desc = get_description(cfg)
        criteria = get_criteria(cfg)
        prompt = PROMPT.format(obs_name=cfg["obs_name"], desc=desc, criteria=criteria, title=r0["titolo"],
                                body=(r0.get("full_text") or "")[:6000])
        log(f"  [dry-run] esempio di prompt (primo candidato, '{r0['titolo'][:50]}'):\n{prompt[:800]}...")
        return None

    rows = compute_knn_ranks(candidates)

    if cluster_sheet_only:
        # NESSUNA chiamata LLM: legge Rank LLM/Δ già scritti da un giro precedente.
        found = read_existing_llm_ranks(cfg["xlsx"], rows)
        log(f"  letti punteggi LLM già scritti nel foglio: {found}/{len(rows)}  |  costo: 0 (nessuna chiamata LLM)")
        if found == 0:
            log("  ATTENZIONE: nessun punteggio trovato — lancia prima il giro completo (senza --cluster-sheet-only)")
            return {"in": 0, "out": 0}
        n_clusters = write_cluster_sheet(cfg["xlsx"], rows, model)
        log(f"  foglio 'Analisi_Cluster' scritto in {cfg['xlsx'].name}: {n_clusters} cluster")

        matched_full = write_llm_columns_into_sheet(cfg["xlsx"], cfg["full_ranking_sheet"], rows, model,
                                                      all_candidates_expected=False)
        log(f"  colonne K/L scritte anche in '{cfg['full_ranking_sheet']}': {matched_full} righe con punteggio LLM su {len(rows)}")

        last = LLM_ONLY_COST_LAST_RUN[name]
        write_cost_sheet(cfg["xlsx"], name, model,
                          {"in": last["tok_in"], "out": last["tok_out"]}, last["n"])
        log(f"  foglio 'Costi' scritto in {cfg['xlsx'].name} (dai numeri dell'ultimo giro reale, nessuna nuova chiamata)")
        return {"in": 0, "out": 0}

    desc = get_description(cfg)
    criteria = get_criteria(cfg)
    rows, tok = score_candidates(rows, desc, criteria, cfg["obs_name"], model, limit=limit)
    rows = compute_llm_ranks(rows)
    ok = sum(1 for r in rows if r.get("llm_score") is not None)
    log(f"  punteggiati: {ok}/{len(rows)}  |  token: {tok['in']} in + {tok['out']} out")

    if limit:
        log(f"  [lotto pilota, {limit} righe] nessuna scrittura sul foglio Excel. Esempi:")
        for r in sorted(rows, key=lambda r: -(r.get("llm_score") or -1))[:5]:
            log(f"    score={r.get('llm_score')}  rank_knn={r['rank_knn_local']}  rank_llm={r.get('rank_llm_local')}  "
                f"'{r['titolo'][:55]}'  motivo: {r.get('llm_motivo','')[:100]}")
        return tok

    matched = write_llm_columns_into_sheet(cfg["xlsx"], "Da_revisionare", rows, model)
    log(f"  scritte {matched} righe (colonne K/L) in Da_revisionare ({cfg['xlsx'].name})")

    matched_full = write_llm_columns_into_sheet(cfg["xlsx"], cfg["full_ranking_sheet"], rows, model,
                                                  all_candidates_expected=False)
    log(f"  colonne K/L scritte anche in '{cfg['full_ranking_sheet']}': {matched_full} righe con punteggio LLM su {len(rows)}")

    n_clusters = write_cluster_sheet(cfg["xlsx"], rows, model)
    log(f"  foglio 'Analisi_Cluster' scritto in {cfg['xlsx'].name}: {n_clusters} cluster")

    write_cost_sheet(cfg["xlsx"], name, model, tok, len(rows))
    log(f"  foglio 'Costi' scritto in {cfg['xlsx'].name}")

    biggest = sorted(rows, key=lambda r: -abs(r["delta_rank"] or 0))[:5]
    log("  maggiori disaccordi (|Δ posizione| più alto):")
    for r in biggest:
        log(f"    Δ={r['delta_rank']:+d}  rank_knn={r['rank_knn_local']:>3}  rank_llm={r['rank_llm_local']:>3}  "
            f"'{r['titolo'][:50]}'  motivo: {r.get('llm_motivo','')[:90]}")
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(PIPELINES) + ["all"], default=None)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="lotto pilota: valuta solo le prime N righe, non scrive sul foglio")
    ap.add_argument("--cluster-sheet-only", action="store_true",
                     help="nessuna chiamata LLM: rilegge Rank LLM/Δ già scritti e ricostruisce solo il foglio Analisi_Cluster")
    ap.add_argument("--full-corpus-study", action="store_true",
                     help="ranking LLM sull'INTERO corpus (non solo i candidati gated) vs kNN, scritto in revisione/ — "
                          "solo pipeline in FULL_CORPUS_STUDY_PIPELINES (oggi: ocx, startup, ai)")
    ap.add_argument("--models", default="gpt-5.4,gpt-5.6-sol",
                     help="modelli da confrontare in --full-corpus-study/--cross-summary; "
                          "in --add-model, i modelli GIÀ presenti nel foglio (separati da virgola)")
    ap.add_argument("--add-model", default=None,
                     help="aggiunge UN modello nuovo a un foglio 'LLM_vs_kNN_corpus_intero' già scritto, "
                          "SENZA rispendere sui modelli già presenti (elencati in --models)")
    ap.add_argument("--cross-summary", action="store_true",
                     help="NESSUNA chiamata LLM: rilegge i fogli 'LLM_vs_kNN_corpus_intero' già scritti da "
                          "--full-corpus-study per le pipeline in --pipelines e li affianca in un file nuovo dentro revisione/")
    ap.add_argument("--refresh-full-corpus-sheet", action="store_true",
                     help="NESSUNA chiamata LLM: rigenera il foglio 'LLM_vs_kNN_corpus_intero' già scritto da "
                          "--full-corpus-study con l'ultima versione dei blocchi riassuntivi (es. dopo aver "
                          "aggiunto una metrica nuova allo script) — rilegge rank e peso ottimo già a foglio, "
                          "non li ricalcola da zero. Richiede --pipeline (o 'all') e --models coi modelli GIÀ nel foglio.")
    ap.add_argument("--pipelines", default="ocx,startup,ai",
                     help="pipeline da includere in --cross-summary, separate da virgola")
    ap.add_argument("--out", default=None, help="percorso di output per --cross-summary")
    args = ap.parse_args()

    if args.cross_summary:
        pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        out_path = Path(args.out) if args.out else ROOT / "revisione" / "Confronto_kNN_vs_LLM.xlsx"
        log(f"Confronto cross-pipeline (zero chiamate LLM): {', '.join(pipelines)}  |  modelli: {', '.join(models)}")
        stats_by_pipeline = {}
        for name in pipelines:
            cfg = PIPELINES[name]
            stats_by_pipeline[name] = read_pipeline_study_stats(cfg["xlsx_revisione"], models)
            log(f"  {name}: letto da {cfg['xlsx_revisione'].name} ({stats_by_pipeline[name]['n_corpus']} articoli)")
        write_cross_pipeline_summary(out_path, stats_by_pipeline, models)
        log(f"\n  salvato: {out_path}")
        log("\nFATTO.")
        return

    if not args.pipeline:
        ap.error("--pipeline è richiesto (tranne con --cross-summary)")
    names = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]

    if args.refresh_full_corpus_sheet:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        log(f"Rigenero 'LLM_vs_kNN_corpus_intero' (zero chiamate LLM) — modelli: {', '.join(models)}")
        for name in names:
            if name not in FULL_CORPUS_STUDY_PIPELINES:
                log(f"  ATTENZIONE: pipeline '{name}' non supportata da questo studio — salto.")
                continue
            refresh_full_corpus_sheet(name, models)
        log("\nFATTO.")
        return

    if args.add_model:
        existing_models = [m.strip() for m in args.models.split(",") if m.strip()]
        log(f"Aggiungo modello '{args.add_model}' (già presenti: {', '.join(existing_models)})")
        totals = {"in": 0, "out": 0}
        for name in names:
            tok = run_add_model_study(name, existing_models, args.add_model, dry_run=args.dry_run, limit=args.limit)
            if tok:
                totals["in"] += tok["in"]
                totals["out"] += tok["out"]
        if not args.dry_run:
            log(f"\n=== COSTO TOTALE ({args.add_model}) ===")
            log(f"  token: {totals['in']} in + {totals['out']} out")
        log("\nFATTO.")
        return

    if args.full_corpus_study:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        log(f"Studio ranking LLM su corpus intero vs kNN — modelli: {', '.join(models)}")
        totals = {"in": 0, "out": 0}
        for name in names:
            tok = run_full_corpus_study(name, models, dry_run=args.dry_run, limit=args.limit)
            if tok:
                totals["in"] += tok["in"]
                totals["out"] += tok["out"]
        if not args.dry_run:
            log(f"\n=== COSTO TOTALE ({', '.join(models)}) ===")
            log(f"  token: {totals['in']} in + {totals['out']} out")
        log("\nFATTO.")
        return

    log(f"Esperimento ranking solo-LLM — modello: {args.model}"
        + ("  [solo ricostruzione Analisi_Cluster, zero chiamate]" if args.cluster_sheet_only else ""))

    totals = {"in": 0, "out": 0}
    for name in names:
        tok = run_pipeline(name, args.model, dry_run=args.dry_run, limit=args.limit,
                            cluster_sheet_only=args.cluster_sheet_only)
        if tok:
            totals["in"] += tok["in"]
            totals["out"] += tok["out"]

    if not args.dry_run:
        log(f"\n=== COSTO TOTALE ({args.model}) ===")
        if args.cluster_sheet_only:
            log("  0 token — nessuna chiamata LLM, solo dati già pagati in un giro precedente")
        else:
            log(f"  token: {totals['in']} in + {totals['out']} out  (nessun listino prezzi nel repo: riporto solo i token)")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
