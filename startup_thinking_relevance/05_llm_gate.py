"""
STEP 5 — Gate LLM di pertinenza per l'Osservatorio Startup Thinking.

Stesso schema di ocx_relevance/05_llm_gate.py: legge articoli (title, full_text/summary),
chiede a GPT si/no di pertinenza. NON usa le etichette del dataset, giudica solo
contenuto + descrizione dell'osservatorio + criteri RIENTRA/NON RIENTRA (nel blocco
COMPITO del prompt, stessa struttura di ai_relevance/ocx_relevance) — fonte originale
"Descrizione Oss. Startup Thinking[44].rtf".

Uso:
  python startup_thinking_relevance/05_llm_gate.py data/dataset.jsonl   # valuta il dataset (verifica)
  (di norma richiamato da 06_rank.py sui soli candidati migliori)
"""
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import log, llm_client, LLM_MODEL, log_gate_cost

DATA = HERE / "data"  # ATTENZIONE: common.py punta a ocx_relevance/data di default, qui serve startup_thinking_relevance/data

STARTUP_DESCRIPTION = """L'Osservatorio Startup Thinking si occupa di diffondere nelle imprese l'Open Innovation e la cultura imprenditoriale attraverso la contaminazione con l'ecosistema startup e con la community dell'innovazione. La ricerca analizza come le imprese consolidate possano generare nuove fonti di valore attraverso un Innovation Management strutturato, aperto e imprenditoriale.

Due direttrici: (1) Spinta all'Open Innovation — coinvolgimento di startup, scaleup, università, incubatori, acceleratori, investitori, partner industriali attraverso scouting, partnership, sperimentazioni, investimenti; (2) Spinta alla trasformazione interna — capacità dell'impresa di organizzare, governare ed eseguire l'innovazione (strategia, governance, cultura, competenze, processi, portafoglio iniziative, metriche)."""

# Criteri di pertinenza — costante a sé (non solo testo inline nel PROMPT) così altri
# script (es. llm_only_rank.py) possono riusarli senza duplicarli/rischiare disallineamento.
STARTUP_CRITERIA = """RIENTRA NELLO SCOPE quando il tema centrale è: una corporate che avvia/modifica una strategia o funzione di innovazione; collaborazione corporate-startup/ecosistema; programmi di scouting, Open Innovation, venture clienting, accelerazione, incubazione; PoC/pilot/progetti sperimentali; criteri o metriche per selezionare/scalare iniziative; cambiamenti in ruoli, governance, cultura dell'innovazione; Corporate Venture Capital, corporate venturing, venture building, spin-off, acquisizioni di startup con finalità strategica; ecosistema startup, investimenti, crescita scaleup, trasferimento tecnologico, policy di sistema; una tecnologia emergente analizzata tramite casi d'uso/startup/adozione aziendale/impatti organizzativi.
NON RIENTRA (o rilevanza molto bassa): "innovativo" usato solo come aggettivo promozionale; lanci prodotto/recensioni senza informazioni sul processo di innovazione o adozione aziendale; annunci di modelli AI/benchmark/funzionalità senza casi d'uso aziendali o implicazioni per l'Innovation Management; articoli tecnici specialistici senza collegamento a startup/imprese/sperimentazione; cronaca finanziaria ordinaria (titoli, trimestrali, dividendi) senza componente di innovazione; M&A motivate solo da consolidamento/quota di mercato; notizie di marketing/CX/sostenibilità/cybersecurity/HR dove l'innovazione organizzativa non è centrale; nomine manageriali non legate all'innovazione; eventi/premi/classifiche senza dati o iniziative concrete; singoli round di investimento di rilevanza limitata senza informazioni su trend/ecosistema/scalabilità; contenuti motivazionali/teorici sull'imprenditorialità senza casi concreti."""

PROMPT = """Sei un classificatore di pertinenza per l'Osservatorio "Startup Thinking" del Politecnico di Milano.

DESCRIZIONE OSSERVATORIO:
{desc}

COMPITO: dato un articolo, decidi se e PERTINENTE a questo osservatorio.
{criteria}

Rispondi SOLO con JSON: {{"rilevante": true/false, "voto": 0-100 (quanto lo sceglieresti davvero per la prossima newsletter, non quanto parla del nostro tema), "motivo": "una frase"}}

ARTICOLO:
Titolo: {title}
Testo: {body}
{cluster_note}
JSON:"""

# Se il ranker segnala che questo articolo ha embedding molto simile ad altri candidati
# dello stesso run (probabile stessa notizia riportata da fonti diverse), lo si dice
# esplicitamente all'LLM e gli si chiede di commentarlo nel motivo — il revisore umano
# vede così, direttamente nel foglio Excel, quali righe sono probabili duplicati.
CLUSTER_NOTE_TMPL = """
ATTENZIONE — possibile notizia duplicata: tra gli altri candidati di questo stesso run,
i seguenti sembrano riguardare lo stesso evento/notizia (embedding molto simile) — sotto,
titolo e un breve estratto di ciascuno, per verificare se è davvero la stessa notizia:
{peers}
Se confermi che è la stessa notizia, dillo nel motivo (es. "stessa notizia di: <titolo>"),
così il revisore può scegliere quale fonte tenere. Se invece tratta un aspetto/evento
diverso pur sullo stesso tema, ignora questo avviso."""


# ATTENZIONE — variabilità tra run: misurato il 20/08 rilanciando lo STESSO identico prompt
# 3 volte sui 150 articoli di Startup Thinking già etichettati da Filippo (nessun controllo di
# temperature/seed su gpt-5.4): accordo 82,0% / 78,0% / 81,3%, kappa 0,573 / 0,495 / 0,568 —
# un'ampiezza di 4 punti/0,08 di kappa tra minimo e massimo, con 2 run su 3 vicine al valore
# alto e una sola più bassa (troppo pochi run per stimare la distribuzione con certezza).
# Conseguenza pratica: una differenza sotto ~5 punti di accordo (o kappa equivalente) tra due
# varianti di prompt NON è interpretabile come un effetto reale senza ripetere la misura più
# volte — potrebbe essere solo rumore tra sessioni, non l'effetto della modifica.
#
# ATTENZIONE — cambio confidence -> voto (25/08): il campo "confidence" è stato sostituito da
# "voto" 0-100 ("quanto lo sceglieresti davvero per la prossima newsletter"), perché la
# confidence — anche presa con il segno giusto in base al verdetto — non discriminava tra i
# "Sì" buoni e i "Sì" sbagliati (media 0,852 vs 0,830, sostanzialmente indistinguibile).
# Test di non regressione sugli stessi 143 giudizi netti di Filippo (le 6 righe ambigue tipo
# "notizia citata più volte"/"Interno" escluse, non contate come No), resto del prompt
# identico byte per byte: accordo 86,0% -> 85,3% (-0,7pt), kappa 0,665 -> 0,620 (-0,044),
# entrambi dentro la banda di rumore misurata sopra. Ma precisione e recall si muovono più di
# quanto l'accordo aggregato suggerisca: precisione 68,1% -> 71,1%, recall 86,5% -> 73,0%
# (-13,5pt). 13/143 articoli cambiano verdetto in entrambe le direzioni (6 Sì persi, 1 Sì
# recuperato, 5 falsi positivi corretti, 1 nuovo falso positivo) — non c'è una banda di rumore
# storica per precisione/recall (lo studio dei 3 rerun sopra ha misurato solo accordo e kappa),
# quindi questo spostamento non è classificabile come rumore o effetto reale con la stessa
# sicurezza.
# Verificato che il voto dei 6 Sì persi (media 24,2) è indistinguibile da quello dei 5 falsi
# positivi corretti (media 22,0) e lontano dal voto dei Sì confermati (media 70,2): il modello
# non riconosce i 6 persi come buoni e risponde No lo stesso — li valuta coerentemente deboli
# su entrambi i campi, non c'è un'incoerenza interna al prompt da sfruttare per recuperarli
# tenendo il sì/no invariato. Il voto separa invece bene DENTRO il gruppo dei Sì (confermati
# media 70,2 vs smentiti media 65,2) e ordina i candidati meglio del rank del ranker (AUC
# 0,897 vs 0,805, precision@10 100% vs 70% su 143/143 articoli) — è il segnale su cui si basa
# l'ordinamento dei candidati a valle del gate, non il criterio di sì/no.
def run_gate(rows, max_workers=8):
    client = llm_client()
    lock = threading.Lock()
    tok = {"in": 0, "out": 0}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cost_log_path = DATA / "gate_cost_log.jsonl"

    def one(r):
        body = (r.get("full_text") or r.get("summary") or "")[:6000]
        peers = r.get("cluster_peers") or []
        peers_txt = "\n".join(f'- {p["title"]}: "{p["excerpt"]}..."' for p in peers)
        cluster_note = CLUSTER_NOTE_TMPL.format(peers=peers_txt) if peers else ""
        prompt = PROMPT.format(desc=STARTUP_DESCRIPTION, criteria=STARTUP_CRITERIA, title=r["title"], body=body, cluster_note=cluster_note)
        for attempt in range(5):
            try:
                resp = client.responses.create(model=LLM_MODEL, input=prompt)
                txt = resp.output_text.strip()
                if txt.startswith("```"):
                    txt = txt.split("```")[1].removeprefix("json").strip()
                res = json.loads(txt)
                with lock:
                    tok["in"] += resp.usage.input_tokens
                    tok["out"] += resp.usage.output_tokens
                log_gate_cost(cost_log_path, run_id, r["title"], resp.usage.input_tokens, resp.usage.output_tokens)
                return {
                    "llm": 1 if res.get("rilevante") else 0,
                    "llm_voto": res.get("voto"),
                    "llm_motivo": res.get("motivo", ""),
                }
            except Exception:
                if attempt == 4:
                    return {"llm": None, "llm_voto": None, "llm_motivo": ""}
                time.sleep(2 ** attempt)
        return {"llm": None, "llm_voto": None, "llm_motivo": ""}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        preds = list(ex.map(one, rows))
    for r, p in zip(rows, preds):
        r["llm"] = p["llm"]
        r["llm_voto"] = p["llm_voto"]
        r["llm_motivo"] = p["llm_motivo"]
    return rows, tok


def main():
    # 1) carica gli articoli da giudicare — default: dataset.jsonl, oppure il file
    # passato da linea di comando
    path = DATA / "dataset.jsonl"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    log(f"STEP 5 — gate LLM ({LLM_MODEL}) su {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    log(f"  articoli: {len(rows)}")

    # 2) gate LLM su ogni articolo
    rows, tok = run_gate(rows)
    si = sum(1 for r in rows if r.get("llm") == 1)
    no = sum(1 for r in rows if r.get("llm") == 0)
    err = sum(1 for r in rows if r.get("llm") is None)
    log(f"\n  LLM: Si={si}  No={no}  errori={err}")
    log(f"  token: {tok['in']} in + {tok['out']} out")

    # 3) se il file ha etichette (es. dataset.jsonl passato come verifica), confronta il
    # giudizio LLM con esse — in produzione (candidati dal corpus) non ci sono label
    if all("label" in r for r in rows):
        tp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 1)
        fp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 0)
        fn = sum(1 for r in rows if r["llm"] == 0 and r["label"] == 1)
        log(f"\n[VERIFICA vs etichette]  TP={tp} FP={fp} FN={fn}")
        log(f"  precisione={tp/(tp+fp) if tp+fp else 0:.0%}  recall={tp/(tp+fn) if tp+fn else 0:.0%}")

    # 4) output — <input>.llm.jsonl, con il campo 'llm' aggiunto a ogni riga
    out = path.with_suffix(".llm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\n  salvato: {out}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
