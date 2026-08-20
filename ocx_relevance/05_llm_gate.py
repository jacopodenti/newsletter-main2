"""
STEP 5 — Gate LLM di pertinenza (precisione).

Legge un file di articoli (jsonl con title, full_text/summary), chiede a GPT si/no
di pertinenza OCX. NON usa le etichette: giudica il contenuto + descrizione osservatorio.

Uso:
  python ocx_relevance/05_llm_gate.py data/dataset.jsonl        # valuta il dataset (per verifica)
  (di norma viene chiamato dallo step 6 sui soli candidati)

Output: aggiunge campo 'llm' (1/0) e salva <input>.llm.jsonl. Logga costo e, se ci sono
le label, la precisione.
"""
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, llm_client, ocx_description, LLM_MODEL, DATA, log_gate_cost

# Criteri di pertinenza — costante a sé (non solo testo inline nel PROMPT) così altri
# script (es. llm_only_rank.py) possono riusarli senza duplicarli/rischiare disallineamento.
OCX_CRITERIA = """E pertinente SOLO se tratta in modo specifico di customer experience omnicanale, customer journey,
gestione della relazione col cliente su piu canali, contact center, CRM, self-service, loyalty, NPS.
NON e pertinente se parla genericamente di digitale, AI, cybersecurity, cloud, innovazione, startup,
fintech, pagamenti, telco, hardware senza un focus concreto sull'esperienza del cliente."""

PROMPT = """Sei un classificatore di pertinenza per l'Osservatorio "Omnichannel Customer Experience" (OCX) del Politecnico di Milano.

DESCRIZIONE OSSERVATORIO:
{desc}

COMPITO: dato un articolo, decidi se e PERTINENTE a questo osservatorio.
{criteria}

Rispondi SOLO con JSON: {{"rilevante": true/false, "confidence": 0.0-1.0, "motivo": "una frase"}}

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


# ATTENZIONE — variabilità tra run: misurato il 20/08 su startup_thinking_relevance (stesso
# meccanismo di gate, stesso modello) rilanciando lo STESSO identico prompt 3 volte sui 150
# articoli già etichettati da un ricercatore (nessun controllo di temperature/seed su gpt-5.4):
# accordo 82,0% / 78,0% / 81,3%, kappa 0,573 / 0,495 / 0,568 — un'ampiezza di 4 punti/0,08 di
# kappa tra minimo e massimo, con 2 run su 3 vicine al valore alto e una sola più bassa (troppo
# pochi run per stimare la distribuzione con certezza).
# Conseguenza pratica: una differenza sotto ~5 punti di accordo (o kappa equivalente) tra due
# varianti di prompt NON è interpretabile come un effetto reale senza ripetere la misura più
# volte — potrebbe essere solo rumore tra sessioni, non l'effetto della modifica.
def run_gate(rows, max_workers=8):
    client = llm_client()
    desc = ocx_description()
    lock = threading.Lock()
    tok = {"in": 0, "out": 0}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cost_log_path = DATA / "gate_cost_log.jsonl"

    # classifica un singolo articolo: prompt con descrizione osservatorio + testo,
    # retry con backoff se l'API fallisce (rate limit, errori transitori)
    def one(r):
        body = (r.get("full_text") or r.get("summary") or "")[:6000]
        peers = r.get("cluster_peers") or []
        peers_txt = "\n".join(f'- {p["title"]}: "{p["excerpt"]}..."' for p in peers)
        cluster_note = CLUSTER_NOTE_TMPL.format(peers=peers_txt) if peers else ""
        prompt = PROMPT.format(desc=desc, criteria=OCX_CRITERIA, title=r["title"], body=body, cluster_note=cluster_note)
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
                    "llm_confidence": res.get("confidence"),
                    "llm_motivo": res.get("motivo", ""),
                }
            except Exception:
                if attempt == 4:
                    return {"llm": None, "llm_confidence": None, "llm_motivo": ""}
                time.sleep(2 ** attempt)  # attesa crescente su 429/errore
        return {"llm": None, "llm_confidence": None, "llm_motivo": ""}

    # lancia le chiamate in parallelo, poi riporta esito/confidenza/motivo su ogni riga originale
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        preds = list(ex.map(one, rows))
    for r, p in zip(rows, preds):
        r["llm"] = p["llm"]
        r["llm_confidence"] = p["llm_confidence"]
        r["llm_motivo"] = p["llm_motivo"]
    return rows, tok


def main():
    # 1) file di input: default dataset.jsonl, oppure passato da riga di comando
    # (di norma è lo step 6 a chiamare run_gate() direttamente sui soli candidati)
    path = DATA / "dataset.jsonl"
    if len(sys.argv) > 1:
        path = __import__("pathlib").Path(sys.argv[1])
    log(f"STEP 5 — gate LLM ({LLM_MODEL}) su {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    log(f"  articoli: {len(rows)}")

    # 2) gate LLM vero e proprio
    rows, tok = run_gate(rows)
    si = sum(1 for r in rows if r.get("llm") == 1)
    no = sum(1 for r in rows if r.get("llm") == 0)
    err = sum(1 for r in rows if r.get("llm") is None)
    log(f"\n  LLM: Si={si}  No={no}  errori={err}")
    log(f"  token: {tok['in']} in + {tok['out']} out")

    # 3) se le righe hanno un'etichetta (solo su dataset.jsonl, mai sui candidati dello
    # step 6): confronta LLM vs etichetta per stimare precisione/recall del gate stesso
    if all("label" in r for r in rows):
        tp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 1)
        fp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 0)
        fn = sum(1 for r in rows if r["llm"] == 0 and r["label"] == 1)
        log(f"\n[VERIFICA vs etichette]  TP={tp} FP={fp} FN={fn}")
        log(f"  precisione={tp/(tp+fp) if tp+fp else 0:.0%}  recall={tp/(tp+fn) if tp+fn else 0:.0%}")
        log(f"  -> falsi positivi dell'LLM: {fp}")

    # 4) salva output
    out = path.with_suffix(".llm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\n  salvato: {out}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
