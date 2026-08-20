"""
STEP 5 — Gate LLM di pertinenza per l'Osservatorio Artificial Intelligence.

Stesso schema di ocx_relevance/05_llm_gate.py: legge articoli (title, full_text/summary),
chiede a GPT si/no di pertinenza AI. NON usa le etichette del dataset, giudica solo
contenuto + descrizione osservatorio.

Uso:
  python ai_relevance/05_llm_gate.py data/dataset.jsonl   # valuta il dataset (verifica)
  (di norma richiamato da 06_rank.py sui soli candidati migliori)
"""
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import log, llm_client, LLM_MODEL

DATA = HERE / "data"  # ATTENZIONE: common.py punta a ocx_relevance/data di default, qui serve ai_relevance/data

AI_DESCRIPTION = """L'Osservatorio Artificial Intelligence nasce nel 2017 per rispondere al crescente interesse di aziende pubbliche e private verso le potenzialità offerte dalle nuove tecnologie abilitanti all'Intelligenza Artificiale (AI). L'obiettivo è quello di creare una Community di confronto per indagare le reali opportunità dell'Intelligenza Artificiale (IA), in un contesto caratterizzato da poca chiarezza sullo stato dell'arte delle sue applicazioni e delle adozioni da parte delle imprese, sui benefici abilitati e sull'evoluzione attesa del mercato, combinando la prospettiva manageriale con quella tecnologica.
Si occupa di: spesa delle imprese italiane in AI; maturità delle imprese verso l'AI; progetti di AI sviluppati dalle imprese italiane; trend tecnologici (Machine Learning, Deep Learning, RPA intelligente, NLP, Generative AI); tecniche di costruzione di soluzioni AI; punto di vista del consumatore (AI Journey); implicazioni etiche e Trustworthy AI (incluso l'AI Act); impatto dell'AI sul mondo del lavoro."""

# Criteri di pertinenza — costante a sé (non solo testo inline nel PROMPT) così altri
# script (es. llm_only_rank.py) possono riusarli senza duplicarli/rischiare disallineamento.
AI_CRITERIA = """E pertinente SOLO se tratta in modo specifico e sostanziale di intelligenza artificiale
(tecnologie, applicazioni aziendali, mercato, adozione, etica/regolamentazione AI, impatto sul lavoro).
NON e pertinente se cita l'AI solo di passaggio, o parla genericamente di digitale/tech/cloud/
cybersecurity/fintech/spazio/salute senza un focus concreto sull'intelligenza artificiale."""

PROMPT = """Sei un classificatore di pertinenza per l'Osservatorio "Artificial Intelligence" del Politecnico di Milano.

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
    lock = threading.Lock()
    tok = {"in": 0, "out": 0}

    def one(r):
        body = (r.get("full_text") or r.get("summary") or "")[:6000]
        peers = r.get("cluster_peers") or []
        peers_txt = "\n".join(f'- {p["title"]}: "{p["excerpt"]}..."' for p in peers)
        cluster_note = CLUSTER_NOTE_TMPL.format(peers=peers_txt) if peers else ""
        prompt = PROMPT.format(desc=AI_DESCRIPTION, criteria=AI_CRITERIA, title=r["title"], body=body, cluster_note=cluster_note)
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
                return {
                    "llm": 1 if res.get("rilevante") else 0,
                    "llm_confidence": res.get("confidence"),
                    "llm_motivo": res.get("motivo", ""),
                }
            except Exception:
                if attempt == 4:
                    return {"llm": None, "llm_confidence": None, "llm_motivo": ""}
                time.sleep(2 ** attempt)
        return {"llm": None, "llm_confidence": None, "llm_motivo": ""}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        preds = list(ex.map(one, rows))
    for r, p in zip(rows, preds):
        r["llm"] = p["llm"]
        r["llm_confidence"] = p["llm_confidence"]
        r["llm_motivo"] = p["llm_motivo"]
    return rows, tok


def main():
    path = DATA / "dataset.jsonl"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    log(f"STEP 5 — gate LLM ({LLM_MODEL}) su {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    log(f"  articoli: {len(rows)}")

    # 1) gate LLM su ogni riga
    rows, tok = run_gate(rows)
    si = sum(1 for r in rows if r.get("llm") == 1)
    no = sum(1 for r in rows if r.get("llm") == 0)
    err = sum(1 for r in rows if r.get("llm") is None)
    log(f"\n  LLM: Si={si}  No={no}  errori={err}")
    log(f"  token: {tok['in']} in + {tok['out']} out")

    # 2) verifica vs etichette — solo se il file le ha (es. dataset.jsonl), per controllare il gate stesso
    if all("label" in r for r in rows):
        tp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 1)
        fp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 0)
        fn = sum(1 for r in rows if r["llm"] == 0 and r["label"] == 1)
        log(f"\n[VERIFICA vs etichette]  TP={tp} FP={fp} FN={fn}")
        log(f"  precisione={tp/(tp+fp) if tp+fp else 0:.0%}  recall={tp/(tp+fn) if tp+fn else 0:.0%}")

    # 3) output — jsonl con il giudizio LLM aggiunto
    out = path.with_suffix(".llm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\n  salvato: {out}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
