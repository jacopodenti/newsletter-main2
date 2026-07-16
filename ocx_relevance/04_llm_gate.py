"""
STEP 4 — Gate LLM di pertinenza (precisione).

Legge un file di articoli (jsonl con title, full_text/summary), chiede a GPT si/no
di pertinenza OCX. NON usa le etichette: giudica il contenuto + descrizione osservatorio.

Uso:
  python ocx_relevance/04_llm_gate.py data/dataset.jsonl        # valuta il dataset (per verifica)
  (di norma viene chiamato dallo step 5 sui soli candidati)

Output: aggiunge campo 'llm' (1/0) e salva <input>.llm.jsonl. Logga costo e, se ci sono
le label, la precisione.
"""
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, llm_client, ocx_description, LLM_MODEL, DATA

PROMPT = """Sei un classificatore di pertinenza per l'Osservatorio "Omnichannel Customer Experience" (OCX) del Politecnico di Milano.

DESCRIZIONE OSSERVATORIO:
{desc}

COMPITO: dato un articolo, decidi se e PERTINENTE a questo osservatorio.
E pertinente SOLO se tratta in modo specifico di customer experience omnicanale, customer journey,
gestione della relazione col cliente su piu canali, contact center, CRM, self-service, loyalty, NPS.
NON e pertinente se parla genericamente di digitale, AI, cybersecurity, cloud, innovazione, startup,
fintech, pagamenti, telco, hardware senza un focus concreto sull'esperienza del cliente.

Rispondi SOLO con JSON: {{"rilevante": true/false, "confidence": 0.0-1.0, "motivo": "una frase"}}

ARTICOLO:
Titolo: {title}
Testo: {body}

JSON:"""


def run_gate(rows, max_workers=8):
    client = llm_client()
    desc = ocx_description()
    lock = threading.Lock()
    tok = {"in": 0, "out": 0}

    def one(r):
        body = (r.get("full_text") or r.get("summary") or "")[:6000]
        prompt = PROMPT.format(desc=desc, title=r["title"], body=body)
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
                return 1 if res.get("rilevante") else 0
            except Exception:
                if attempt == 4:
                    return None
                time.sleep(2 ** attempt)  # attesa crescente su 429/errore
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        preds = list(ex.map(one, rows))
    for r, p in zip(rows, preds):
        r["llm"] = p
    return rows, tok


def main():
    path = DATA / "dataset.jsonl"
    if len(sys.argv) > 1:
        path = __import__("pathlib").Path(sys.argv[1])
    log(f"STEP 4 — gate LLM ({LLM_MODEL}) su {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    log(f"  articoli: {len(rows)}")

    rows, tok = run_gate(rows)
    si = sum(1 for r in rows if r.get("llm") == 1)
    no = sum(1 for r in rows if r.get("llm") == 0)
    err = sum(1 for r in rows if r.get("llm") is None)
    log(f"\n  LLM: Si={si}  No={no}  errori={err}")
    log(f"  token: {tok['in']} in + {tok['out']} out")

    if all("label" in r for r in rows):
        tp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 1)
        fp = sum(1 for r in rows if r["llm"] == 1 and r["label"] == 0)
        fn = sum(1 for r in rows if r["llm"] == 0 and r["label"] == 1)
        log(f"\n[VERIFICA vs etichette]  TP={tp} FP={fp} FN={fn}")
        log(f"  precisione={tp/(tp+fp) if tp+fp else 0:.0%}  recall={tp/(tp+fn) if tp+fn else 0:.0%}")
        log(f"  -> falsi positivi dell'LLM: {fp}")

    out = path.with_suffix(".llm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\n  salvato: {out}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
