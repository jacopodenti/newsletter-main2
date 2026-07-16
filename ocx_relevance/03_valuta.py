"""
STEP 3 — Valuta il ranker SENZA addestramento e salva il set di riferimento.

Nessun modello da addestrare: il ranker confronta la news con gli ESEMPI OCX.
Qui misuriamo quant'è bravo, in modo trasparente (k-fold, ogni numero spiegato),
e salviamo gli esempi (vettori + label) che serviranno da riferimento.

Output: models/reference_set.npz  (X = vettori esempi, y = label)
Uso:    python ocx_relevance/03_valuta.py
"""
import json
import sys

import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, MODELS, DATA
from ranker import score_similarity
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score


def main():
    log("STEP 3 — valutazione del ranker (NIENTE addestramento: somiglianza agli esempi)")
    d = np.load(MODELS / "dataset_vectors.npz")
    X, y = d["X"], d["y"]
    rows = [json.loads(l) for l in open(DATA / "dataset.jsonl", encoding="utf-8")]

    log("\n[COME FUNZIONA IL RANKER]")
    log("  - Nessun modello addestrato. Ogni news prende punteggio dalla SOMIGLIANZA agli esempi:")
    log("    media(coseno con i 3 'Si' più vicini) − media(coseno con i 3 'No' più vicini).")
    log("  - Gli esempi OCX SONO il riferimento. Aggiungere un esempio = aggiungere un vettore.")

    log("\n[COME LO MISURIAMO — leggi]")
    log("  - 5 gruppi (fold), stessa proporzione Si/No. Riferimento = 4 gruppi, misuro sul 5° (mai visto).")
    log("  - Ripeto 20 volte. Ogni news è giudicata solo da esempi che NON la contengono (niente leakage).")
    log("  - AUC = probabilità che un rilevante prenda punteggio più alto di un non-rilevante (0.5=caso, 1=perfetto).")

    from collections import Counter
    log(f"\n[DATI]  totale {len(y)}  |  positivi {int(y.sum())}  |  negativi {int((1-y).sum())}")
    for o, n in Counter(r["origin"] for r in rows).items():
        log(f"    origine '{o}': {n}")

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    splits = list(rkf.split(X, y))
    aucs = []
    for rep in range(20):
        oof = np.zeros(len(y))
        for f in range(5):
            tr, te = splits[rep * 5 + f]
            oof[te] = score_similarity(X[te], X[tr], y[tr], k=3)
        aucs.append(roc_auc_score(y, oof))
    aucs = np.array(aucs)
    log(f"\n[RISULTATO — ranker per somiglianza]")
    log(f"  AUC media (20 ripetizioni): {aucs.mean():.3f}  (dev.std {aucs.std():.3f})")

    # precision@k su un ordinamento out-of-fold onesto
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = splits[f]
        oof[te] = score_similarity(X[te], X[tr], y[tr], k=3)
    order = np.argsort(-oof)
    log(f"\n[PRECISIONE IN CIMA — quanti dei primi K sono davvero rilevanti]")
    for k in (10, 20, 30, 50):
        log(f"  primi {k:>2}: {y[order[:k]].mean():.0%}  ({int(y[order[:k]].sum())}/{k})")

    log(f"\n[CAUTELE]")
    log("  - Dataset piccolo e non del tutto rappresentativo. È un PRE-FILTRO (ordina/restringe),")
    log("    non la decisione finale: quella la mette il gate LLM (step 4), che non usa queste etichette.")

    np.savez_compressed(MODELS / "reference_set.npz", X=X, y=y)
    log(f"\n  salvato set di riferimento (gli esempi) -> {MODELS/'reference_set.npz'}")
    log("  (nessun modello addestrato da salvare: gli esempi SONO il modello)")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
