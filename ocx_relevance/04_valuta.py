"""
STEP 4 — Valuta il ranker SENZA addestramento e salva il set di riferimento.

Nessun modello da addestrare: il ranker confronta la news con gli ESEMPI OCX.
Qui misuriamo quant'è bravo, in modo trasparente (k-fold, ogni numero spiegato),
e salviamo gli esempi (vettori + label) che serviranno da riferimento.

Output: models/reference_set.npz  (X = vettori esempi, y = label)
Uso:    python ocx_relevance/04_valuta.py
"""
import json
import sys

import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import log, MODELS, DATA, detect_lang
from ranker import score_similarity
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score


def main():
    log("STEP 4 — valutazione del ranker (NIENTE addestramento: somiglianza agli esempi)")
    # 1) carica vettori + dataset, rileva la lingua di ogni articolo (serve ai pool per lingua)
    d = np.load(MODELS / "dataset_vectors.npz")
    X, y = d["X"], d["y"]
    rows = [json.loads(l) for l in open(DATA / "dataset.jsonl", encoding="utf-8")]
    langs = np.array([detect_lang(f"{r['title']}. {r.get('full_text', '')}") for r in rows])

    log("\n[COME FUNZIONA IL RANKER]")
    log("  - Nessun modello addestrato. Ogni news prende punteggio dalla SOMIGLIANZA agli esempi:")
    log("    media(coseno con i 3 'Si' più vicini) − media(coseno con i 3 'No' più vicini).")
    log("  - Gli esempi OCX SONO il riferimento. Aggiungere un esempio = aggiungere un vettore.")
    log("  - Confronto UNICO, non separato per lingua (text-embedding-3-large è multilingua per")
    log("    costruzione — vedi ranker.py per perché la separazione per lingua è stata rimossa).")

    from collections import Counter
    log(f"\n[DATI]  totale {len(y)}  |  positivi {int(y.sum())}  |  negativi {int((1-y).sum())}")
    for o, n in Counter(r["origin"] for r in rows).items():
        log(f"    origine '{o}': {n}")
    log("  per lingua:")
    for lang, n in Counter(langs).items():
        sub_y = y[langs == lang]
        log(f"    '{lang}': {n}  (Si={int(sub_y.sum())}, No={int((1-sub_y).sum())})")

    # 2) k-fold ripetuto — stima l'AUC del ranker senza mai valutare su esempi
    # visti nel pool di riferimento dello stesso fold (altrimenti il punteggio sarebbe gonfiato)
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

    # 3) precision@k — quanti dei primi K sono davvero rilevanti, su un ordinamento
    # out-of-fold onesto (un solo giro di 5 fold, non i 20 ripetuti usati per l'AUC sopra)
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
    log("    non la decisione finale: quella la mette il gate LLM (step 5), che non usa queste etichette.")

    # 4) salva il set di riferimento: gli esempi SONO il modello, non c'è nulla da addestrare
    np.savez_compressed(MODELS / "reference_set.npz", X=X, y=y, langs=langs)
    log(f"\n  salvato set di riferimento (gli esempi) -> {MODELS/'reference_set.npz'}")
    log("  (nessun modello addestrato da salvare: gli esempi SONO il modello)")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
