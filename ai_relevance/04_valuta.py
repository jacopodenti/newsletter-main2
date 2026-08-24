"""
STEP 4 — Valuta il ranker SENZA addestramento (stesso schema di ocx_relevance/04_valuta.py),
con controllo di stratificazione per origine incluso da subito (lezione imparata dal test OCX:
il numero aggregato può essere gonfiato dai casi "facili").

Output: models/reference_set.npz  (X = vettori esempi, y = label)
Uso:    python ai_relevance/04_valuta.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
from common import log, detect_lang
from ranker import score_similarity
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = HERE / "data"
MODELS = HERE / "models"


def fold_auc(X, y, n_splits=5, n_repeats=20, k=3, seed=0):
    """AUC out-of-fold — stesso meccanismo usato in produzione da 06_rank.py (confronto
    UNICO, non separato per lingua — vedi ranker.py), per una misura onesta di quanto sarà
    accurato."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rkf.split(X, y))
    aucs = []
    for rep in range(n_repeats):
        oof = np.zeros(len(y))
        for f in range(n_splits):
            tr, te = splits[rep * n_splits + f]
            oof[te] = score_similarity(X[te], X[tr], y[tr], k=k)
        aucs.append(roc_auc_score(y, oof))
    return np.array(aucs)


def main():
    # 1) carica il dataset embeddato + etichette/origine/lingua
    log("STEP 4 — valutazione del ranker AI (NIENTE addestramento: somiglianza agli esempi)")
    d = np.load(MODELS / "dataset_vectors.npz")
    X, y = d["X"], d["y"]
    rows = [json.loads(l) for l in open(DATA / "dataset.jsonl", encoding="utf-8")]
    origin = np.array([r["origin"] for r in rows])
    langs = np.array([detect_lang(f"{r['title']}. {r.get('full_text', '')}") for r in rows])

    log(f"\n[DATI]  totale {len(y)}  |  positivi {int(y.sum())}  |  negativi {int((1-y).sum())}")
    from collections import Counter
    for o, n in Counter(origin).items():
        sub_y = y[origin == o]
        log(f"    origine '{o}': {n}  (Si={int(sub_y.sum())}, No={int((1-sub_y).sum())})")
    log("  per lingua:")
    for lang, n in Counter(langs).items():
        sub_y = y[langs == lang]
        log(f"    '{lang}': {n}  (Si={int(sub_y.sum())}, No={int((1-sub_y).sum())})")

    # 2) AUC su TUTTO il dataset — numero probabilmente gonfiato (vedi controllo al punto 3)
    log("\n[RISULTATO — su TUTTO il dataset, 600 esempi]")
    aucs_all = fold_auc(X, y)
    log(f"  AUC media (20 ripetizioni): {aucs_all.mean():.3f}  (dev.std {aucs_all.std():.3f})")
    log("  ATTENZIONE: 500/600 sono 'editorial_ai', tutti positivi per costruzione — come nel test")
    log("  OCX, questo numero è probabilmente gonfiato. Vedi il controllo sotto.")

    # 3) controllo — AUC solo su 'llm_judged', l'unico gruppo con veri negativi
    # (lezione imparata dal test OCX: il numero aggregato può essere gonfiato dai casi facili)
    mask_llm = origin == "llm_judged"
    Xj, yj = X[mask_llm], y[mask_llm]
    log(f"\n[CONTROLLO — solo sui 100 'llm_judged' (l'unico gruppo con veri negativi)]")
    log(f"  positivi={int(yj.sum())}  negativi={int((1-yj).sum())}  (sbilanciato: pochi negativi anche qui)")
    if len(set(yj)) > 1 and min(Counter(yj).values()) >= 5:
        aucs_llm = fold_auc(Xj, yj, n_splits=5, n_repeats=20)
        log(f"  AUC media (20 ripetizioni, solo llm_judged): {aucs_llm.mean():.3f}  (dev.std {aucs_llm.std():.3f})")
    else:
        log("  troppo pochi esempi in una classe per un fold-CV a 5 gruppi affidabile — numero NON riportato")

    log(f"\n[CAUTELE]")
    log("  - I negativi (81) vengono da giudizio LLM MAI rivisto da una persona — rumore etichette possibile.")
    log("  - I positivi 'editorial_ai' sono newsletter già pubblicate: pertinenti quasi per definizione.")
    log("  - Dataset molto più sbilanciato di OCX (86% positivi qui vs 76% lì).")

    # 4) salva il reference set (esempi + label) usato in produzione dal ranker
    np.savez_compressed(MODELS / "reference_set.npz", X=X, y=y, langs=langs)
    log(f"\n  salvato set di riferimento -> {MODELS/'reference_set.npz'}")
    log("\nFATTO.")


if __name__ == "__main__":
    main()
