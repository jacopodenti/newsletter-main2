"""
Ranker SENZA addestramento — "somiglianza agli esempi".

Il punteggio di una news = quanto assomiglia agli esempi "Si" più vicini,
meno quanto assomiglia agli esempi "No" più vicini (coseno).
Nessun modello da addestrare: gli esempi OCX SONO il riferimento.
Aggiungere un esempio = aggiungere un vettore, subito usabile.
"""
import numpy as np


def _l2(M):
    M = np.asarray(M, dtype=float)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _topk_mean(S, k):
    k = min(k, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(axis=1)


def score_similarity(X_new, X_examples, y_examples, k=3):
    """Punteggio di rilevanza per ogni riga di X_new, confrontando con gli esempi.
    score = media(coseno con i k 'Si' più vicini) − media(coseno con i k 'No' più vicini)."""
    Xn = _l2(X_new)
    E = _l2(X_examples)
    y = np.asarray(y_examples)
    pos, neg = E[y == 1], E[y == 0]
    sp = _topk_mean(Xn @ pos.T, k)
    sn = _topk_mean(Xn @ neg.T, k)
    return sp - sn


def clean_threshold(scores, labels, margin_frac=0.5):
    """Soglia 'zona pulita': appena sopra il primo 'No' scendendo dall'alto."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(-scores)
    first_no = next((p for p, i in enumerate(order) if labels[i] == 0), None)
    if first_no is None:
        return float(scores.min() - 1e-6)
    if first_no == 0:
        return float(scores[order[0]] + 1e-6)
    hi = scores[order[first_no - 1]]
    lo = scores[order[first_no]]
    return float(lo + (hi - lo) * margin_frac)
