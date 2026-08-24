"""
Ranker SENZA addestramento — "somiglianza agli esempi".

Il punteggio di una news = quanto assomiglia agli esempi "Si" più vicini,
meno quanto assomiglia agli esempi "No" più vicini (coseno).
Nessun modello da addestrare: gli esempi OCX SONO il riferimento.
Aggiungere un esempio = aggiungere un vettore, subito usabile.

Confronto UNICO, non separato per lingua (rimossi build_lang_pools/score_similarity_multilang/
MIN_PER_CLASS il 25/08): text-embedding-3-large è multilingua per costruzione, e la separazione
per lingua veniva comunque aggirata dal proprio fallback su Startup Thinking, il cui pool
inglese ha solo 3 esempi Sì (sotto qualunque soglia MIN_PER_CLASS sensata) — ogni articolo
inglese finiva confrontato con l'italiano lo stesso, ma passando per un meccanismo che
sembrava fare una cosa (confronto per lingua) e ne faceva un'altra (fallback quasi sempre).
"""
import numpy as np

# Soglia di coseno sopra la quale due articoli tra i candidati sono "sospetti duplicati"
# (stessa notizia/evento raccontata da fonti diverse). Calibrata su un caso reale
# (22/07): la coppia confermata dello stesso vertice USA-Cina sull'AI, una fonte IT e
# una EN, aveva coseno 0.744; le coppie sullo stesso tema ma notizie diverse stavano
# per lo più sotto 0.66. È un filtro per SOSPETTI (alto richiamo, non precisione
# garantita): il gate LLM commenta il sospetto e il revisore umano decide.
CLUSTER_SIM_THRESHOLD = 0.72


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


def find_similar_clusters(X, threshold=CLUSTER_SIM_THRESHOLD, return_similarity=False):
    """Raggruppa le righe di X (embedding) i cui coseni superano `threshold`: probabile
    stessa notizia/evento riportato da fonti diverse.

    Complete-linkage, NON union-find: un gruppo si forma solo se OGNI coppia al suo
    interno supera la soglia, non una semplice catena di coppie vicine (A~B~C non
    implica A~C). Con union-find, su dati reali (22/07, corpus OCX) 44 articoli su 150
    finivano incollati in un solo cluster — decine di saggi generici su "AI in azienda"
    incatenati a due a due, non vere notizie duplicate. Complete-linkage evita il
    problema: ogni cluster resta un gruppo di articoli TUTTI vicini tra loro.

    Ritorna una lista lunga come X: un id di cluster (intero >=1) per riga in un gruppo
    di 2+ elementi, None per le righe senza nessun gruppo abbastanza coeso.

    `return_similarity=True` (default False, comportamento invariato per i chiamanti
    esistenti): ritorna anche la matrice di coseni (n, n) già calcolata qui dentro —
    serve a chi vuole metadati per cluster (similarità media/min/max interna, vedi
    06_rank.py) senza ricalcolarla da zero."""
    from collections import Counter
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    Xn = _l2(X)
    n = len(Xn)
    if n < 2:
        empty_sim = np.zeros((n, n))
        return ([None] * n, empty_sim) if return_similarity else [None] * n

    sim = Xn @ Xn.T
    dist = 1 - sim
    dist = (dist + dist.T) / 2  # forza simmetria esatta (arrotondamenti in virgola mobile)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="complete")
    flat = fcluster(Z, t=1 - threshold, criterion="distance")

    sizes = Counter(flat)
    cluster_id = [None] * n
    id_map = {}
    next_id = 0
    for i, lab in enumerate(flat):
        if sizes[lab] < 2:
            continue
        if lab not in id_map:
            next_id += 1
            id_map[lab] = next_id
        cluster_id[i] = id_map[lab]
    return (cluster_id, sim) if return_similarity else cluster_id
