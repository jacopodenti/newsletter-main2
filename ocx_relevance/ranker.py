"""
Ranker SENZA addestramento — "somiglianza agli esempi".

Il punteggio di una news = quanto assomiglia agli esempi "Si" più vicini,
meno quanto assomiglia agli esempi "No" più vicini (coseno).
Nessun modello da addestrare: gli esempi OCX SONO il riferimento.
Aggiungere un esempio = aggiungere un vettore, subito usabile.
"""
import numpy as np

# Sotto questa soglia di esempi per classe, il pool di una lingua è troppo
# rumoroso/sbilanciato per fare kNN affidabile (es. "0 positivi inglesi, 73
# negativi inglesi": qualunque articolo inglese finirebbe classificato "No"
# solo perché non esistono esempi positivi con cui confrontarlo). In quel
# caso si usa il pool di fallback (italiano: è quello sempre solido).
MIN_PER_CLASS = 10

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


def build_lang_pools(X, y, langs):
    """Divide gli esempi di riferimento in un pool per lingua.
    Ritorna {lingua: (X_lingua, y_lingua)}."""
    X, y, langs = np.asarray(X), np.asarray(y), np.asarray(langs)
    return {lang: (X[langs == lang], y[langs == lang]) for lang in set(langs)}


def score_similarity_multilang(X_new, langs_new, pools, fallback_lang="it", k=3, min_per_class=MIN_PER_CLASS):
    """Come score_similarity, ma ogni articolo è confrontato SOLO con gli esempi della sua
    stessa lingua (rilevanza linguistica: un articolo inglese va confrontato con esempi
    inglesi, non con l'italiano — pool misti mascherano un bias quando lingua e label sono
    correlate nel set di riferimento). Se il pool della lingua di un articolo ha meno di
    `min_per_class` esempi Sì o No, quel pool non è abbastanza affidabile: si ricade sul
    pool di `fallback_lang` (di norma l'italiano, il più numeroso e bilanciato)."""
    X_new = np.asarray(X_new)
    langs_new = np.asarray(langs_new)
    scores = np.zeros(len(X_new))
    fallback_X, fallback_y = pools[fallback_lang]
    for lang in set(langs_new):
        idx = np.where(langs_new == lang)[0]
        Xp, yp = pools.get(lang, (None, None))
        usable = Xp is not None and (yp == 1).sum() >= min_per_class and (yp == 0).sum() >= min_per_class
        if not usable:
            Xp, yp = fallback_X, fallback_y
        scores[idx] = score_similarity(X_new[idx], Xp, yp, k=k)
    return scores


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
