"""
score_similarity e find_similar_clusters (ranker.py) — pura algebra su vettori,
nessuna chiamata di rete o API. Vettori sintetici piccoli (non serve la dimensione
reale 3072 di text-embedding-3-large: la formula è agnostica alla dimensione).
"""
import numpy as np

from ranker import score_similarity, find_similar_clusters, CLUSTER_SIM_THRESHOLD


class TestScoreSimilarity:
    def setup_method(self):
        # 3 esempi "Si" vicini a [1,0], 3 esempi "No" vicini a [0,1]
        self.X_examples = np.array([
            [1.0, 0.0], [0.9, 0.1], [0.95, 0.05],
            [0.0, 1.0], [0.1, 0.9], [0.05, 0.95],
        ])
        self.y_examples = np.array([1, 1, 1, 0, 0, 0])

    def test_candidato_vicino_ai_positivi_ha_score_positivo(self):
        X_new = np.array([[0.98, 0.02]])
        score = score_similarity(X_new, self.X_examples, self.y_examples, k=3)
        assert score[0] > 0

    def test_candidato_vicino_ai_negativi_ha_score_negativo(self):
        X_new = np.array([[0.02, 0.98]])
        score = score_similarity(X_new, self.X_examples, self.y_examples, k=3)
        assert score[0] < 0

    def test_ordina_correttamente_piu_candidati(self):
        X_new = np.array([[0.98, 0.02], [0.5, 0.5], [0.02, 0.98]])
        scores = score_similarity(X_new, self.X_examples, self.y_examples, k=3)
        assert scores[0] > scores[1] > scores[2]

    def test_k_maggiore_degli_esempi_disponibili_non_solleva_errore(self):
        # _topk_mean usa min(k, esempi disponibili) — non deve esplodere con k=10 su 3 per classe
        X_new = np.array([[1.0, 0.0]])
        score = score_similarity(X_new, self.X_examples, self.y_examples, k=10)
        assert np.isfinite(score[0])

    def test_vettore_non_normalizzato_da_stesso_risultato_di_uno_normalizzato(self):
        # la normalizzazione L2 è interna: scalare il vettore in input non deve cambiare lo score
        X_new_a = np.array([[2.0, 0.0]])
        X_new_b = np.array([[1.0, 0.0]])
        a = score_similarity(X_new_a, self.X_examples, self.y_examples, k=3)
        b = score_similarity(X_new_b, self.X_examples, self.y_examples, k=3)
        assert np.isclose(a[0], b[0])


class TestFindSimilarClusters:
    def test_due_vettori_quasi_identici_finiscono_nello_stesso_cluster(self):
        X = np.array([
            [1.0, 0.0],   # quasi identico al successivo
            [0.99, 0.01],
            [0.0, 1.0],   # ortogonale, nessuna relazione con i primi due
        ])
        clusters = find_similar_clusters(X, threshold=CLUSTER_SIM_THRESHOLD)
        assert clusters[0] is not None
        assert clusters[0] == clusters[1]
        assert clusters[2] is None

    def test_vettori_tutti_distinti_nessun_cluster(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        clusters = find_similar_clusters(X, threshold=CLUSTER_SIM_THRESHOLD)
        assert clusters == [None, None, None]

    def test_soglia_piu_permissiva_puo_unire_vettori_altrimenti_separati(self):
        # due vettori a coseno ~0.8: sopra soglia di default (0.72) ma sotto una soglia stretta (0.95)
        X = np.array([[1.0, 0.0], [0.8, 0.6]])
        assert find_similar_clusters(X, threshold=0.72)[0] is not None
        assert find_similar_clusters(X, threshold=0.95) == [None, None]

    def test_singola_riga_nessun_cluster_possibile(self):
        X = np.array([[1.0, 0.0]])
        assert find_similar_clusters(X) == [None]

    def test_return_similarity_restituisce_anche_la_matrice_di_coseni(self):
        X = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        clusters, sim = find_similar_clusters(X, return_similarity=True)
        assert sim.shape == (3, 3)
        assert np.allclose(np.diag(sim), 1.0)  # un vettore ha coseno 1.0 con se stesso

    def test_return_similarity_con_singola_riga(self):
        X = np.array([[1.0, 0.0]])
        clusters, sim = find_similar_clusters(X, return_similarity=True)
        assert clusters == [None]
        assert sim.shape == (1, 1)
