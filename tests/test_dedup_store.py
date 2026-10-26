"""
content_hash, dedup_by_content_hash, group_by_cluster (common.py) — logica pura su
dict/liste, nessuna chiamata di rete o API.
"""
from common import content_hash, dedup_by_content_hash, group_by_cluster


class TestContentHash:
    def test_full_text_produce_un_hash(self):
        a = {"title": "Titolo", "full_text": "Corpo dell'articolo", "text_source": "full"}
        h = content_hash(a)
        assert h is not None
        assert len(h) == 64  # sha256 esadecimale

    def test_fallback_non_produce_hash(self):
        a = {"title": "Titolo", "full_text": "", "text_source": "fallback"}
        assert content_hash(a) is None

    def test_stesso_contenuto_normalizzato_stesso_hash(self):
        a = {"title": "Titolo Uno", "full_text": "Corpo del primo articolo", "text_source": "full"}
        b = {"title": "titolo   uno", "full_text": "CORPO DEL PRIMO ARTICOLO", "text_source": "full"}
        assert content_hash(a) == content_hash(b)

    def test_contenuto_diverso_hash_diverso(self):
        a = {"title": "Titolo Uno", "full_text": "Corpo del primo articolo", "text_source": "full"}
        b = {"title": "Titolo Due", "full_text": "Corpo completamente diverso", "text_source": "full"}
        assert content_hash(a) != content_hash(b)


class TestDedupByContentHash:
    def test_duplicato_esatto_scartato_fallback_sempre_tenuto(self):
        a = {"title": "Titolo Uno", "full_text": "Corpo del primo articolo", "text_source": "full"}
        b = {"title": "titolo   uno", "full_text": "CORPO DEL PRIMO ARTICOLO", "text_source": "full"}
        c = {"title": "Titolo Due", "full_text": "Corpo diverso", "text_source": "full"}
        d = {"title": "Titolo Tre", "full_text": "", "text_source": "fallback"}

        kept, counts = dedup_by_content_hash([a, b, c, d])

        assert kept == [a, c, d]
        assert counts == {"saltati_fallback": 1, "scartati_duplicato_hash": 1}

    def test_nessun_duplicato_tiene_tutto(self):
        a = {"title": "A", "full_text": "Testo A", "text_source": "full"}
        b = {"title": "B", "full_text": "Testo B", "text_source": "full"}
        kept, counts = dedup_by_content_hash([a, b])
        assert kept == [a, b]
        assert counts == {"saltati_fallback": 0, "scartati_duplicato_hash": 0}


class TestGroupByCluster:
    def test_membri_dello_stesso_cluster_accostati_dopo_la_prima_occorrenza(self):
        rows = [
            {"title": "A", "cluster_id": None},
            {"title": "B", "cluster_id": 1},
            {"title": "C", "cluster_id": None},
            {"title": "D", "cluster_id": 1},
            {"title": "E", "cluster_id": 2},
        ]
        ordered = group_by_cluster(rows)
        assert [r["title"] for r in ordered] == ["A", "B", "D", "C", "E"]

    def test_nessun_cluster_ordine_invariato(self):
        rows = [{"title": "A", "cluster_id": None}, {"title": "B", "cluster_id": None}]
        assert group_by_cluster(rows) == rows
