"""
select_for_newsletter e gruppo_editoriale_di (common.py) — la logica del compositore a
quote. select_for_newsletter legge config/editorial_groups.json reale (non mockato):
i test usano sia testate realmente mappate lì sia testate inventate per verificare il
fallback "una testata non mappata vale come gruppo di se stessa".
"""
from common import select_for_newsletter, gruppo_editoriale_di, load_editorial_groups


def make_candidate(title, rank, pertinenza_llm="Si", fonte="TestataX", content_type=None,
                    gate_confidence=None, url_canonico=None):
    return {
        "title": title,
        "rank": rank,
        "pertinenza_llm": pertinenza_llm,
        "fonte": fonte,
        "content_type": content_type,
        "gate_confidence": gate_confidence,
        "url_canonico": url_canonico or f"https://example.com/{title}",
        "link": f"https://example.com/{title}",
    }


BASE_CONFIG = {
    "n_articoli_totali": 2,
    "max_per_testata": 2,
    "max_per_gruppo_editoriale": 2,
    "max_per_content_type": {},
    "ordine_allentamento": ["ripescaggio", "max_per_testata", "max_per_gruppo_editoriale", "max_per_content_type"],
    "max_round_allentamento": 2,
}


class TestGruppoEditorialeDi:
    def test_testata_mappata_restituisce_il_gruppo(self):
        groups_map = load_editorial_groups()
        # CIO fa parte del gruppo Foundry (ex-IDG) in config/editorial_groups.json
        assert gruppo_editoriale_di("CIO", groups_map) == "Foundry (ex-IDG)"

    def test_testata_non_mappata_vale_come_gruppo_di_se_stessa(self):
        groups_map = load_editorial_groups()
        assert gruppo_editoriale_di("TestataInventataXYZ", groups_map) == "TestataInventataXYZ"


class TestSelectForNewsletterRiempimentoBase:
    def test_due_candidati_validi_riempiono_gli_slot(self):
        candidates = [
            make_candidate("Art1", rank=1, fonte="Fonte1"),
            make_candidate("Art2", rank=2, fonte="Fonte2"),
        ]
        selected, esclusi, log = select_for_newsletter(candidates, BASE_CONFIG, already_selected_urls=set())

        assert [s["title"] for s in selected] == ["Art1", "Art2"]
        assert all(s["_via"] == "gate-si" for s in selected)
        assert esclusi == []
        assert "pieno senza allentare nulla" in log

    def test_ordina_i_si_per_rank_non_per_ordine_di_input(self):
        candidates = [
            make_candidate("Secondo", rank=5, fonte="Fonte1"),
            make_candidate("Primo", rank=1, fonte="Fonte2"),
        ]
        config = {**BASE_CONFIG, "n_articoli_totali": 1}
        selected, _, _ = select_for_newsletter(candidates, config, already_selected_urls=set())
        assert selected[0]["title"] == "Primo"


class TestSelectForNewsletterVincoli:
    def test_max_per_testata_esclude_il_terzo_della_stessa_fonte(self):
        candidates = [
            make_candidate("Art1", rank=1, fonte="StessaFonte"),
            make_candidate("Art2", rank=2, fonte="StessaFonte"),
            make_candidate("Art3", rank=3, fonte="StessaFonte"),
        ]
        config = {**BASE_CONFIG, "n_articoli_totali": 3, "max_per_testata": 2,
                   "ordine_allentamento": [], "max_round_allentamento": 0}
        selected, esclusi, _ = select_for_newsletter(candidates, config, already_selected_urls=set())

        assert [s["title"] for s in selected] == ["Art1", "Art2"]
        assert len(esclusi) == 1
        assert esclusi[0]["title"] == "Art3"
        assert "max_per_testata" in esclusi[0]["motivo_esclusione"]

    def test_max_per_content_type_esclude_oltre_quota(self):
        candidates = [
            make_candidate("Art1", rank=1, fonte="F1", content_type="analysis"),
            make_candidate("Art2", rank=2, fonte="F2", content_type="analysis"),
        ]
        config = {**BASE_CONFIG, "n_articoli_totali": 2,
                   "max_per_content_type": {"analysis": 1},
                   "ordine_allentamento": [], "max_round_allentamento": 0}
        selected, esclusi, _ = select_for_newsletter(candidates, config, already_selected_urls=set())

        assert [s["title"] for s in selected] == ["Art1"]
        assert esclusi[0]["motivo_esclusione"].startswith("max_per_content_type")


class TestSelectForNewsletterRipescaggio:
    def test_recupera_dal_pool_dei_no_quando_i_si_non_bastano(self):
        candidates = [
            make_candidate("UnicoSi", rank=1, pertinenza_llm="Si", fonte="F1"),
            make_candidate("NoSicuro", rank=2, pertinenza_llm="No", fonte="F2", gate_confidence=0.9),
            make_candidate("NoBorderline", rank=3, pertinenza_llm="No", fonte="F3", gate_confidence=0.1),
        ]
        selected, _, log = select_for_newsletter(candidates, BASE_CONFIG, already_selected_urls=set())

        assert [s["title"] for s in selected] == ["UnicoSi", "NoBorderline"]
        assert selected[1]["_via"] == "ripescato"
        assert any("ripescaggio" in line for line in log)

    def test_ripescaggio_usa_rank_come_proxy_se_gate_confidence_manca(self):
        candidates = [
            make_candidate("UnicoSi", rank=1, pertinenza_llm="Si", fonte="F1"),
            make_candidate("NoRankAlto", rank=10, pertinenza_llm="No", fonte="F2"),
            make_candidate("NoRankBasso", rank=2, pertinenza_llm="No", fonte="F3"),
        ]
        selected, _, _ = select_for_newsletter(candidates, BASE_CONFIG, already_selected_urls=set())
        assert selected[1]["title"] == "NoRankBasso"


class TestSelectForNewsletterAllentamento:
    def test_round_di_allentamento_recupera_oltre_il_vincolo_base(self):
        candidates = [
            make_candidate("Art1", rank=1, fonte="StessaFonte"),
            make_candidate("Art2", rank=2, fonte="StessaFonte"),
        ]
        config = {**BASE_CONFIG, "n_articoli_totali": 2, "max_per_testata": 1,
                   "ordine_allentamento": ["max_per_testata"], "max_round_allentamento": 1}
        selected, esclusi, log = select_for_newsletter(candidates, config, already_selected_urls=set())

        assert [s["title"] for s in selected] == ["Art1", "Art2"]
        assert selected[1]["_via"] == "round1:max_per_testata"
        assert esclusi == []


class TestSelectForNewsletterStore:
    def test_gia_pubblicato_mai_recuperabile_nemmeno_allentando(self):
        candidates = [
            make_candidate("GiaPubblicato", rank=1, fonte="F1", url_canonico="https://example.com/pubblicato"),
            make_candidate("Nuovo", rank=2, fonte="F2"),
        ]
        config = {**BASE_CONFIG, "n_articoli_totali": 2, "max_per_testata": 1,
                   "ordine_allentamento": ["max_per_testata"], "max_round_allentamento": 2}
        selected, esclusi, _ = select_for_newsletter(
            candidates, config, already_selected_urls={"https://example.com/pubblicato"}
        )

        assert [s["title"] for s in selected] == ["Nuovo"]
        assert esclusi[0]["title"] == "GiaPubblicato"
        assert "già pubblicato" in esclusi[0]["motivo_esclusione"]
