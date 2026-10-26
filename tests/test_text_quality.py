"""
Rilevamento qualità testo: token sospetti, taglio del testo offuscato, declassamento per
dominio noto e per collisione tra articoli della stessa testata (common.py). Pura logica
su stringhe/dict, nessuna chiamata di rete o API.
"""
from common import (
    _is_suspicious_token,
    _suspicious_frac,
    _find_obfuscation_cut,
    truncate_obfuscated_text,
    downgrade_known_bad_domains,
    downgrade_colliding_fulltext,
)

CLEAN_PARA = (
    "Il mercato della space economy italiana cresce grazie a nuovi investimenti nel "
    "settore satellitare e nella space economy europea con numerosi progetti industriali "
    "e universitari che generano interesse concreto delle imprese"
)
CLEAN_SHORT = "Oggi il tempo e sereno ovunque"
GARBLED_PARA = "qzxcvb wplmnk bfghjkl strngth wwwzzz qxcvbn mnbvcx zzzzaa qplmzx vbnmqw"


class TestIsSuspiciousToken:
    def test_parola_normale_non_sospetta(self):
        assert _is_suspicious_token("casa") is False
        assert _is_suspicious_token("sviluppo") is False

    def test_parola_con_qu_non_sospetta(self):
        assert _is_suspicious_token("quando") is False

    def test_q_senza_u_e_sospetta(self):
        assert _is_suspicious_token("qwerty") is True

    def test_cinque_consonanti_di_fila_sospetta(self):
        assert _is_suspicious_token("strngth") is True

    def test_lettera_ripetuta_tre_volte_sospetta(self):
        assert _is_suspicious_token("wowww") is True

    def test_parola_troppo_corta_mai_sospetta(self):
        assert _is_suspicious_token("ok") is False
        assert _is_suspicious_token("qq") is False


class TestSuspiciousFrac:
    def test_testo_pulito_frazione_bassa(self):
        text = " ".join([CLEAN_PARA] * 2)
        assert _suspicious_frac(text) < 0.1

    def test_testo_tutto_offuscato_frazione_alta(self):
        text = " ".join([GARBLED_PARA] * 3)
        assert _suspicious_frac(text) > 0.9

    def test_testo_troppo_corto_restituisce_none(self):
        assert _suspicious_frac("poche parole qui", min_tokens=20) is None

    def test_min_tokens_personalizzato(self):
        assert _suspicious_frac("una due tre quattro cinque", min_tokens=5) is not None


class TestFindObfuscationCut:
    def test_trova_il_punto_dove_comincia_il_degrado(self):
        paras = [CLEAN_PARA, GARBLED_PARA, GARBLED_PARA]
        cut = _find_obfuscation_cut(paras, threshold=0.10)
        assert cut == 1  # tiene solo il primo paragrafo (pulito)

    def test_singolo_paragrafo_offuscato_ritorna_zero(self):
        # nessun indice i>=1 da valutare: non c'e' nessun prefisso pulito da salvare
        assert _find_obfuscation_cut([GARBLED_PARA], threshold=0.10) == 0

    def test_nessun_degrado_rilevante_tiene_tutto(self):
        paras = [CLEAN_PARA, CLEAN_PARA, CLEAN_PARA]
        # con testo tutto pulito, nessun taglio "bad" si attiva in nessun punto >=1:
        # la funzione ritorna 0 (fallback) perché il ciclo non trova mai 80% di paragrafi sospetti
        assert _find_obfuscation_cut(paras, threshold=0.10) == 0


class TestTruncateObfuscatedText:
    def test_prefisso_pulito_piu_lungo_del_summary_diventa_partial(self):
        articles = [{
            "fonte": "Sifted",
            "text_source": "full",
            "full_text": f"{CLEAN_PARA}\n{GARBLED_PARA}\n{GARBLED_PARA}",
            "summary": "Breve.",
        }]
        result = truncate_obfuscated_text(articles, threshold=0.10)
        assert articles[0]["text_source"] == "partial"
        assert articles[0]["full_text"] == CLEAN_PARA
        assert result["partial"].get("Sifted") == 1

    def test_prefisso_pulito_piu_corto_del_summary_diventa_fallback(self):
        summary_lungo = (
            "Il riassunto RSS originale di questo articolo e' piu' lungo del paragrafo "
            "iniziale rimasto pulito, quindi il confronto finale deve ricadere sul "
            "fallback invece che sul parziale."
        )
        articles = [{
            "fonte": "Sifted",
            "text_source": "full",
            "full_text": f"{CLEAN_SHORT}\n{GARBLED_PARA}\n{GARBLED_PARA}",
            "summary": summary_lungo,
        }]
        result = truncate_obfuscated_text(articles, threshold=0.10)
        assert articles[0]["text_source"] == "fallback"
        assert articles[0]["full_text"] == summary_lungo
        assert result["fallback"].get("Sifted") == 1

    def test_testo_pulito_non_viene_toccato(self):
        articles = [{
            "fonte": "Wired",
            "text_source": "full",
            "full_text": " ".join([CLEAN_PARA] * 3),
            "summary": "Breve.",
        }]
        original = articles[0]["full_text"]
        result = truncate_obfuscated_text(articles, threshold=0.10)
        assert articles[0]["text_source"] == "full"
        assert articles[0]["full_text"] == original
        assert result == {"partial": {}, "fallback": {}}

    def test_articolo_gia_in_fallback_viene_ignorato(self):
        articles = [{"fonte": "X", "text_source": "fallback", "summary": "s"}]
        result = truncate_obfuscated_text(articles, threshold=0.10)
        assert articles[0]["text_source"] == "fallback"
        assert result == {"partial": {}, "fallback": {}}


class TestDowngradeKnownBadDomains:
    def setup_method(self):
        self.bad_domains = {
            "badsite.com": {"estensione": "sempre"},
            "partial-site.it": {"estensione": "parziale"},
            "sifted.eu": {"estensione": "sempre"},
        }

    def test_dominio_sempre_viene_declassato(self):
        articles = [{"link": "https://badsite.com/x", "fonte": "BadSite",
                     "text_source": "full", "full_text": "testo catturato male", "summary": "riassunto"}]
        counts = downgrade_known_bad_domains(articles, self.bad_domains)
        assert articles[0]["text_source"] == "fallback"
        assert articles[0]["full_text"] == "riassunto"
        assert counts == {"BadSite": 1}

    def test_dominio_parziale_non_viene_toccato(self):
        articles = [{"link": "https://partial-site.it/y", "fonte": "Partial",
                     "text_source": "full", "full_text": "testo originale", "summary": "riassunto2"}]
        counts = downgrade_known_bad_domains(articles, self.bad_domains)
        assert articles[0]["text_source"] == "full"
        assert articles[0]["full_text"] == "testo originale"
        assert counts == {}

    def test_sifted_eu_escluso_anche_se_sempre(self):
        # gestito da truncate_obfuscated_text, non da questa funzione
        articles = [{"link": "https://sifted.eu/z", "fonte": "Sifted",
                     "text_source": "full", "full_text": "testo originale", "summary": "riassunto3"}]
        counts = downgrade_known_bad_domains(articles, self.bad_domains)
        assert articles[0]["text_source"] == "full"
        assert counts == {}

    def test_dominio_non_in_lista_non_viene_toccato(self):
        articles = [{"link": "https://goodsite.com/w", "fonte": "Good",
                     "text_source": "full", "full_text": "testo originale", "summary": "riassunto4"}]
        counts = downgrade_known_bad_domains(articles, self.bad_domains)
        assert articles[0]["text_source"] == "full"
        assert counts == {}

    def test_articolo_gia_in_fallback_ignorato(self):
        articles = [{"link": "https://badsite.com/already", "fonte": "BadSite2",
                     "text_source": "fallback", "full_text": "", "summary": "x"}]
        counts = downgrade_known_bad_domains(articles, self.bad_domains)
        assert counts == {}


class TestDowngradeCollidingFulltext:
    WIDGET_TEXT = "Questo e' un testo widget ripetuto identico catturato per errore dall'estrattore"

    def test_stessa_testata_testo_identico_entrambi_declassati(self):
        articles = [
            {"fonte": "TestataX", "text_source": "full", "full_text": self.WIDGET_TEXT, "summary": "sommario A1"},
            {"fonte": "TestataX", "text_source": "full", "full_text": self.WIDGET_TEXT, "summary": "sommario A2"},
        ]
        counts = downgrade_colliding_fulltext(articles)
        assert articles[0]["text_source"] == "fallback" and articles[0]["full_text"] == "sommario A1"
        assert articles[1]["text_source"] == "fallback" and articles[1]["full_text"] == "sommario A2"
        assert counts == {"TestataX": 2}

    def test_stessa_testata_testo_diverso_non_toccato(self):
        articles = [
            {"fonte": "TestataX", "text_source": "full", "full_text": self.WIDGET_TEXT, "summary": "s1"},
            {"fonte": "TestataX", "text_source": "full",
             "full_text": "Un articolo completamente diverso su un argomento distinto e originale",
             "summary": "s2"},
        ]
        counts = downgrade_colliding_fulltext(articles)
        assert articles[0]["text_source"] == "full"
        assert articles[1]["text_source"] == "full"
        assert counts == {}

    def test_testate_diverse_stesso_testo_non_confrontate(self):
        articles = [
            {"fonte": "TestataX", "text_source": "full", "full_text": self.WIDGET_TEXT, "summary": "s1"},
            {"fonte": "TestataY", "text_source": "full", "full_text": self.WIDGET_TEXT, "summary": "s2"},
        ]
        counts = downgrade_colliding_fulltext(articles)
        assert articles[0]["text_source"] == "full"
        assert articles[1]["text_source"] == "full"
        assert counts == {}
