"""
canonicalize_url, normalize_text, parse_date/format_date (common.py) — pura logica di
stringa/data, nessuna chiamata di rete o API.
"""
from datetime import timezone

from common import canonicalize_url, normalize_text, parse_date, format_date


TRACKING = {"utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid"}


class TestCanonicalizeUrl:
    def test_rimuove_parametri_di_tracking(self):
        url = "https://Example.com/articolo?utm_source=newsletter&id=42"
        assert canonicalize_url(url, TRACKING) == "https://example.com/articolo?id=42"

    def test_host_minuscolo(self):
        assert canonicalize_url("https://EXAMPLE.com/x", TRACKING) == "https://example.com/x"

    def test_rimuove_slash_finale(self):
        assert canonicalize_url("https://example.com/articolo/", TRACKING) == canonicalize_url(
            "https://example.com/articolo", TRACKING
        )

    def test_rimuove_fragment(self):
        assert "#" not in canonicalize_url("https://example.com/x#section2", TRACKING)

    def test_due_url_equivalenti_solo_per_tracking_diventano_uguali(self):
        a = "https://example.com/x?utm_source=fb&utm_campaign=lancio"
        b = "https://example.com/x?gclid=abc123"
        assert canonicalize_url(a, TRACKING) == canonicalize_url(b, TRACKING)

    def test_due_url_con_path_diverso_restano_diversi(self):
        a = canonicalize_url("https://example.com/articolo-1", TRACKING)
        b = canonicalize_url("https://example.com/articolo-2", TRACKING)
        assert a != b

    def test_stringa_vuota_invariata(self):
        assert canonicalize_url("", TRACKING) == ""

    def test_query_senza_parametri_di_tracking_resta_intatta(self):
        url = "https://example.com/x?id=42&page=2"
        out = canonicalize_url(url, TRACKING)
        assert "id=42" in out and "page=2" in out


class TestNormalizeText:
    def test_minuscolo(self):
        assert normalize_text("Articolo IMPORTANTE") == "articolo importante"

    def test_rimuove_tag_html(self):
        assert normalize_text("<p>Testo <b>in grassetto</b></p>") == "testo in grassetto"

    def test_unescape_entita_html(self):
        assert normalize_text("Rock &amp; Roll") == "rock & roll"

    def test_collassa_spazi_multipli(self):
        assert normalize_text("troppi    spazi\n\ntra  le parole") == "troppi spazi tra le parole"

    def test_strip_di_testa_e_coda(self):
        assert normalize_text("   testo con margini   ") == "testo con margini"

    def test_input_vuoto_o_none(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestParseDate:
    def test_rfc822(self):
        dt = parse_date("Tue, 21 Jul 2026 08:47:09 +0000")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 7 and dt.day == 21

    def test_iso8601_con_timezone(self):
        # formato usato da The Verge/Business Insider/Fast Company (fix del 18/08)
        dt = parse_date("2026-08-18T09:35:35-04:00")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 8 and dt.day == 18

    def test_iso8601_senza_timezone_normalizzato_a_utc(self):
        dt = parse_date("2026-08-18T09:35:35")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc or dt.utcoffset().total_seconds() == 0

    def test_stringa_vuota_restituisce_none(self):
        assert parse_date("") is None
        assert parse_date(None) is None

    def test_stringa_non_interpretabile_restituisce_none(self):
        assert parse_date("non e' una data") is None

    def test_format_date_su_data_valida(self):
        assert format_date("Tue, 21 Jul 2026 08:47:09 +0000") == "21/07/2026"

    def test_format_date_su_data_non_valida_restituisce_stringa_vuota(self):
        assert format_date("boh") == ""
