"""
Scrape each observatory page and enrich config/osservatori.json
with extended description, topics, and research areas.

Usage:
    python src/enrich_osservatori.py
"""
import html as html_mod
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://www.osservatori.net"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "osservatori.json"
DELAY = 1.5

NOISE = re.compile(
    r"800 033|supporto@|cookie|copyright|scarica|accedi|registr"
    r"|abbonament|carrello|newsletter|privacy|contatt"
    r"|diventa sostenitore|advisory board|esplora i temi"
    r"|le attività dell|dipartimento di ingegneria"
    r"|politecnico di milano|dichiarazione di accessibilità"
    r"|vision e mission|cosa facciamo|som polimi|altri progetti"
    r"|incontri previsti|workshop|convegno pubblico|ai talk"
    r"|accedi per scaricare|le soluzioni per aziende",
    re.IGNORECASE,
)

# "nasce nel YYYY" — marks the origin paragraph
_ORIGIN = re.compile(r"nasce nel \d{4}", re.IGNORECASE)
# FAQ h3 titles: "Cos'è X", "Qual è X", "Quando X", "Come X", "In quali"
_FAQ_H3 = re.compile(r"^(cos['’]è|qual['’]è|quando|come |in quali|quali sono)", re.IGNORECASE)

# nav items concatenated without spaces (e.g. "RisorseInfograficheBusiness")
_NAV_CONCAT = re.compile(r"[a-z][A-Z].*[a-z][A-Z]")
_NAME_LINE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+.{0,5}\(")


class VisibleTextParser(HTMLParser):
    """Collects all visible text from block-level elements, handling nested tags."""

    BLOCK = {"p", "li", "h1", "h2", "h3", "h4"}
    SKIP = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._block_depth = 0
        self._buf = []
        self.blocks = []  # list of (tag, text)
        self._current_block_tag = None

    def handle_starttag(self, tag, _attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK:
            self._block_depth += 1
            if self._block_depth == 1:
                self._buf = []
                self._current_block_tag = tag

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK and self._block_depth > 0:
            self._block_depth -= 1
            if self._block_depth == 0 and self._buf:
                text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
                text = html_mod.unescape(text)
                if text:
                    self.blocks.append((self._current_block_tag, text))
                self._buf = []

    def handle_data(self, data):
        if self._skip_depth or not self._block_depth:
            return
        self._buf.append(data)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def _is_noise(text: str) -> bool:
    if NOISE.search(text):
        return True
    if _NAME_LINE.match(text):
        return True
    if _NAV_CONCAT.search(text):
        return True
    if len(text) < 25:
        return True
    return False


def _extract(raw_html: str) -> dict:
    parser = VisibleTextParser()
    parser.feed(raw_html)

    seen = set()
    origin = None
    description_extended = []
    research_topics = []   # <li> from "Ricerca" section
    faq = []               # h3 question + following p answers

    # track whether we're inside a FAQ section (after an h3 question)
    current_faq_q = None
    current_faq_answers = []

    def flush_faq():
        nonlocal current_faq_q, current_faq_answers
        if current_faq_q and current_faq_answers:
            faq.append({"q": current_faq_q, "a": current_faq_answers[:3]})
        current_faq_q = None
        current_faq_answers = []

    for tag, text in parser.blocks:
        if text in seen:
            continue
        seen.add(text)

        # origin check before noise filter — origin paragraphs often mention Politecnico
        if tag == "p" and not origin and _ORIGIN.search(text):
            origin = text
            continue

        if _is_noise(text):
            continue

        if tag == "h3":
            flush_faq()
            if _FAQ_H3.match(text):
                current_faq_q = text
            continue

        if tag == "p":
            if current_faq_q and len(text) > 40:
                current_faq_answers.append(text)
                continue
            if len(text) > 60:
                description_extended.append(text)

        elif tag == "li" and len(text) > 25:
            research_topics.append(text)

    flush_faq()

    return {
        "origin": origin,
        "description_extended": description_extended[:4],
        "research_topics": research_topics[:10],
        "faq": faq[:8],
    }


def scrape(slug: str) -> dict:
    url = f"{BASE_URL}/{slug}/"
    try:
        raw = _fetch(url)
    except Exception as e:
        print(f"    ERROR: {e}")
        return {}
    return _extract(raw)


def main():
    osservatori = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    total = len(osservatori)

    for i, obs in enumerate(osservatori, 1):
        slug = obs["slug"]

        for k in ("enriched", "description_extended", "topics_scraped", "origin", "research_topics", "faq"):
            obs.pop(k, None)

        print(f"[{i}/{total}] {slug}...")
        extra = scrape(slug)

        if extra:
            obs.update(extra)
            obs["enriched"] = True
            nd = len(extra.get("description_extended", []))
            nr = len(extra.get("research_topics", []))
            nf = len(extra.get("faq", []))
            print(f"    desc: {nd} | research: {nr} | faq: {nf} | origin: {'si' if extra.get('origin') else 'no'}")
        else:
            print(f"    nessun dato")

        CONFIG_PATH.write_text(
            json.dumps(osservatori, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        time.sleep(DELAY)

    print(f"\nFatto. {CONFIG_PATH}")


if __name__ == "__main__":
    main()
