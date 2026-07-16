"""
Content Type classification via Azure OpenAI (few-shot).

Usage as module:
    from classify_ct_llm import classify_ct
    result = classify_ct(title, summary)
    # result = {"label": "analysis", "confidence": 0.92, "reason": "..."}

Usage as CLI:
    python src/classify_ct_llm.py --title "..." --summary "..."
"""
import json
import os
import re
import argparse
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = ROOT / "config" / "classification_taxonomy.json"

VALID_LABELS = {"analysis", "research_report", "regulation", "vendor_announcement", "case_study"}


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


# Few-shot examples per label — 3 per "analysis" per mostrare che opinion piece,
# trend article e interview confluiscono tutti in quella categoria.
FEW_SHOT: dict[str, list[tuple[str, str]]] = {
    "analysis": [
        (
            "Why Most Enterprise AI Projects Fail Before Reaching Production",
            "A former Fortune 500 CDO argues that the real barrier to AI ROI is organizational"
            " culture, not technology. He identifies three structural patterns that systematically"
            " kill AI initiatives before they scale. (Opinion piece — va in analysis)",
        ),
        (
            "Generative AI Adoption Accelerates in European Enterprises: 2026 Market Overview",
            "The article maps the adoption trajectory of GenAI across large European companies,"
            " discusses emerging trends and analyst projections through 2028."
            " (Trend article — va in analysis)",
        ),
        (
            "'AI Is Not a Silver Bullet': The CDO of ING on the Real Limits of Banking Automation",
            "In this Q&A, ING's Chief Data Officer discusses the bank's three-year AI"
            " transformation and why most AI failures stem from poor problem framing."
            " (Intervista — va in analysis)",
        ),
    ],
    "research_report": [
        (
            "Gartner Survey: 78% of Enterprises Plan to Increase AI Budget in 2027",
            "Gartner polled 1,400 CIOs across 18 countries. Average AI spending will grow 34%"
            " YoY; talent acquisition is the #1 barrier; ROI measurement remains the top"
            " challenge for 62% of respondents.",
        ),
        (
            "Osservatori Polimi 2026: Il 60% delle PMI Italiane Ha Avviato Almeno un Progetto"
            " di Digital Transformation",
            "Lo studio, condotto su 820 PMI con fatturato tra 10 e 500 milioni, evidenzia che"
            " cloud e cybersecurity sono le aree prioritarie. Solo il 18% ha implementato"
            " soluzioni di AI generativa.",
        ),
    ],
    "regulation": [
        (
            "EU AI Act Enters into Force: Compliance Deadlines for High-Risk System Providers",
            "Providers of high-risk AI systems in HR, credit scoring, and critical infrastructure"
            " must complete conformity assessments by August 2026. The article summarizes key"
            " timelines, obligations, and potential fines.",
        ),
        (
            "Garante Privacy Sanziona Meta per Violazioni GDPR nel Trattamento dei Dati dei Minori",
            "Il Garante ha comminato una sanzione di 200 milioni di euro a Meta per aver trattato"
            " dati di minori senza adeguata base giuridica. La decisione segue un'indagine"
            " avviata nel 2024.",
        ),
    ],
    "vendor_announcement": [
        (
            "Salesforce Launches Einstein Copilot for Service Cloud With Real-Time Sentiment Analysis",
            "Salesforce announced general availability of Einstein Copilot integrated into Service"
            " Cloud. The AI assistant surfaces next-best-action recommendations and automates case"
            " summarization for contact center agents.",
        ),
        (
            "Microsoft and SAP Expand Partnership to Bring Copilot Into ERP Workflows",
            "The companies signed an agreement to embed Microsoft 365 Copilot directly into SAP"
            " S/4HANA interfaces. Partnership e accordi commerciali tra vendor vanno in"
            " vendor_announcement.",
        ),
    ],
    "case_study": [
        (
            "How Zalando Cut Return Rates by 23% With AI-Powered Size Recommendations",
            "The fashion retailer deployed a computer vision model trained on 500M product images"
            " to suggest accurate sizing at checkout. The 18-month project reduced return-related"
            " logistics costs by €47M annually.",
        ),
        (
            "UniCredit's Biometric Onboarding Journey: From Pilot to Full Rollout in 14 Months",
            "The bank replaced its manual KYC process with facial recognition and liveness"
            " detection. The case describes regulatory challenges, vendor selection, integration"
            " steps, and the final 92% digital onboarding rate.",
        ),
    ],
}


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_ct_labels() -> list[dict]:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return taxonomy["content_type"]


def _build_prompt(title: str, summary: str, ct_labels: list[dict]) -> str:
    lines = [
        "Sei un classificatore di articoli per una newsletter B2B di tecnologia e innovazione digitale,"
        " destinata a decision-maker enterprise italiani (CDO, CIO, responsabili AI)"
        " degli Osservatori del Politecnico di Milano.",
        "",
        "Classifica il Content Type dell'articolo scegliendo ESATTAMENTE UNA delle 5 categorie seguenti.",
        "Devi SEMPRE restituire una delle 5 label: non esiste una risposta 'incerto'."
        " Se hai dubbi, scegli la categoria più probabile.",
        "",
        "## Categorie",
        "",
    ]

    for label in ct_labels:
        slug = label["slug"]
        desc = label["descrizione"]
        lines.append(f"**{slug}**")
        lines.append(f"Definizione: {desc}")
        examples = FEW_SHOT.get(slug, [])
        if examples:
            lines.append("Esempi:")
            for ex_title, _ in examples:
                lines.append(f'  - "{ex_title}"')
        lines.append("")

    clean_summary = _strip_html(summary)[:1500]

    lines += [
        "## Articolo da classificare",
        "",
        f"Titolo: {title}",
        f"Sommario: {clean_summary}",
        "",
        "## Formato di output",
        "",
        'Restituisci SOLO JSON valido, nessun altro testo:',
        '{"label": "<uno dei 5 slug>", "confidence": <float 0.0-1.0>, "reason": "<una frase in italiano>"}',
        "",
        "Scala di confidence:",
        "  0.9-1.0 → categoria inequivocabile",
        "  0.7-0.9 → plausibile ma qualche ambiguità",
        "  < 0.7   → più categorie applicabili, ma scegli comunque la più probabile",
    ]
    return "\n".join(lines)


def classify_ct(
    title: str,
    summary: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict:
    """
    Classifies article content type using Azure OpenAI.

    Returns dict with keys:
        label       – one of 5 CT slugs (always a valid label, never "uncertain")
        confidence  – float 0-1
        reason      – one-sentence explanation in Italian
    """
    if not isinstance(client, OpenAI):
        client = _build_client()
    if model is None:
        model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    ct_labels = _load_ct_labels()
    prompt = _build_prompt(title, summary, ct_labels)

    response = client.responses.create(model=model, input=prompt)
    raw_text = response.output_text.strip()

    # strip markdown code fences if present
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "label": "analysis",
            "confidence": 0.0,
            "reason": f"JSON parse error — fallback a 'analysis'. Raw output: {raw_text[:200]}",
        }

    raw_label = str(result.get("label", ""))
    confidence = float(result.get("confidence", 0.0))
    reason = str(result.get("reason", ""))

    if raw_label not in VALID_LABELS:
        # mappa label obsolete al rimpiazzo più vicino
        legacy_map = {
            "opinion_analysis": "analysis",
            "trend_article": "analysis",
            "interview": "analysis",
            "partnership_event": "vendor_announcement",
            "uncertain": "analysis",
        }
        raw_label = legacy_map.get(raw_label, "analysis")
        reason = f"(label rimappata) {reason}"

    return {
        "label": raw_label,
        "confidence": confidence,
        "reason": reason,
    }


if __name__ == "__main__":
    default_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    parser = argparse.ArgumentParser(description="Classifica Content Type di un articolo via Azure OpenAI")
    parser.add_argument("--title", required=True, help="Titolo dell'articolo")
    parser.add_argument("--summary", default="", help="Sommario dell'articolo")
    parser.add_argument("--model", default=None, help=f"Deployment Azure (default: {default_deployment})")
    args = parser.parse_args()

    out = classify_ct(args.title, args.summary, model=args.model)
    print(json.dumps(out, ensure_ascii=False, indent=2))
