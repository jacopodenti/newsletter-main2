"""
Tagging degli articoli — content_type e target_audience, CHIAMATA SEPARATA dal gate di
pertinenza. Non tocca il ranking: il kNN resta l'unico meccanismo d'ordinamento, il gate resta
un filtro binario Si/No. Questo modulo assegna solo metadati aggiuntivi, da consumare a valle
(compositore) — non modifica in nessun modo `order`/`scores`/`ranking.csv`.

Perché una chiamata separata e non un campo in più nel gate: testato il 20/08 aggiungendo
content_type/target_audience alla stessa chiamata di pertinenza — l'accordo con il giudizio
umano è sceso (78,0%->70,7%, kappa 0,495->0,389, soprattutto per calo di precisione), quindi il
gate resta invariato e il tagging vive qui.

Content type: few-shot, riusa gli esempi scritti a mano in ../src/classify_ct_llm.py (FEW_SHOT)
e le descrizioni estese di config/classification_taxonomy.json.

Modello: NON hardcoded — letto da config/models.json ("tagging_model"), separato dal modello
del gate ("gate_model"). Verificato il 21/08: gpt-5.4-nano dà una distribuzione content_type
simile a gpt-5.4 (differenza massima 9 punti su 150 articoli reali) a un ventesimo del costo —
promosso a default del tagging. Il gate resta su gpt-5.4: sia nano (recall 35% contro 81-87%)
sia mini (kappa 0,384 sotto la soglia 0,45) falliscono lì, vedi config/models.json.
Diverso anche dalla valutazione dell'84% del 26/06 (gpt-5.4-nano lì per un motivo diverso — era
il default non intenzionale di AZURE_OPENAI_DEPLOYMENT — su scelta forzata senza
non_classificato e input troncato al sommario, non il testo intero: non è la stessa misura).

Target audience: NON "chi ha autorità sul tema" ma "quale funzione trarrebbe beneficio
operativo dall'articolo" — riformulazione richiesta perché il primo tentativo (20/08) produceva
una distribuzione degenere (64% executive, 0% sales, 0% customer_care) contro una distribuzione
umana equilibrata sulle stesse categorie (12% executive, 40% it_data, 32% cross_functional).
"executive" è riservato a contenuti ESPLICITAMENTE strategici, non a qualunque notizia che un
dirigente potrebbe trovare interessante.

In entrambi i casi "non_classificato" è un valore alla pari degli altri, non un fallback dedotto
da una soglia — il modello lo usa se non trova una categoria che descriva chiaramente
l'articolo, invece di scegliere quella più vicina.

Uso: al momento questo modulo espone solo le funzioni di costruzione prompt e classificazione.
NON eseguito ancora — vedi tag_prompts_preview() per vedere il testo esatto dei due prompt senza
fare nessuna chiamata.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ocx_relevance"))
sys.path.insert(0, str(ROOT / "src"))

from common import llm_client  # noqa: E402
from classify_ct_llm import FEW_SHOT as CT_FEW_SHOT  # noqa: E402 — riuso, non copia

TAXONOMY_PATH = ROOT / "config" / "classification_taxonomy.json"
MODELS_CONFIG_PATH = ROOT / "config" / "models.json"
BODY_MAX_CHARS = 6000  # stesso cap usato dal gate (05_llm_gate.py), non il sommario troncato


def _load_taxonomy():
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def tagging_model() -> str:
    """Modello usato dal tagging — parametro esplicito in config/models.json, non hardcoded:
    cambiarlo è una riga in quel file, non una modifica al codice. Verificato il 21/08 su
    gpt-5.4-nano (distribuzione simile a gpt-5.4, costo un ventesimo) — non lo stesso modello
    del gate (config["gate_model"], resta gpt-5.4: nano e mini falliscono lì, vedi _nota_gate_model)."""
    return json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))["tagging_model"]


# --- target_audience: definizioni RISCRITTE sul beneficio operativo, non sull'autorità sul
# tema (quelle di config/classification_taxonomy.json restano "chi ha in carico l'argomento" —
# è esattamente la cornice che ha prodotto la distribuzione degenere del 20/08, quindi qui non
# si riusano). La tassonomia resta l'unica fonte per l'INSIEME degli slug, non per il testo. ---
AUDIENCE_DESCRIPTIONS: dict[str, str] = {
    "executive": "Beneficio operativo per chi decide su investimenti, governance dell'innovazione,"
                 " riorganizzazioni o M&A — SOLO se il contenuto è esplicitamente strategico a"
                 " questo livello, non per qualunque tecnologia o trend che un dirigente"
                 " potrebbe trovare genericamente interessante.",
    "marketing": "Beneficio operativo per chi gestisce campagne, comunicazione digitale,"
                 " martech, personalizzazione o customer engagement — qualcosa che un"
                 " responsabile marketing può applicare direttamente al proprio lavoro.",
    "sales": "Beneficio operativo per chi gestisce il ciclo di vendita — CRM, sales enablement,"
             " pricing, lead generation, strumenti digitali per vendite B2B/B2C.",
    "customer_care": "Beneficio operativo per chi gestisce l'assistenza clienti — piattaforme di"
                     " supporto, chatbot, automazione del supporto, metriche di customer"
                     " satisfaction/NPS.",
    "it_data": "Beneficio operativo per chi lavora su infrastrutture, dati o sviluppo — cloud,"
               " sicurezza, AI/ML, integrazione sistemi, architetture tecniche.",
    "cross_functional": "Beneficio operativo condiviso realmente da PIÙ funzioni insieme, non"
                        " riconducibile a una sola — es. un'iniziativa che ridisegna insieme"
                        " processi, persone e tecnologia in più reparti contemporaneamente.",
    "research": "Materiale di lavoro per chi fa ricerca o analisi — metodologia, dati primari,"
                " framework concettuali utilizzabili nel proprio lavoro, non solo un argomento"
                " interessante da leggere.",
}

# Pool di esempi few-shot scritti per la riformulazione "beneficio operativo", non "autorità
# sul tema". Un esempio per categoria, stesso stile di CT_FEW_SHOT.
AUDIENCE_FEW_SHOT: dict[str, list[tuple[str, str]]] = {
    "executive": [
        (
            "Il CDA di Enel approva un piano di investimento da 400 milioni per l'AI generativa",
            "Il consiglio di amministrazione ha deliberato un investimento pluriennale in AI"
            " generativa come priorità strategica di gruppo, con impatto su struttura"
            " organizzativa e allocazione del budget IT dei prossimi 3 anni."
            " (Decisione di board/investimento rilevante — va in executive, non genericamente"
            " 'AI di tendenza')",
        ),
    ],
    "marketing": [
        (
            "Nike lancia una campagna di personalizzazione AI-driven su 12 mercati europei",
            "Il caso descrive come il team marketing ha costruito segmenti dinamici e contenuti"
            " personalizzati per canale, con un aumento del 18% nel tasso di conversione delle"
            " campagne email. (Beneficio operativo diretto per chi gestisce campagne — va in"
            " marketing)",
        ),
    ],
    "sales": [
        (
            "Salesforce integra la generazione automatica di preventivi nel CRM Sales Cloud",
            "La nuova funzionalità permette ai sales rep di generare preventivi personalizzati"
            " direttamente dal CRM, riducendo il ciclo di vendita B2B di alcuni giorni."
            " (Strumento operativo per chi gestisce il ciclo di vendita — va in sales)",
        ),
    ],
    "customer_care": [
        (
            "Un contact center su 3 automatizza il primo livello di supporto con agenti AI",
            "L'articolo descrive l'adozione di chatbot basati su AI generativa per il primo"
            " livello di assistenza, con impatto su tempi di risposta e CSAT."
            " (Beneficio operativo per chi gestisce l'assistenza clienti — va in customer_care)",
        ),
    ],
    "it_data": [
        (
            "Una banca migra la pipeline dati verso un'architettura lakehouse su cloud ibrido",
            "Il pezzo descrive le scelte tecniche di migrazione, i vincoli di sicurezza e"
            " compliance e l'impatto su latenza e costi di storage."
            " (Contenuto tecnico su infrastrutture/dati — va in it_data)",
        ),
    ],
    "cross_functional": [
        (
            "Un programma di trasformazione digitale ridisegna processi, ruoli e competenze in"
            " tre reparti contemporaneamente",
            "Il caso descrive un programma che coinvolge insieme IT, operations e HR, con"
            " governance condivisa e metriche trasversali — nessuna delle funzioni beneficia"
            " più delle altre. (Beneficio operativo realmente condiviso, non riconducibile a"
            " una sola funzione — va in cross_functional)",
        ),
    ],
    "research": [
        (
            "Uno studio accademico propone un framework per misurare la maturità AI delle PMI",
            "Il paper descrive metodologia, dataset primario su 400 imprese e un framework"
            " concettuale replicabile per classificare il livello di maturità."
            " (Materiale di lavoro per chi fa ricerca/analisi, non solo interessante da leggere"
            " — va in research)",
        ),
    ],
}


def build_content_type_prompt(title: str, body: str) -> str:
    """Prompt few-shot per il content_type. Riusa CT_FEW_SHOT (src/classify_ct_llm.py) e le
    descrizioni estese della tassonomia. Input = testo intero (troncato a 6000 caratteri, come
    il gate), non il sommario troncato a 1500 usato nella valutazione dell'84%."""
    taxonomy = _load_taxonomy()
    lines = [
        "Sei un classificatore di Content Type per articoli di una newsletter B2B di tecnologia"
        " e innovazione digitale, destinata a decision-maker enterprise italiani (CDO, CIO,"
        " responsabili AI) degli Osservatori del Politecnico di Milano.",
        "",
        "Classifica il Content Type dell'articolo scegliendo UNA categoria tra quelle elencate."
        " Se l'articolo non rientra chiaramente in nessuna delle 5 categorie di contenuto,"
        " usa \"non_classificato\" — non forzare una scelta incerta solo per evitarlo.",
        "",
        "## Categorie",
        "",
    ]
    for label in taxonomy["content_type"]:
        slug, desc = label["slug"], label["descrizione"]
        lines.append(f"**{slug}**")
        lines.append(f"Definizione: {desc}")
        examples = CT_FEW_SHOT.get(slug, [])
        if examples:
            lines.append("Esempi:")
            for ex_title, _ in examples:
                lines.append(f'  - "{ex_title}"')
        lines.append("")
    lines.append("**non_classificato**")
    lines.append("Definizione: nessuna delle 5 categorie sopra descrive chiaramente l'articolo,"
                  " o più categorie sono ugualmente plausibili senza un elemento dominante.")
    lines.append("")

    body_trunc = (body or "")[:BODY_MAX_CHARS]
    lines += [
        "## Articolo da classificare",
        "",
        f"Titolo: {title}",
        f"Testo: {body_trunc}",
        "",
        "## Formato di output",
        "",
        "Restituisci SOLO JSON valido, nessun altro testo:",
        '{"content_type": "<analysis|research_report|regulation|vendor_announcement|case_study|non_classificato>", '
        '"confidence": <float 0.0-1.0>, "motivo": "<una frase in italiano>"}',
    ]
    return "\n".join(lines)


def build_audience_prompt(title: str, body: str) -> str:
    """Prompt few-shot per il target_audience. Riformulazione: non 'chi ha autorità sul tema'
    ma 'quale funzione trarrebbe beneficio operativo dall'articolo' — executive riservato a
    contenuti esplicitamente strategici. Un esempio per categoria (AUDIENCE_FEW_SHOT)."""
    taxonomy = _load_taxonomy()
    lines = [
        "Sei un classificatore di Target Audience per articoli di una newsletter B2B di"
        " tecnologia e innovazione digitale, destinata a decision-maker enterprise italiani"
        " degli Osservatori del Politecnico di Milano.",
        "",
        "COMPITO: non stai identificando chi ha autorità formale sul tema trattato, ma quale"
        " funzione aziendale trarrebbe un beneficio OPERATIVO dal contenuto dell'articolo —"
        " qualcosa che può usare direttamente nel proprio lavoro, non solo un argomento che"
        " potrebbe interessarle. \"executive\" è riservato ai contenuti ESPLICITAMENTE"
        " strategici (decisioni di investimento, governance, riorganizzazioni, M&A) — non va"
        " usato per qualunque notizia che un dirigente potrebbe trovare rilevante in generale.",
        "",
        "Scegli UNA categoria tra quelle elencate. Se non è chiaro quale funzione tragga un"
        " beneficio operativo specifico, o il beneficio è troppo generico per essere"
        " attribuito, usa \"non_classificato\" — non forzare una scelta incerta.",
        "",
        "## Categorie",
        "",
    ]
    for label in taxonomy["target_audience"]:
        slug = label["slug"]
        desc = AUDIENCE_DESCRIPTIONS.get(slug, label["descrizione"])
        lines.append(f"**{slug}**")
        lines.append(f"Definizione: {desc}")
        examples = AUDIENCE_FEW_SHOT.get(slug, [])
        if examples:
            lines.append("Esempio:")
            for ex_title, _ in examples:
                lines.append(f'  - "{ex_title}"')
        lines.append("")
    lines.append("**non_classificato**")
    lines.append("Definizione: nessuna funzione trae un beneficio operativo chiaramente"
                  " identificabile dall'articolo, o il beneficio è troppo diffuso/generico.")
    lines.append("")

    body_trunc = (body or "")[:BODY_MAX_CHARS]
    lines += [
        "## Articolo da classificare",
        "",
        f"Titolo: {title}",
        f"Testo: {body_trunc}",
        "",
        "## Formato di output",
        "",
        "Restituisci SOLO JSON valido, nessun altro testo:",
        '{"target_audience": "<executive|marketing|sales|customer_care|it_data|cross_functional|research|non_classificato>", '
        '"confidence": <float 0.0-1.0>, "motivo": "<una frase in italiano>"}',
    ]
    return "\n".join(lines)


def classify_content_type(title: str, body: str, client=None) -> dict:
    """Chiamata al modello di tagging configurato in config/models.json — non il gate,
    non hardcoded. Separata dal gate di pertinenza."""
    client = client or llm_client()
    prompt = build_content_type_prompt(title, body)
    resp = client.responses.create(model=tagging_model(), input=prompt)
    return json.loads(resp.output_text.strip())


def classify_audience(title: str, body: str, client=None) -> dict:
    """Chiamata al modello di tagging configurato in config/models.json — separata dal gate
    e dalla classificazione di content_type."""
    client = client or llm_client()
    prompt = build_audience_prompt(title, body)
    resp = client.responses.create(model=tagging_model(), input=prompt)
    return json.loads(resp.output_text.strip())


def tag_prompts_preview(title: str = "{title}", body: str = "{body}") -> tuple[str, str]:
    """Ritorna (prompt_content_type, prompt_audience) SENZA fare nessuna chiamata — per
    revisionare il testo esatto prima di girarli."""
    return build_content_type_prompt(title, body), build_audience_prompt(title, body)


if __name__ == "__main__":
    ct_prompt, aud_prompt = tag_prompts_preview()
    print("=" * 80)
    print("PROMPT CONTENT_TYPE")
    print("=" * 80)
    print(ct_prompt)
    print()
    print("=" * 80)
    print("PROMPT AUDIENCE")
    print("=" * 80)
    print(aud_prompt)
