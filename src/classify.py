import json
import os
from pathlib import Path

from openai import OpenAI


def load_osservatori(path: str = "config/osservatori.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "prompt_template.txt"


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(article: dict, osservatori: list[dict], template: str) -> str:
    obs_lines = "\n".join(
        f"- {o['slug']}: {o.get('descrizione_enhanced') or o['descrizione']}"
        for o in osservatori
    )
    replacements = {
        "{title}": article["title"],
        "{summary}": article["summary"],
        "{link}": article["link"],
        "{published_date}": article["published_date"],
        "{osservatori_list}": obs_lines,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def normalize_observatories(values: list, osservatori: list[dict]) -> list[str]:
    id_to_slug = {o["id"]: o["slug"] for o in osservatori}
    slug_set = {o["slug"] for o in osservatori}
    result = []
    for v in values:
        v = str(v).strip()
        if v in slug_set:
            result.append(v)
        elif v in id_to_slug:
            result.append(id_to_slug[v])
    return result


_FALLBACK = {
    "observatories": [],
    "content_type": None,
    "target_audience": None,
    "macro_topics": [],
    "reason": "Output del modello non parsabile in JSON.",
}


def classify_article(
    article: dict,
    client: OpenAI,
    osservatori: list[dict],
    deployment: str,
    template: str | None = None,
) -> dict:
    if template is None:
        template = load_prompt_template()
    prompt = build_prompt(article, osservatori, template)
    for attempt in range(5):
        try:
            response = client.responses.create(
                model=deployment,
                input=prompt,
            )
            break
        except Exception as e:
            if "429" in str(e) or "too_many_requests" in str(e).lower():
                import time
                time.sleep(2 ** attempt)
            else:
                raise
    else:
        return _FALLBACK.copy()
    raw = response.output_text

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # strip possible markdown fences
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return _FALLBACK.copy()

    return {
        "observatories": normalize_observatories(parsed.get("observatories", []), osservatori),
        "content_type": parsed.get("content_type"),
        "target_audience": parsed.get("target_audience"),
        "macro_topics": parsed.get("macro_topics", []),
        "reason": parsed.get("reason"),
    }
