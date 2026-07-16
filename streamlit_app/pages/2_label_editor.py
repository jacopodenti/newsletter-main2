import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
from rlhf_store import load_all, mark_reviewed, confirm_reviewed, get_review_queue

ROOT = Path(__file__).parent.parent.parent
OSSERVATORI_PATH = ROOT / "config" / "osservatori.json"
TAXONOMY_PATH = ROOT / "config" / "classification_taxonomy.json"

st.set_page_config(page_title="Label Editor", layout="wide")
st.title("✏️ Label Editor")

# --- load vocab ---
osservatori_data = json.loads(OSSERVATORI_PATH.read_text(encoding="utf-8"))
osservatori_slugs = [o["slug"] for o in osservatori_data]
obs_slug_to_label = {o["slug"]: o["slug"].replace("-", " ").title() for o in osservatori_data}

taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
content_types = [x["slug"] for x in taxonomy["content_type"]]
target_audiences = [x["slug"] for x in taxonomy["target_audience"]]
macro_topics_list = [x["slug"] for x in taxonomy["macro_topics"]]

# --- find osservatori with pending articles ---
all_rows = load_all()
pending_rows = [r for r in all_rows if r.get("reviewed", "False") == "False"]

obs_with_pending = sorted({
    slug.strip()
    for r in pending_rows
    for slug in r.get("observatories", "").split("|")
    if slug.strip()
})

if not obs_with_pending:
    st.success("Nessun articolo da revisionare.")
    st.stop()

# --- Step 1: select observatory ---
selected_obs = st.selectbox(
    "Osservatorio",
    obs_with_pending,
    format_func=lambda s: obs_slug_to_label.get(s, s),
    key="selected_obs",
)

queue = get_review_queue(observatory=selected_obs)

if not queue:
    st.success(f"Nessun articolo da revisionare per **{obs_slug_to_label.get(selected_obs, selected_obs)}**.")
    st.stop()

# --- Navigation ---
article_ids_queue = [r["article_id"] for r in queue]
obs_key = f"edit_id_{selected_obs}"

# handle incoming edit_id from review queue page
if "edit_id" in st.session_state and st.session_state["edit_id"] in article_ids_queue:
    st.session_state[obs_key] = st.session_state.pop("edit_id")

if obs_key not in st.session_state or st.session_state[obs_key] not in article_ids_queue:
    st.session_state[obs_key] = article_ids_queue[0]

current_idx = article_ids_queue.index(st.session_state[obs_key])

col_prev, col_counter, col_next = st.columns([1, 4, 1])
with col_prev:
    if st.button("← Prec.", disabled=(current_idx == 0)):
        st.session_state[obs_key] = article_ids_queue[current_idx - 1]
        st.rerun()
with col_counter:
    st.markdown(f"<div style='text-align:center'>{current_idx + 1} / {len(queue)}</div>", unsafe_allow_html=True)
with col_next:
    if st.button("Succ. →", disabled=(current_idx == len(queue) - 1)):
        st.session_state[obs_key] = article_ids_queue[current_idx + 1]
        st.rerun()

# --- Load article ---
row_map = {r["article_id"]: r for r in all_rows}
row = row_map[st.session_state[obs_key]]

# --- Two-column layout ---
col_article, col_labels = st.columns([5, 4], gap="large")

with col_article:
    st.subheader(row["title"])
    st.caption(f"**Feed:** {row['feed_name']} · **Data:** {row['published_date'][:10]}")
    st.markdown(f"[Apri articolo]({row['link']})")

    with st.expander("Sommario", expanded=True):
        st.write(row.get("summary_clean") or "—")

    if row.get("conf_overall"):
        with st.expander("Confidence scores (modello)", expanded=False):
            def _show_conf(label, val):
                try:
                    f = float(val)
                    st.progress(f, text=f"{label}: {f:.0%}")
                except (ValueError, TypeError):
                    pass
            _show_conf("Content type", row.get("conf_content_type"))
            _show_conf("Target audience", row.get("conf_target_audience"))
            _show_conf("Osservatori", row.get("conf_observatories"))
            _show_conf("Macro topics", row.get("conf_macro_topics"))

with col_labels:
    current_obs = [s.strip() for s in row["observatories"].split("|") if s.strip()]
    current_ct = row.get("content_type") or content_types[0]
    current_ta = row.get("target_audience") or target_audiences[0]
    current_mt = [s.strip() for s in row["macro_topics"].split("|") if s.strip()]

    new_obs = st.multiselect(
        "Osservatori",
        options=osservatori_slugs,
        default=current_obs,
        format_func=lambda s: obs_slug_to_label.get(s, s),
    )
    new_ct = st.selectbox(
        "Content type",
        options=content_types,
        index=content_types.index(current_ct) if current_ct in content_types else 0,
    )
    new_ta = st.selectbox(
        "Target audience",
        options=target_audiences,
        index=target_audiences.index(current_ta) if current_ta in target_audiences else 0,
    )
    new_mt = st.multiselect(
        "Macro topics (max 3)",
        options=macro_topics_list,
        default=current_mt,
        max_selections=3,
    )
    notes = st.text_input("Note revisore (opzionale)", value="")

    st.markdown("---")

    def _advance():
        if current_idx < len(queue) - 1:
            st.session_state[obs_key] = article_ids_queue[current_idx + 1]

    btn_confirm, btn_save, btn_skip = st.columns(3)

    with btn_confirm:
        if st.button("✅ Conferma", use_container_width=True, help="Accetta etichette LLM senza modifiche"):
            confirm_reviewed(row["article_id"])
            st.success("Confermato!")
            _advance()
            st.rerun()

    with btn_save:
        if st.button("💾 Salva", use_container_width=True, type="primary", help="Salva modifiche e vai al prossimo"):
            mark_reviewed(
                row["article_id"],
                labels={
                    "observatories": new_obs,
                    "content_type": new_ct,
                    "target_audience": new_ta,
                    "macro_topics": new_mt,
                },
                reviewer_notes=notes,
            )
            st.success("Salvato!")
            _advance()
            st.rerun()

    with btn_skip:
        if st.button("⏭️ Salta", use_container_width=True):
            _advance()
            st.rerun()
