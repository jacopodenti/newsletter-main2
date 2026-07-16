import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
from rlhf_store import get_review_queue, load_all, stats

st.set_page_config(page_title="Review Queue", layout="wide")
st.title("📋 Review Queue")

ROOT = Path(__file__).parent.parent.parent
OSSERVATORI_PATH = ROOT / "config" / "osservatori.json"
osservatori_data = json.loads(OSSERVATORI_PATH.read_text(encoding="utf-8"))
obs_slug_to_label = {o["slug"]: o["slug"].replace("-", " ").title() for o in osservatori_data}

s = stats()
st.caption(f"{s['reviewed']}/{s['total']} revisionati — {s['pending']} in coda")

all_rows = load_all()
feed_names = sorted({r["feed_name"] for r in all_rows if r.get("feed_name")})

pending_rows = [r for r in all_rows if r.get("reviewed", "False") == "False"]
obs_with_pending = sorted({
    slug.strip()
    for r in pending_rows
    for slug in r.get("observatories", "").split("|")
    if slug.strip()
})

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    obs_filter = st.selectbox(
        "Osservatorio",
        ["(tutti)"] + obs_with_pending,
        format_func=lambda s: s if s == "(tutti)" else obs_slug_to_label.get(s, s),
    )
with col2:
    feed_filter = st.selectbox("Feed", ["(tutti)"] + feed_names)
with col3:
    date_filter = st.text_input("Data minima (YYYY-MM-DD)", placeholder="es. 2026-04-01")

obs_arg = None if obs_filter == "(tutti)" else obs_filter
feed_arg = None if feed_filter == "(tutti)" else feed_filter
date_arg = date_filter.strip() or None

queue = get_review_queue(feed_name=feed_arg, date_from=date_arg, observatory=obs_arg)

st.markdown(f"**{len(queue)} articoli in coda**")

if not queue:
    st.success("Nessun articolo da revisionare con questi filtri.")
    st.stop()

for row in queue:
    with st.expander(f"[{row['published_date'][:10]}] {row['title'][:100]}", expanded=False):
        st.write(f"**Feed:** {row['feed_name']}")
        st.write(f"**Link:** {row['link']}")
        st.write(f"**Sommario:** {(row.get('summary_clean') or '')[:400]}")
        st.write(f"**Osservatori (LLM):** {row['observatories'] or '—'}")
        st.write(f"**Content type:** {row['content_type'] or '—'}")
        st.write(f"**Target audience:** {row['target_audience'] or '—'}")
        st.write(f"**Macro topics:** {row['macro_topics'] or '—'}")
        if st.button("Apri in editor →", key=f"open_{row['article_id']}"):
            st.session_state["edit_id"] = row["article_id"]
            st.switch_page("pages/2_label_editor.py")
