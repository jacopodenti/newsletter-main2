import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="Newsletter RLHF",
    page_icon="📰",
    layout="wide",
)

from rlhf_store import stats

st.title("📰 Newsletter RLHF — Human Review")

s = stats()
col1, col2, col3 = st.columns(3)
col1.metric("Totale articoli", s["total"])
col2.metric("Revisionati", s["reviewed"])
col3.metric("Da revisionare", s["pending"])

if s["total"] > 0:
    pct = round(s["reviewed"] / s["total"] * 100)
    st.progress(pct / 100, text=f"{pct}% completato")
else:
    st.info("Nessun articolo ancora. Esegui il pipeline per popolare il dataset.")

st.markdown("---")
st.markdown("""
**Pagine disponibili:**
- **Review Queue** — lista articoli da revisionare, filtrabile per osservatorio
- **Label Editor** — revisiona articolo per articolo, per osservatorio
- **Model Metrics** — accuracy modelli e retraining
- **Drift Alerts** — monitoraggio distribuzione classi
""")
