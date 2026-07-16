import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pickle
import streamlit as st
from rlhf_store import get_training_set, stats

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"

ACCURACY_GATE = 0.80
MIN_EXAMPLES = 100

st.set_page_config(page_title="Model Metrics", layout="wide")
st.title("📊 Model Metrics")

s = stats()
st.caption(f"{s['reviewed']} articoli revisionati su {s['total']} totali")

fields = ["content_type", "target_audience", "macro_topics"]
reviewed_all = get_training_set()


def _load_model(field: str) -> dict | None:
    path = MODELS_DIR / f"{field}_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


rows_data = []
for field in fields:
    m = _load_model(field)
    n = len([r for r in reviewed_all if r.get(field, "").strip()])
    if m is None:
        acc = None
        status = "🔴 LLM only"
    else:
        acc = m.get("loo_accuracy", 0.0)
        if acc >= ACCURACY_GATE and n >= MIN_EXAMPLES:
            status = "🟢 Autonomous"
        elif acc >= ACCURACY_GATE:
            status = f"🟡 Hybrid (need {MIN_EXAMPLES - n} more examples)"
        else:
            status = f"🟡 Hybrid (accuracy {acc:.2f} < {ACCURACY_GATE})"
    rows_data.append({"Campo": field, "LOO Accuracy": f"{acc:.2f}" if acc is not None else "—",
                      "Esempi reviewed": n, "Status": status})

st.table(rows_data)

# progress bars toward gate
st.markdown("### Progresso verso autonomia")
for r in rows_data:
    field = r["Campo"]
    acc_val = float(r["LOO Accuracy"]) if r["LOO Accuracy"] != "—" else 0.0
    n_val = r["Esempi reviewed"]
    acc_pct = min(acc_val / ACCURACY_GATE, 1.0)
    ex_pct = min(n_val / MIN_EXAMPLES, 1.0)
    col1, col2 = st.columns(2)
    col1.progress(acc_pct, text=f"{field} accuracy: {acc_val:.2f}/{ACCURACY_GATE}")
    col2.progress(ex_pct, text=f"{field} esempi: {n_val}/{MIN_EXAMPLES}")

st.markdown("---")

# retraining
st.markdown("### Retraining")
min_ex = st.number_input("Esempi minimi per campo", min_value=5, max_value=500, value=20)

if st.button("🔄 Avvia Retraining", type="primary"):
    with st.spinner("Retraining in corso..."):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "retrain.py"), "--min-reviewed", str(min_ex)],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        st.success("Retraining completato!")
        st.code(result.stdout)
    else:
        st.error("Errore durante il retraining")
        st.code(result.stderr)
