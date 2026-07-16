import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
from rlhf_store import get_training_set

ROOT = Path(__file__).parent.parent.parent
DRIFT_PATH = ROOT / "data" / "drift_baseline.json"
DRIFT_THRESHOLD = 0.15  # 15pp change triggers alert

st.set_page_config(page_title="Drift Alerts", layout="wide")
st.title("📡 Drift Alerts")

if not DRIFT_PATH.exists():
    st.warning("Nessun baseline disponibile. Esegui il retraining almeno una volta per generarlo.")
    st.stop()

baseline = json.loads(DRIFT_PATH.read_text(encoding="utf-8"))
reviewed = get_training_set()

# last 30 days
from datetime import datetime, timedelta, timezone
cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()[:10]
recent = [r for r in reviewed if r.get("review_timestamp", "") >= cutoff]

st.caption(f"Confronto baseline (training) vs ultimi 30 giorni ({len(recent)} articoli)")

fields = ["content_type", "target_audience", "macro_topics"]
alerts = []

for field in fields:
    st.markdown(f"### {field}")

    base_counts = baseline.get(field, {})
    base_total = sum(base_counts.values()) or 1

    if field == "macro_topics":
        recent_counts: Counter = Counter()
        for r in recent:
            for s in r.get(field, "").split("|"):
                if s.strip():
                    recent_counts[s.strip()] += 1
    else:
        recent_counts = Counter(r.get(field, "") for r in recent if r.get(field, "").strip())

    recent_total = sum(recent_counts.values()) or 1

    all_classes = sorted(set(list(base_counts.keys()) + list(recent_counts.keys())))
    if not all_classes:
        st.info("Nessun dato.")
        continue

    table_rows = []
    for cls in all_classes:
        base_pct = base_counts.get(cls, 0) / base_total
        recent_pct = recent_counts.get(cls, 0) / recent_total
        delta = recent_pct - base_pct
        flag = ""
        if abs(delta) >= DRIFT_THRESHOLD:
            flag = "⚠️"
            alerts.append(f"{field}/{cls}: {delta:+.0%}")
        elif cls not in recent_counts:
            flag = "🔴"
            alerts.append(f"{field}/{cls}: scomparso")
        table_rows.append({
            "Classe": cls,
            "Baseline": f"{base_pct:.0%}",
            "Recente (30d)": f"{recent_pct:.0%}",
            "Delta": f"{delta:+.0%}",
            "": flag,
        })

    st.table(table_rows)

st.markdown("---")
if alerts:
    st.error(f"**{len(alerts)} alert attivi:**")
    for a in alerts:
        st.write(f"  - {a}")
else:
    st.success("Nessun drift significativo rilevato.")
