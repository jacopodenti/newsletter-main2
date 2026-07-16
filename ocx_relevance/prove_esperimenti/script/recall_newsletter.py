"""
Test di recall sui 110 articoli editoriali (OCX26_Newsletter.xlsx).
Sono tutti POSITIVI veri: quanti li avrebbe pescati il nostro ranking?
"""
import sys, json, pickle
sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv("/Users/gabriele/Desktop/newsletter-main2/.env")
import numpy as np
import openpyxl
import embed_classify as ec
from knn_relevance import KNNRelevanceScorer, load_labeled_set

# --- 1. carica i 110 articoli editoriali (dedup per link) ---
wb = openpyxl.load_workbook("OCX26_Newsletter.xlsx", data_only=True)
picks, seen = [], set()
for ws in wb.worksheets:
    hdr = {str(ws.cell(row=1, column=c).value).strip().lower(): c
           for c in range(1, ws.max_column + 1) if ws.cell(row=1, column=c).value}
    tcol = hdr.get("titolo"); scol = hdr.get("riassunto"); lcol = hdr.get("link")
    ncol = next((v for k, v in hdr.items() if k.startswith("note")), None)
    for r in range(2, ws.max_row + 1):
        t = ws.cell(row=r, column=tcol).value if tcol else None
        if not t or not str(t).strip():
            continue
        link = str(ws.cell(row=r, column=lcol).value or "") if lcol else ""
        if link and link in seen:
            continue
        seen.add(link)
        note = str(ws.cell(row=r, column=ncol).value or "").strip().lower() if ncol else ""
        is_neg = any(w in note for w in ["non va", "fuori scope", "non bene", "scartato"])
        summ = str(ws.cell(row=r, column=scol).value or "") if scol else ""
        picks.append({"title": str(t), "summary": summ, "neg": is_neg, "top": note == "top"})
pos = [p for p in picks if not p["neg"]]
print(f"Articoli editoriali: {len(picks)} (positivi {len(pos)}, note-negative {len(picks)-len(pos)})")

# --- 2. embedding dei picks ---
client = ec._get_client()
texts = [ec.normalize_text(f"{p['title']}. {p['summary']}") for p in picks]
E = []
for i in range(0, len(texts), 16):
    E.extend(ec._embed(client, texts[i:i+16]))
E = np.array(E)

# --- vettore OCX + corpus + kNN sui 50 ---
ocx = np.array(next(o["embedding"] for o in pickle.load(open("models/observatory_embeddings.pkl","rb"))["embeddings"]
                    if o["slug"] == "omnichannel-customer-experience"))
ocx = ocx / np.linalg.norm(ocx)
try:
    corpus = np.load("/private/tmp/claude-501/-Users-gabriele-Desktop-newsletter-main2/950c4057-6d9f-4d00-8ac2-c7388f4fad62/scratchpad/all_vectors.npz")["V"]
    corpus_ocx = (corpus / np.linalg.norm(corpus, axis=1, keepdims=True)) @ ocx
except FileNotFoundError:
    corpus_ocx = None  # distribuzione nota: min -0.02, mediana 0.19, max 0.49

rows50 = load_labeled_set("LABEL-Newsletter.backup.xlsx", sheet="Ranking")
t50 = [ec.normalize_text(f"{r['title']}. {r['summary']}") for r in rows50]
X50 = np.array([e for i in range(0, len(t50), 16) for e in ec._embed(client, t50[i:i+16])])
y50 = np.array([r["label"] for r in rows50])
knn = KNNRelevanceScorer(k=3).fit(X50, y50)

# --- 3. score dei picks ---
picks_ocx = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ ocx
picks_knn = knn.score_batch(E)
# kNN clean threshold sui 50 (LOO)
loo = np.zeros(len(y50))
for i in range(len(y50)):
    m = np.ones(len(y50), bool); m[i] = False
    loo[i] = KNNRelevanceScorer(k=3).fit(X50[m], y50[m]).score(X50[i])
order = np.argsort(-loo); first_no = next(p for p, i in enumerate(order) if y50[i] == 0)
thr_knn = (loo[order[first_no-1]] + loo[order[first_no]]) / 2

pos_mask = np.array([not p["neg"] for p in picks])
po = picks_ocx[pos_mask]; pk = picks_knn[pos_mask]

CORPUS_MED = np.median(corpus_ocx) if corpus_ocx is not None else 0.19

print("\n=== RECALL su coseno OCX (soglia produzione 0.35) ===")
print(f"  picks positivi score OCX: min={po.min():.3f} mediana={np.median(po):.3f} max={po.max():.3f}")
print(f"  >= 0.35 (soglia attuale): {(po>=0.35).sum()}/{len(po)} = {(po>=0.35).mean():.0%}")
if corpus_ocx is not None:
    print(f"  percentile mediano nel corpus 1418: {np.median([(corpus_ocx<v).mean()*100 for v in po]):.0f}%")
print(f"  sotto mediana corpus ({CORPUS_MED:.2f}): {(po<CORPUS_MED).sum()}/{len(po)}")

print("\n=== RECALL su kNN (soglia pulita %.3f) ===" % thr_knn)
print(f"  picks positivi score kNN: min={pk.min():.3f} mediana={np.median(pk):.3f} max={pk.max():.3f}")
print(f"  >= soglia pulita: {(pk>=thr_knn).sum()}/{len(pk)} = {(pk>=thr_knn).mean():.0%}")
print(f"  >= 0 (piu vicino a Si che a No): {(pk>=0).sum()}/{len(pk)} = {(pk>=0).mean():.0%}")

print("\n=== I 10 picks editoriali con score OCX piu BASSO (i 'persi') ===")
idx = np.argsort(po)
poslist = [p for p in picks if not p["neg"]]
for j in idx[:10]:
    print(f"  ocx={po[j]:.3f} knn={pk[j]:+.3f}  {poslist[j]['title'][:60]}")
