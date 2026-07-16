"""
(A) Riscoro i 1418 col kNN addestrato sul set espanso (159 esempi) e scrivo
il foglio 'Ranking_kNN_v2'. Vettori corpus salvati in models/ (persistente).
"""
import os, sys
sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv("/Users/gabriele/Desktop/newsletter-main2/.env")
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import embed_classify as ec
from knn_relevance import KNNRelevanceScorer, load_labeled_set, clean_threshold

ROOT = "/Users/gabriele/Desktop/newsletter-main2"
CORPUS_VEC = f"{ROOT}/models/corpus_vectors.npz"

# --- articoli corpus (Dati) ---
src = openpyxl.load_workbook("LABEL-Newsletter.xlsx", data_only=True)
dt = src["Dati"]
arts = []
for d in range(2, 1420):
    t = dt.cell(row=d, column=1).value
    if not t: continue
    arts.append({"title": str(t).strip(),
                 "summary": str(dt.cell(row=d, column=17).value or ""),
                 "fonte": str(dt.cell(row=d, column=4).value or ""),
                 "data": str(dt.cell(row=d, column=3).value or ""),
                 "link": str(dt.cell(row=d, column=2).value or ""),
                 "ct": str(dt.cell(row=d, column=6).value or "")})

client = ec._get_client()
if os.path.exists(CORPUS_VEC):
    V = np.load(CORPUS_VEC)["V"]; print("corpus da cache")
else:
    print(f"riembeddo {len(arts)} articoli...")
    texts = [ec.normalize_text(f"{a['title']}. {a['summary']}") for a in arts]
    V = []
    for i in range(0, len(texts), 16):
        V.extend(ec._embed(client, texts[i:i+16]))
        if (i//16) % 15 == 0: print(f"  {i}/{len(texts)}")
    V = np.array(V); np.savez_compressed(CORPUS_VEC, V=V)
assert len(V) == len(arts)

# --- modello espanso ---
d = np.load(f"{ROOT}/models/train_expanded.npz")
Xtr, ytr = d["X"], d["y"]
knn = KNNRelevanceScorer(k=3).fit(Xtr, ytr)

# soglia ricalibrata (LOO sul training)
loo = np.zeros(len(ytr))
for i in range(len(ytr)):
    m = np.ones(len(ytr), bool); m[i] = False
    loo[i] = KNNRelevanceScorer(k=3).fit(Xtr[m], ytr[m]).score(Xtr[i])
thr = clean_threshold(loo, ytr, margin_frac=0.5)

scores = knn.score_batch(V)
order = np.argsort(-scores)
above = int((scores >= thr).sum())
print(f"Soglia {thr:.3f} | articoli sopra: {above}/{len(arts)}")

# label note (dai 50)
lab = {r["title"].strip(): r["label"] for r in load_labeled_set("LABEL-Newsletter.backup.xlsx", sheet="Ranking")}

# --- scrivi foglio ---
wb = openpyxl.load_workbook("LABEL-Newsletter.xlsx")
if "Ranking_kNN_v2" in wb.sheetnames: del wb["Ranking_kNN_v2"]
ws = wb.create_sheet("Ranking_kNN_v2")
hf = Font(bold=True, color="FFFFFF"); hd = PatternFill("solid", fgColor="1F6F3E")
green = PatternFill("solid", fgColor="E2EFDA"); red = PatternFill("solid", fgColor="FCE4D6")
ws["A1"] = "Ranking kNN v2 — modello ESPANSO (159 esempi: 106 editoriali + 50). Score kNN, tutti i 1418."
ws["A1"].font = Font(bold=True, size=12)
ws["A2"] = f"Soglia pulita {thr:.3f} -> {above} articoli sopra. Verde=Si noto, Rosso=No noto."
ws["A2"].font = Font(italic=True, size=10, color="666666")
heads = ["Rank","Score kNN","Sopra soglia","Label nota","Titolo","Sommario","Fonte","Data","Content Type","Link"]
for c,h in enumerate(heads,1):
    cell=ws.cell(row=4,column=c,value=h); cell.font=hf; cell.fill=hd
for rank,i in enumerate(order,1):
    a=arts[i]; row=4+rank; L=lab.get(a["title"])
    ws.cell(row=row,column=1,value=rank)
    ws.cell(row=row,column=2,value=round(float(scores[i]),4))
    ws.cell(row=row,column=3,value="SI" if scores[i]>=thr else "")
    ws.cell(row=row,column=4,value=("Si" if L==1 else "No") if L is not None else "")
    ws.cell(row=row,column=5,value=a["title"])
    ws.cell(row=row,column=6,value=a["summary"][:400])
    ws.cell(row=row,column=7,value=a["fonte"]); ws.cell(row=row,column=8,value=a["data"])
    ws.cell(row=row,column=9,value=a["ct"]); ws.cell(row=row,column=10,value=a["link"])
    if L==0:
        for c in range(1,11): ws.cell(row=row,column=c).fill=red
    elif L==1:
        for c in range(1,11): ws.cell(row=row,column=c).fill=green
    ws.cell(row=row,column=6).alignment=Alignment(wrap_text=True,vertical="top")
for col,w in {"A":6,"B":10,"C":12,"D":10,"E":55,"F":60,"G":18,"H":20,"I":18,"J":28}.items():
    ws.column_dimensions[col].width=w
ws.freeze_panes="A5"
wb.save("LABEL-Newsletter.xlsx")
print(f"Foglio 'Ranking_kNN_v2' scritto. Fogli: {wb.sheetnames}")

print("\nTOP 15 (modello espanso):")
for rank,i in enumerate(order[:15],1):
    L=lab.get(arts[i]["title"]); m={1:"Si  ",0:"NO x",None:". "}[L]
    fl="OK" if scores[i]>=thr else ""
    print(f"  #{rank:>2} {scores[i]:+.3f} [{m}] {fl:>3}  {arts[i]['title'][:52]}")
