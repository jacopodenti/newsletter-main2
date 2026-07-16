import sys, json
sys.path.insert(0,'src')
from dotenv import load_dotenv; load_dotenv('/Users/gabriele/Desktop/newsletter-main2/.env')
import numpy as np, openpyxl
import embed_classify as ec
ROOT='/Users/gabriele/Desktop/newsletter-main2'
ft=json.load(open(f'{ROOT}/models/fulltext.json'))
dt=openpyxl.load_workbook(f'{ROOT}/LABEL-Newsletter.xlsx',data_only=True)['Dati']
titles=[str(dt.cell(row=d,column=1).value).strip() for d in range(2,1420) if dt.cell(row=d,column=1).value]
def txt(t):
    v=ec.normalize_text(ft.get(t,{}).get('text',t))
    return v if v.strip() else t.lower()   # guard: mai stringa vuota
texts=[txt(t) for t in titles]
empt=sum(1 for x in texts if not x.strip())
print('vuoti residui:',empt,'| articoli:',len(texts))
V=[]
for i in range(0,len(texts),16):
    V.extend(ec._embed(ec._get_client(), texts[i:i+16]))
    if (i//16)%20==0: print(f'  {i}/{len(texts)}',flush=True)
np.savez_compressed(f'{ROOT}/models/corpus_vectors_full.npz', V=np.array(V))
print('Salvato corpus_vectors_full.npz  shape', np.array(V).shape)
print('FINITO')
