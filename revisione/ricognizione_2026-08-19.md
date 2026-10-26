# Ricognizione — kNN vs LLM (Startup Thinking) e tagging content-type/audience

Data: 2026-08-19. Solo analisi — nessun file di pipeline modificato, nessuna chiamata a pagamento.
File letti (in sola lettura): `revisione/StartupThinking_ranking_revisione_umana.xlsx`,
`revisione/StartupThinking_ranking_revisione.xlsx`, `revisione/Confronto_kNN_vs_LLM.xlsx`,
`LABEL-Newsletter.xlsx` (foglio Dizionario), `config/classification_taxonomy.json`, README.md,
e il codice sorgente di `ocx_relevance/`, `ai_relevance/`, `startup_thinking_relevance/`, `src/`.

---

## Punto 1 — kNN vs LLM (gpt-5.4, gpt-5.6-sol, gpt-5.6-luna) contro il giudizio umano

**I file esistono, non serve rifare nulla.** Trovati:

- `revisione/StartupThinking_ranking_revisione_umana.xlsx` → foglio `Da_revisionare`: 150 righe con
  un `Rank` (posizione kNN) e un giudizio umano in `Parere ricercatore` (di Filippo). Distribuzione:
  106 "No", 35 "Si", più alcune etichette ambigue (6 "notizia citata più volte", 1 "Interno",
  1 "Sì, ma titolo sbagliato nel file", 1 "Si, sopra notizia analoga valutata positivamente", 1 vuota).
- `revisione/StartupThinking_ranking_revisione.xlsx` → foglio `LLM_vs_kNN_corpus_intero`: 1370-1378
  articoli con rank per kNN, gpt-5.4, gpt-5.6-sol, gpt-5.6-luna (i valori n=1373-1378 citati
  corrispondono esattamente a questo foglio).

**Attenzione**: sono due run diversi. Il kNN che Filippo ha revisionato (nel file `_umana`) e la
colonna "Rank kNN" nel foglio di confronto non hanno lo stesso ordine esatto (stesso corpus di
1370 articoli, ranking leggermente diverso — coerente con la non-determinismo dell'embedding già
documentato altrove nel progetto). Le due tabelle sono state incrociate per **titolo**, non per
posizione di riga.

### Validazione — riproduco i valori noti prima di calcolare quelli nuovi

Dal file `_umana` da solo (Rank kNN + Parere ricercatore, 150 righe):

| Metrica | Valore noto (fornito) | Valore ricalcolato | Nota |
|---|---|---|---|
| precision@10 | 80% | **80%** | esatto |
| precision@20 | 75% | **75%** | esatto |
| precision@50 | 50% | **48%** | vicino — dipende da come si trattano le 6+1+1 etichette ambigue sopra; con "notizia citata più volte" e "Interno" trattate come No ottengo 48%, con altre combinazioni si va dal 46% al 57% |
| tau | 0,348 | **0,343** (valore assoluto — il segno grezzo è negativo perché Rank 1 = migliore, quindi cresce mentre la relazione col Sì diminuisce) | stessa combinazione di etichette ambigue di sopra |

Considero la corrispondenza sufficiente a confermare che è il dataset giusto e il metodo giusto.
Il residuo (48% vs 50%, 0,343 vs 0,348) è dentro il margine plausibile di come si trattano 8 righe
su 150 con un giudizio non binario secco.

### Confronto con gpt-5.4, gpt-5.6-sol, gpt-5.6-luna

Incrocio riuscito su **149 dei 150** titoli etichettati (1 escluso: "Interno", non classificabile
Sì/No). Per ciascun modello: rank sull'intero corpus, confrontato con l'etichetta umana.

**Kendall tau (rank del modello vs etichetta umana Sì/No, sull'intero campione etichettato — la
misura più solida qui, n=148-149):**

| Modello | tau | n |
|---|---|---|
| kNN (di questo run) | 0,362 | 149 |
| **gpt-5.4** | **0,474** | 148 |
| **gpt-5.6-sol** | **0,431** | 149 |
| **gpt-5.6-luna** | **0,431** | 149 |

Tutti e tre gli LLM hanno tau più alto del kNN contro il giudizio umano, su un campione di
dimensione ragionevole. Questo è il segnale più difendibile che ho.

**Precision@k — stessa tabella, ma con un limite reale da leggere per intero, non solo il numero:**

| Modello | @10 (copertura) | @20 (copertura) | @50 (copertura) |
|---|---|---|---|
| kNN (di questo run) | 88% (8/10 coperti) | 71% (14/20) | 59% (29/50) |
| gpt-5.4 | 100% (**4/10 coperti**) | 78% (9/20) | 73% (15/50) |
| gpt-5.6-sol | 80% (5/10 coperti) | 67% (6/20) | 75% (16/50) |
| gpt-5.6-luna | 100% (**2/10 coperti**) | 67% (6/20) | 56% (18/50) |

**Il limite**: Filippo ha revisionato gli articoli che il kNN ha portato in cima (il suo top-150),
non un campione indipendente. Il vero top-10 di gpt-5.4 e gpt-5.6-luna è in gran parte fatto di
articoli che il kNN non aveva selezionato, quindi mai visti da un umano — la "precision 100%" di
gpt-5.6-luna@10 poggia su **2 soli articoli etichettati**, non è un numero su cui costruire una
decisione. Le colonne @20/@50 hanno una base un po' più larga ma restano parziali.

### Cosa dice, con le cautele sopra

- Il segnale più solido (tau sull'intero campione etichettato) va nella direzione di "gli LLM
  ordinano in modo più vicino al giudizio umano del kNN attuale", per tutti e tre i modelli
  testati (gpt-5.4 il migliore dei tre).
- Il segnale meno solido (precision@10/20) va nella stessa direzione ma è costruito su una
  copertura troppo bassa per essere conclusivo da solo, specialmente al top-10.
- Non è ancora un confronto "ad armi pari": servirebbe un campione di articoli etichettato in modo
  indipendente dal metodo di selezione (non solo il top-150 kNN) per poter fidarsi delle metriche
  di precision@k tanto quanto del tau.

**Nessuna proposta di intervento qui, come richiesto — solo i numeri.**

---

## Punto 2 — Prompt del gate LLM nelle 3 pipeline

Stessa struttura in tutte e tre: `client.responses.create(model=LLM_MODEL, input=prompt)`
(Azure/OpenAI Responses API), `LLM_MODEL = "gpt-5.4"` (un'unica costante in
`ocx_relevance/common.py`, importata anche da ai/startup — non tre copie separate). Nessun
`temperature` impostato (default API). Nessun `response_format`/structured output: si conta sul
prompt ("Rispondi SOLO con JSON") + parsing manuale con fallback su markdown fences. Stesso schema
di retry (backoff `2**attempt` su 429/errori transitori), stesso `CLUSTER_NOTE_TMPL` per segnalare
sospetti duplicati all'LLM, stesso formato di risposta:
`{"rilevante": true/false, "confidence": 0.0-1.0, "motivo": "una frase"}`.

**Differenza reale trovata**: OCX carica la propria descrizione **dinamicamente** da
`config/osservatori.json` (funzione `ocx_description()` in common.py — se il json viene arricchito
in futuro, il prompt lo riflette in automatico). AI e Startup Thinking invece hanno
`AI_DESCRIPTION`/`STARTUP_DESCRIPTION` e i relativi `_CRITERIA` **hardcoded come stringhe** dentro
i rispettivi `05_llm_gate.py`, senza nessun collegamento a `config/osservatori.json`. Se le
descrizioni in quel file cambiano, solo OCX se ne accorge.

---

## Punto 3 — Codice che assegna content_type / target_audience / macro_topics

**Non esiste dentro ocx/ai/startup_thinking_relevance.** Esiste nella pipeline di produzione più
vecchia e separata in `src/`:

- `src/embed_classify.py::classify_article_embed()` — assegna gli **osservatori** per similarità
  coseno tra embedding dell'articolo ed embedding "prototipo" per osservatorio (costruito una
  volta dalla descrizione+topics+FAQ). Zero chiamate LLM per questo campo.
- `src/train_classifiers.py` + `src/embed_classify.py::predict_with_confidence()` — per
  **content_type**, **target_audience**, **macro_topics**: classificatori sklearn (Logistic
  Regression/SVM) allenati sopra gli embedding, con etichette LLM/umane come ground truth,
  salvati in `models/*_classifier.pkl`.
- `src/confidence.py::classify_with_gate()` — orchestratore: usa il classificatore locale se
  esiste, ha `loo_accuracy ≥ 0.80` e la confidence della predizione è `≥ 0.75`; altrimenti richiama
  l'LLM (`src/classify.py::classify_article()`) solo per quel campo, articolo per articolo.

**Tassonomia — `config/classification_taxonomy.json`**: 5 content_type (analysis, research_report,
regulation, vendor_announcement, case_study) e 7 target_audience (executive, marketing, sales,
customer_care, it_data, cross_functional, research).

**Allineamento col Dizionario di `LABEL-Newsletter.xlsx`**: **verificato, combacia esattamente** —
stessi 5 slug CT, stessi 7 slug TA, nello stesso ordine. Nessuna discrepanza trovata (l'ho
controllato riga per riga, non solo il conteggio).

**Stato verificabile — da leggere con attenzione**:
- `models/` non contiene nessun file `.pkl`: non c'è prova che un classificatore sia mai stato
  allenato in questo checkout.
- `output/` è vuota: nessun `results_*.csv` o `ranking_interattivo.xlsx` presente.
- **Il README descrive un'architettura che non corrisponde al codice presente.** Cita
  `src/scoring.py`, `src/export_excel.py` e `config/scoring.json` come i moduli che calcolano uno
  Score pesato (`w_obs*obs_norm + w_ct*(...) + w_ta*(...)`) e generano un workbook Excel
  interattivo a 4 fogli. **Nessuno dei tre esiste in questo repo** (verificato: file assenti,
  nessun riferimento a `openpyxl`/`xlsx`/`Score =` in `src/pipeline.py`). Il README cita anche una
  cartella `old/` con codice archiviato (`old/ai/classify.py`, `old/models/confidence.py` ecc.):
  **quella cartella non esiste**, quei file sono invece direttamente dentro `src/`, attivi
  nell'import di `pipeline.py`, non archiviati.
  Il file README.md ha data di modifica (23/06) precedente al commit iniziale di questo backup
  (16/07): descrive verosimilmente uno stato precedente o pianificato, non quello effettivo di
  questo checkout. Trattare il README come riferimento affidabile per "cosa fa oggi il codice" è
  rischioso — l'ho verificato leggendo `pipeline.py` riga per riga, non fidandomi della descrizione.
- Cosa fa `src/pipeline.py` **davvero** oggi, verificato leggendo `main()`: fetch feed →
  classificazione (LLM diretto, o confidence-gate) → **CSV piatto** (`results_*.csv` /
  `compare_*.csv`) con le predizioni. Nessuno step di scoring pesato, nessun export Excel.

**Conclusione punto 3**: il pattern architetturale (embedding riusato + classificatore leggero +
gate di confidenza verso l'LLM) è reale, ben scritto, e la tassonomia è già coerente con quella che
serve a voi. Ma non è un sistema verificabilmente in produzione oggi in questo checkout, e la parte
di "composizione pesata" che il README descrive come già pronta non esiste come codice.

---

## Punto 4 — Cosa esiste per un compositore bilanciato per content-type/audience

**Niente di funzionante.** Cercato per pattern (`quota`, `bilanci`, `balance`, `composizione`) e
letto il codice: nessun meccanismo di selezione bilanciata per tipo di contenuto o audience in
nessuna delle pipeline (ocx/ai/startup_thinking) né in `src/`.

L'unico riferimento a una logica di composizione è, di nuovo, nel **README** (non nel codice): uno
Score finale pesato con moltiplicatori configurabili per CT/TA (`Config!C12:C26` in un Excel
ipotetico) — che avvicina un articolo alla cima o lo allontana in base al tipo/audience, ma è
comunque un **peso dentro un unico ranking**, non una selezione bilanciata per quota (es. "3 use
case, 2 notizie di mercato, 1 funding"). Come detto al punto 3, il codice che dovrebbe calcolare
questo Score (`scoring.py`) non esiste nel repo.

**In sintesi**: un compositore per la newsletter — pesato o a quota — andrebbe scritto da zero. Il
materiale su cui si appoggerebbe (tag per osservatorio via similarità già calcolabile dagli score
kNN esistenti; tag di content-type/audience via il pattern embedding+classificatore leggero di
`src/embed_classify.py`) è un buon punto di partenza concettuale, ma nessuna parte della catena
selezione→composizione esiste oggi, verificata o meno.
