# OCX Relevance — soluzione sperimentale (isolata dalla produzione)

Cartella **a sé**. Non tocca `../src` né la pipeline di produzione.
Serve a testare l'approccio: **embedding (testo intero) → ranker per somiglianza → gate LLM**.

Stesso schema, stesso codice condiviso (`common.py`, `ranker.py`) e stesso corpus di
`../ai_relevance` e `../startup_thinking_relevance`: le 3 pipeline sono concettualmente
identiche, cambiano solo le testate/descrizione specifiche dell'osservatorio.

## Come funziona (in 7 step)

```
articoli
  1. testo INTERO scaricato dagli URL (non più il sommario da 25 parole)
  2. EMBEDDING  text-embedding-3-large (3072 dim)
  3. RANKER     SOMIGLIANZA agli esempi OCX, per lingua (niente addestramento) → score  [usa gli esempi]
  4. GATE LLM   gpt-5.4 legge l'articolo e dice si/no di pertinenza                     [NON usa le etichette]
  5. RANK       pipeline completa sul corpus condiviso (../corpus/corpus_raw.jsonl):
                embedding + ranker + gate sui primi N -> data/ranking.csv
  6. EXPORT     scrive il ranking come foglio Excel dentro LABEL-Newsletter.xlsx
  7. REVISIONE  costruisce un foglio "Da_revisionare" curato per il controllo umano
```

Il ranker NON addestra nulla: gli esempi OCX SONO il riferimento. Il punteggio di una news =
`media(coseno con i 3 esempi "Si" più vicini) − media(coseno con i 3 esempi "No" più vicini)`,
calcolato **dentro la stessa lingua** dell'articolo (pool separati per lingua in `ranker.py`,
con fallback sul pool italiano se la lingua dell'articolo ha meno di 10 esempi Sì o No —
altrimenti un pool sbilanciato, es. "0 positivi inglesi", darebbe punteggi inaffidabili).
Aggiungere un esempio = aggiungere un vettore, subito usabile (scala a 60 osservatori senza ri-addestrare).

**Divisione dei compiti (perché regge con poche etichette):**
- Il **ranker** usa le poche etichette solo come *pre-filtro economico* (ordina/restringe).
- Il **gate LLM** fa la decisione di precisione **senza usare le nostre etichette** (usa la descrizione OCX + il suo giudizio).
- Quindi la parte fragile (poche label) è solo il pre-filtro; la precisione la porta l'LLM.

## Ordine di esecuzione (da bash — lancia tu e ispeziona i log)

Numerazione allineata a `ai_relevance`/`startup_thinking_relevance`: qui non c'è uno STEP 1
perché OCX non costruisce candidati negativi via LLM (`01_build_negative_candidates.py`) come
le altre due — usa direttamente le etichette già presenti in `LABEL-Newsletter.xlsx`.

```bash
cd /Users/gabriele/Desktop/newsletter-main2

# STEP 2 — costruisce il dataset nuovo (con testo intero) e lo salva
venv/bin/python ocx_relevance/02_build_dataset.py

# STEP 3 — embedding del dataset
venv/bin/python ocx_relevance/03_embed.py

# STEP 4 — VALUTA il ranker (niente addestramento; ogni numero è spiegato nel log) e salva reference_set.npz
venv/bin/python ocx_relevance/04_valuta.py

# STEP 5 (opzionale) — verifica del gate LLM sul dataset (precisione vs etichette)
venv/bin/python ocx_relevance/05_llm_gate.py ocx_relevance/data/dataset.jsonl

# STEP 6 — pipeline completa sul corpus condiviso (ranker + LLM sui top-N)
venv/bin/python ocx_relevance/06_rank.py --topn 150

# STEP 7 — scrive il foglio Excel col ranking (dentro LABEL-Newsletter.xlsx)
venv/bin/python ocx_relevance/07_export_excel.py

# STEP 8 — costruisce il foglio "Da_revisionare" per il controllo umano
venv/bin/python ocx_relevance/08_build_review_sheet.py
```

Il corpus (`../corpus/corpus_raw.jsonl`) è condiviso e va aggiornato a parte, solo ogni tanto:
`venv/bin/python corpus/fetch_corpus.py`.

## Dataset (`data/dataset.jsonl`)
Una riga per articolo: `title, link, full_text, text_source, label, origin, id`.

| origine | cosa | label |
|---|---|---|
| `set50` | i 50 già etichettati | 26 Si / 24 No |
| `editorial` | articoli scelti per le newsletter OCX | 106 Si / 3 No |
| `offtopic_manual` | 25 palesemente fuori tema (dal corpus) | 25 No |

Ogni esempio traccia l'origine → puoi filtrare/togliere una fonte quando vuoi
(es. buttare via `set50`).

## Interpretabilità delle performance (step 3)
Il log dello step 3 spiega **come** si ottiene ogni numero:
- **k-fold stratificato 5×20**: ogni articolo è giudicato solo da un modello che non l'ha mai visto (niente leakage).
- **AUC**: probabilità che un rilevante prenda punteggio più alto di un non-rilevante (0.5 = caso, 1.0 = perfetto).
- **precisione@K**: quanti dei primi K sono davvero rilevanti.
- Stampa anche le **cautele** (dataset piccolo/parziale; il numero misura il pre-filtro, non la decisione finale).

## Output
- `data/dataset.jsonl` / `dataset_meta.json` — il dataset.
- `models/dataset_vectors.npz` — embedding.
- `models/reference_set.npz` — gli esempi (vettori + label) = il riferimento del ranker (nessun modello addestrato).
- `data/ranking.csv` — classifica finale (ranker + pertinenza LLM).

## Modelli usati (unico punto: `common.py`)
- Embedding: `text-embedding-3-large` (3072 dim), Azure.
- LLM: `gpt-5.4`, Azure.
- Ranker: somiglianza agli esempi (coseno, k=3) — **nessun addestramento** (`ranker.py`).

## Nota
Le mie modifiche a `../src/embed_classify.py`, `pipeline.py`, `confidence.py` (fatte prima,
sul percorso `--gate` NON di default) sono ancora lì. Se vuoi, le rimuovo per lasciare la
produzione esattamente com'era.
