# Newsletter — Classificazione e composizione articoli

Pipeline per raccogliere articoli da feed RSS, giudicarne la pertinenza per un Osservatorio
del Politecnico di Milano, e proporre la selezione per il numero successivo della newsletter.

Tre pipeline attive, una per osservatorio — **OCX** (`ocx_relevance/`), **AI**
(`ai_relevance/`), **Startup Thinking** (`startup_thinking_relevance/`) — concettualmente
identiche, isolate da un sistema di produzione precedente (`src/` + `streamlit_app/`, vedi
[`src/README.md`](src/README.md)) che non è più il percorso in uso.

**Per capire un singolo stadio nel dettaglio** (algoritmo esatto, parametri, cosa è
persistito, punti fragili, cosa è cambiato di recente e perché): leggi
[`revisione/documentazione_tecnica.md`](revisione/documentazione_tecnica.md). Questo README è
solo la mappa d'insieme.

## Flusso (identico nelle 3 pipeline, in 9 step numerati)

```
corpus condiviso (corpus/corpus_raw.jsonl, da corpus/fetch_corpus.py)
  │
  ├─ 01  candidati negativi probabili, giudicati da un LLM come punto di partenza
  │      (solo AI/Startup Thinking — OCX riusa etichette editoriali già esistenti)
  ├─ 02  dataset etichettato (testo intero scaricato dagli URL, fallback al sommario)
  ├─ 03  embedding del dataset (text-embedding-3-large, Azure)
  ├─ 04  valutazione del ranker (k-fold ripetuto) + salva il set di riferimento
  ├─ 05  gate LLM di pertinenza (gpt-5.4) — non usa le etichette, giudica autonomamente
  ├─ 06  pipeline di produzione sul corpus condiviso:
  │      embedding → ranker per similarità → deduplicazione → gate LLM → tagging
  │      (content_type/target_audience) sui primi N candidati → ranking.csv
  ├─ 07  esporta il ranking in Excel
  ├─ 08  costruisce il foglio di revisione umana
  └─ 09  compositore: seleziona gli articoli per il numero (quote per testata/gruppo/
         content_type) → composizione.xlsx, due fogli (Selezionati/Esclusi)
```

Il ranker non addestra nulla: ogni candidato prende punteggio dalla similarità coseno con i 3
esempi "Sì" e "No" più vicini nel set di riferimento (stesso impianto per le 3 pipeline,
nessuna separazione per lingua — rimossa il 24/08, vedi documentazione tecnica). Il gate LLM è
una chiamata separata e indipendente che decide senza vedere le etichette.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Serve un file `.env` nella root con le credenziali Azure OpenAI:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

**I dati reali non sono in questo repository** (è pubblico — vedi `.gitignore`): corpus
scaricato, dataset di riferimento, export Excel, store delle selezioni storiche. Vanno
recuperati da chi gestisce il progetto, non generati da zero (molti richiedono fogli
editoriali storici non ricostruibili).

## Uso — una pipeline per volta

```bash
cd /path/al/repo

# costruisce/aggiorna il corpus condiviso (va fatto ogni tanto, non ad ogni run)
venv/bin/python corpus/fetch_corpus.py

# dentro ciascuna cartella pipeline, in ordine (esempio: ocx_relevance)
venv/bin/python ocx_relevance/02_build_dataset.py
venv/bin/python ocx_relevance/03_embed.py
venv/bin/python ocx_relevance/04_valuta.py
venv/bin/python ocx_relevance/06_rank.py --topn 150
venv/bin/python ocx_relevance/07_export_excel.py
venv/bin/python ocx_relevance/08_build_review_sheet.py
venv/bin/python ocx_relevance/09_compose.py
```

Dettagli, varianti e opzioni per singola pipeline: README dentro ciascuna cartella
(`ocx_relevance/README.md` e analoghi).

## Struttura del repository

```
corpus/           fetch feed RSS (condiviso dalle 3 pipeline) + tagging content_type/audience
ocx_relevance/    pipeline Osservatorio OCX
ai_relevance/     pipeline Osservatorio Artificial Intelligence
startup_thinking_relevance/   pipeline Osservatorio Startup Thinking
config/           feed, tassonomie, config del compositore, gruppi editoriali, modelli LLM
store/            (ignorato da git) stato persistente condiviso: articoli visti, selezioni
                  storiche, embedding cumulativi
revisione/        documentazione tecnica, audit, fogli di revisione umana (xlsx ignorati)
legacy/           script isolati non verificati come ancora in uso — vedi legacy/README.md
src/              sistema di produzione precedente, in gran parte superato — vedi src/README.md
streamlit_app/    interfaccia del sistema precedente
tests/            test automatici sulla logica pura (vedi sotto)
```

`ocx_relevance/common.py` e `ocx_relevance/ranker.py` sono condivisi dalle 3 pipeline (le
altre due li raggiungono con `sys.path`, non li duplicano). `corpus/tag_articles.py` idem per
la classificazione content_type/target_audience.

## Test

```bash
venv/bin/pytest tests/
```

Coprono solo la logica deterministica e pura — canonicalizzazione URL, normalizzazione testo,
rilevamento qualità/offuscamento del testo, punteggio del ranker, clustering per
deduplicazione, selezione del compositore. **Nessun test chiama un modello LLM o l'API di
embedding**: costerebbe denaro reale ad ogni run e non sarebbe deterministico. Il
comportamento del gate/tagging/embedding è verificato con esperimenti ad hoc documentati in
`revisione/`, non con una suite automatica.

## Modelli usati

| Uso | Modello | Dove configurato |
|---|---|---|
| Imbedding | `text-embedding-3-large` (3072 dim) | `ocx_relevance/common.py` |
| Gate di pertinenza | `gpt-5.4` | `config/models.json` (`gate_model`) |
| Tagging content_type/audience | `gpt-5.4-nano` | `config/models.json` (`tagging_model`) |

Prezzi e note di affidabilità (solo gpt-5.4 ha una tariffa confermata) in
`config/models.json`.
