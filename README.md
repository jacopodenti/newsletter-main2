README.md
# Newsletter — Classificazione articoli e ranking interattivo

Pipeline per il fetch di articoli da feed RSS, classificazione automatica via **embedding Azure OpenAI** (osservatori, content type, target audience, macro topics) e generazione di un Excel interattivo per la revisione manuale.

## Output

A ogni run vengono prodotti in `output/`:

- `results_YYYYMMDD_HHMMSS.jsonl` — dataset grezzo con tutti i score e le probabilità per ogni articolo
- `ranking_interattivo.xlsx` — workbook a 4 fogli per esplorare e revisionare i risultati:
  - **Config** — pesi e moltiplicatori modificabili (lo Score si ricalcola in tempo reale)
  - **Ranking** — classifica ordinata con dropdown per revisione manuale di Rilevante / Content Type / Target Audience
  - **Dizionario** — glossario di tutte le variabili usate nei vari fogli
  - **Dati** — sorgente grezza (nascosta, alimenta le formule del Ranking)

## Setup

### Prerequisiti

- Python 3.11+
- Account Azure OpenAI con deployment `text-embedding-3-small` (o equivalente)

### Installazione

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Variabili d'ambiente

Crea un file `.env` nella root:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# opzionali — se l'embedding sta su un endpoint/deployment diverso:
AZURE_EMBEDDING_ENDPOINT=...
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_EMBEDDING_API_VERSION=2024-02-01
# soglia per la selezione dei macro topic (default 0.35):
TAXONOMY_MACRO_THRESHOLD=0.35
```

## Uso

### Run completo

```bash
python src/pipeline.py
```

Esegue: fetch feed → embedding articoli → classificazione → scoring → genera jsonl + Excel interattivo.

### Opzioni

```bash
python src/pipeline.py --max-per-feed 5             # limita articoli per feed (default 10)
python src/pipeline.py --scoring-config path.json   # override config scoring
```

### Workflow tipico

1. Lancia `python src/pipeline.py` (può durare diversi minuti per ~600 articoli, consuma token Azure embeddings).
2. Apri `output/ranking_interattivo.xlsx`.
3. Foglio **Config** → seleziona l'osservatorio target in `C2`, ritocca pesi `C5:C7` e moltiplicatori `C12:C26`.
4. Foglio **Ranking** → la classifica si riordina automaticamente. Per ogni riga:
   - `Rilevante` (Si/No) — marca l'articolo come effettivamente pertinente
   - `Content Type` / `Target Audience` — sovrascrivi via dropdown se la predizione è errata
5. I dati corretti possono poi essere usati come ground truth per ri-tarare i parametri.

## Struttura

```
config/
  feeds.yaml                       # lista feed RSS
  osservatori.json                 # definizioni e descrizioni osservatori
  classification_taxonomy.json     # label e descrizioni di CT, TA, macro topics
  scoring.json                     # pesi default per il ranking
cache/
  observatory_embeddings.pkl       # cache embedding osservatori
  taxonomy_embeddings.pkl          # cache embedding label tassonomia
src/
  pipeline.py                      # entrypoint
  fetch_feeds.py                   # parser RSS
  embed_classify.py                # classificazione via embedding Azure
  scoring.py                       # calcolo score finale
  export_excel.py                  # generazione workbook a 4 fogli
  enrich_osservatori.py            # utility scraping descrizioni osservatori
output/
  results_*.jsonl                  # dataset grezzi per timestamp
  ranking_interattivo.xlsx         # workbook revisione
old/
  ai/                              # vecchio stack LLM (classify.py + eval)
  models/                          # vecchio stack ML locale (NLI, sklearn, streamlit review app)
```

## Modello di classificazione

L'embed pipeline calcola, per ogni articolo:

- `score_obs` — similarità coseno tra l'embedding dell'articolo e quello di ogni osservatorio
- `score_ct` / `score_ta` — probabilità della classe più vicina nella tassonomia (zero-shot via embedding)
- `score_macro` — multilabel con soglia configurabile (`TAXONOMY_MACRO_THRESHOLD`)

Lo **Score finale** (foglio Ranking) è:

```
Score = w_obs * obs_norm + w_ct * (ct_score * ct_mul)_norm + w_ta * (ta_score * ta_mul)_norm
```

dove `obs_norm`/`ct_norm`/`ta_norm` sono normalizzazioni min-max sull'intero batch e i moltiplicatori per content type / target audience sono ritoccabili in `Config!C12:C26`.

## Cache

Gli embedding di osservatori e tassonomia sono persistiti in `cache/*.pkl` con hash della config: vengono ricalcolati solo se `osservatori.json` o `classification_taxonomy.json` cambiano. Per forzare il ricalcolo, cancella i file in `cache/`.

## Codice archiviato

`old/` contiene approcci precedenti non più usati ma mantenuti come riferimento:

- `old/ai/` — classificazione via prompt a un LLM Azure (`classify.py`), eval di accuracy, confronti LLM vs embed
- `old/models/` — zero-shot NLI con mDeBERTa (`classify_nli.py`), classificatori sklearn allenati su revisioni umane (`train_classifiers.py`, `retrain.py`), gate hybrid embed+local (`confidence.py`), app Streamlit per review queue e drift monitoring

Questi moduli hanno path interni a `src/` non più validi; per riattivarli serve aggiustare gli import.