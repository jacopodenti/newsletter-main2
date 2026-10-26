# src/ — vecchia pipeline (superata)

Sistema di produzione precedente alle 3 pipeline per osservatorio (`ocx_relevance/`,
`ai_relevance/`, `startup_thinking_relevance/`). `pipeline.py` (entrypoint) classifica ogni
articolo su **tutti** gli osservatori del Politecnico insieme (`config/osservatori.json`,
~44 — multi-label, non un osservatorio scelto a monte) in una di 3 modalità: LLM puro
(default), ibrido con gate di confidenza che usa modelli locali a embedding e ricade sull'LLM
solo se incerto (`--gate`), o un confronto diagnostico LLM-vs-embedding (`--compare`). Nessun
gate di pertinenza Sì/No né compositore: produce un CSV/jsonl grezzo, e la formula a pesi del
vecchio README (`Score = w_obs*...`) è calcolata nell'Excel (`streamlit_app/` ne è
l'interfaccia), non in `pipeline.py`. Strutturalmente completo e lanciabile — config e `.env`
necessari sono tutti presenti — ma non è più il sistema su cui si lavora.

**Non condivide dati con le pipeline attuali**, nemmeno dove il codice sembra in comune:
`pipeline.py` legge `config/feeds.yaml` (5KB, fermo al 25/06) e scrive in `output/`, mai in
`corpus/corpus_raw.jsonl`. Le pipeline nuove leggono invece `corpus/feeds.yaml` (9KB, più
recente, con un commento esplicito in testa che dice di essere stato separato apposta da
`config/feeds.yaml` per non toccare la produzione) — le due liste sono già divergenti.

**Due file di questa cartella sono invece dipendenze di codice vive** (stesso modulo
Python, non stessi dati), importate oggi dalle pipeline attive:
- `fetch_feeds.py` — le funzioni di fetch RSS (`fetch_all`, usata solo da `pipeline.py`, e
  `fetch_all_with_diagnostics`, usata da `corpus/fetch_corpus.py` con `corpus/feeds.yaml`).
- `classify_ct_llm.py` — solo il dizionario `FEW_SHOT` (esempi per categoria), riusato da
  `corpus/tag_articles.py`. La funzione `classify_ct()` e il resto del file (client Azure
  proprio, prompt, CLI) **non** sono usati dalle pipeline attuali.

Il resto della cartella (`pipeline.py`, `classify.py`, `embed_classify.py`, `confidence.py`,
`train_classifiers.py`, `retrain.py`, `batch_classify_ct.py`, `compare_taxonomy.py`,
`reclassify_failed.py`, `rlhf_store.py`, `enrich_osservatori.py`, `evaluate_ct.py`) appartiene
al vecchio sistema e non è referenziato da `ocx_relevance/`, `ai_relevance/` o
`startup_thinking_relevance/`.

Non spostato né smontato in questa fase di riordino: romperebbe i due import live sopra se
fatto senza poter testare ogni percorso end-to-end. Se il vecchio sistema non serve più,
la mossa pulita è estrarre `fetch_feeds.py` e `FEW_SHOT` in un modulo condiviso e archiviare
il resto — non fatto qui per non introdurre un cambiamento non verificato prima del push.
