# src/ — vecchia pipeline (superata)

Sistema di produzione precedente alle 3 pipeline per osservatorio (`ocx_relevance/`,
`ai_relevance/`, `startup_thinking_relevance/`). Classifica un articolo su **tutti** gli
osservatori del Politecnico (`config/osservatori.json`) con un unico passaggio a embedding,
senza gate LLM né compositore, e produce un solo Excel interattivo
(`streamlit_app/` ne è l'interfaccia). Strutturalmente completo e lanciabile — config e `.env`
necessari sono tutti presenti — ma non è più il sistema su cui si lavora.

**Due file di questa cartella sono invece dipendenze vive**, importate oggi dalle pipeline
attive:
- `fetch_feeds.py` — il fetch dei feed RSS, usato da `corpus/fetch_corpus.py`.
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
