# legacy/

Script isolati, non parte del flusso delle 3 pipeline attive
(`ocx_relevance/`, `ai_relevance/`, `startup_thinking_relevance/`). Spostati qui il 26/10/2026
in fase di riordino, non cancellati: non è stato verificato a fondo se qualcuno li lancia
ancora a mano, quindi il rischio minore era archiviarli, non eliminarli.

- **`score_articles.py`** — sperimenta pesi di scoring su un JSONL della vecchia pipeline
  (`src/`), non sulle pipeline attuali.
- **`build_review_file.py`** — costruisce un Excel di revisione a due fogli a partire da
  `data_dir`/`models_dir` passati da CLI; scritto prima che `07_export_excel.py`/
  `08_build_review_sheet.py` diventassero il percorso standard per questo.
- **`backfill_selected_from_editorial.py`** — backfill una tantum dello store
  (`store/articles.jsonl`) dai fogli editoriali storici (OCX26_Newsletter.xlsx,
  ExportNewsLetters.xlsx, SUT_Elenco news.xlsx). Operazione di popolamento iniziale, non
  pensata per girare di routine.
- **`llm_only_rank.py`** — confronto sperimentale ranker kNN vs LLM-only sulle 3 pipeline,
  scrive in `revisione/Confronto_kNN_vs_LLM.xlsx`. Contiene stime di costo (`PROD_COST`) usate
  in passato come riferimento di scala.
- **`classify_ct_llm_test.py`** — script manuale per testare `src/classify_ct_llm.py`
  (vecchia pipeline), non una suite automatica.

I path interni sono stati corretti per il nuovo livello di nidificazione (`ROOT` punta ancora
alla radice del repo), verificato solo a livello di sintassi (`py_compile`) e risoluzione path,
non eseguendo gli script end-to-end (richiedono rete/API).
