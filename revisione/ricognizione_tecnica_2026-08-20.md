# Ricognizione tecnica — gate, clustering, costo reale, tagging, materiale ignorato

Data: 2026-08-20. Solo lettura — nessun file modificato, nessuna chiamata a pagamento. Tutto
verificato leggendo i sorgenti attuali (post-commit di questa sessione), non ricostruito da
documentazione o nomi di file. Dove qualcosa contraddice un'analisi precedente, è segnalato
esplicitamente con **CORREZIONE**.

---

## 1. Il gate LLM — passo per passo (verificato su `startup_thinking_relevance/06_rank.py` +
`05_llm_gate.py`, identico nelle 3 pipeline salvo desc/criteria)

**Input**: `topn = [arts[i] for i in order[:args.topn]]` — i primi N (default 150) articoli per
`order` (l'ordine di ranking, calcolato al passo 3, PRIMA del clustering). Ogni articolo arriva a
`run_gate()` con: `title`, `full_text` (o `summary` in fallback), `fonte`, `link`,
`published_date`, `cluster_id`, `cluster_peers` (lista di `{"title", "excerpt"}` per gli altri
membri dello stesso cluster, esclusi per titolo — non per identità univoca).

**Costruzione del prompt — ordine esatto dei blocchi** (da `PROMPT` in `05_llm_gate.py`):
1. Riga di ruolo: `Sei un classificatore di pertinenza per l'Osservatorio "{name}"...`
2. `DESCRIZIONE OSSERVATORIO:\n{desc}` — statico, costante per pipeline
3. `COMPITO: ... PERTINENTE ...\n{criteria}` — statico, costante per pipeline
4. `Rispondi SOLO con JSON: {"rilevante":.., "confidence":.., "motivo":..}` — statico, istruzione di formato
5. `ARTICOLO:\nTitolo: {title}\nTesto: {body}` — **variabile**, `body` = `full_text` (o summary) troncato a 6000 caratteri
6. `{cluster_note}` — **variabile**, presente solo se l'articolo ha peer nel cluster
7. `JSON:` — statico, cue finale

I blocchi 1-4 sono identici per ogni chiamata della stessa pipeline (prefisso statico, 719 token
misurato per Startup Thinking — sotto la soglia di 1024 per il caching automatico di Azure, vedi
nota già aggiunta al codice il 20/08).

**Iniezione dei membri del cluster**: avviene nel blocco 6, DOPO il testo dell'articolo principale
e prima del cue finale `JSON:` — quindi è l'ultima cosa che il modello legge prima di rispondere.
Campi passati per ciascun peer: solo `title` e un estratto di 240 caratteri del suo `full_text`
(non il testo intero — tenuto corto apposta perché si somma al costo per ogni candidato con 2+
peer nello stesso cluster). Il testo esatto chiesto al modello (`CLUSTER_NOTE_TMPL`):
*"ATTENZIONE — possibile notizia duplicata... Se confermi che è la stessa notizia, dillo nel
motivo... Se invece tratta un aspetto/evento diverso pur sullo stesso tema, ignora questo
avviso."*

**Il modello può modificare il raggruppamento? No — verificato, non solo dedotto dal testo del
prompt.** Lo schema JSON di risposta ha solo 3 campi (`rilevante`, `confidence`, `motivo`): nessun
campo per correggere/confermare il cluster_id o riassegnare un articolo. L'unico modo in cui il
modello "agisce" sul cluster è scrivere un commento in linguaggio naturale dentro `motivo` (es.
"stessa notizia di: X") — un testo che il revisore umano legge nel foglio Excel, non un dato che
la pipeline consuma per modificare `cluster_id` o l'ordine. Il raggruppamento resta quello
calcolato prima e non cambia in base a cosa risponde il modello.

**Output e uso a valle**: la risposta parsata diventa `{"llm": 0/1, "llm_confidence":..,
"llm_motivo":..}`, unita a ogni riga di `topn`. In `06_rank.py`, tre mappe (`llm_map`,
`confidence_map`, `motivo_map`) indicizzate per `title` alimentano `ranking.csv`: colonne
`pertinenza_llm` (Si/No), `confidence_llm`, `parere_revisore`. **Nota su questo nome**: la colonna
si chiama "parere_revisore" ma il suo contenuto è il `motivo` scritto dal MODELLO, non da un
revisore umano — un'etichetta che potrebbe confondere chi legge il CSV pensando sia già una nota
umana. Solo i 150 articoli passati dal gate hanno queste 3 colonne compilate; il resto del corpus
(tutte le righe di `order`, scritte comunque nel CSV) le ha vuote.

---

## 2. Il clustering — dove, come, e se influenza il ranking

**Dove**: passo 4 di `06_rank.py`, DOPO l'embedding (passo 2) e DOPO il calcolo di `order` (passo
3, `order = np.argsort(-scores)`), PRIMA del gate (passo 5). Gira su TUTTO il corpus deduplicato,
non solo sui candidati che arriveranno al gate (commento esplicito nel codice: "costa niente in
più, stessi embedding del passo 2").

**Algoritmo e soglia**: `find_similar_clusters(X, return_similarity=True)` in `ranker.py` —
clustering hierarchical a complete-linkage, soglia coseno `CLUSTER_SIM_THRESHOLD = 0.72`.

**Cosa produce, concretamente — tre superfici distinte, verificate una per una**:
1. `cluster_id` per articolo (intero corpus) → colonna `cluster_id` in `ranking.csv`.
2. `cluster_metadata.json` (passo 4b) — per cluster: `n_articoli`, `n_testate`, `testate`,
   `sim_media/min/max`, `spread_ore`. Commento esplicito nel codice: "SOLO calcolo e persistenza,
   NESSUN effetto sul ranking".
3. Colore nel foglio Excel — in `07_export_excel.py` (non in `06_rank.py`): `cluster_fill()`
   assegna un colore di sfondo al titolo in base a `cluster_id % len(CLUSTER_COLORS)`. Pura
   presentazione.

**Il clustering influenza il ranking o la selezione? No — verificato direttamente nel codice, non
solo nei commenti.** `order` (l'ordine di ranking usato ovunque a valle, incluso quali articoli
arrivano al gate) è calcolato al passo 3, PRIMA che il clustering (passo 4) venga anche solo
eseguito. Nessuna riga di codice successiva modifica `order` o `scores`. Questo confema, con
lettura diretta del codice, la decisione già presa "niente peso automatico sui duplicati" — nessuna
contraddizione trovata.

---

## 3. Costo reale del gate — dai log citati come di produzione, con una riserva

I numeri vivono in un unico posto: `PROD_COST` dentro `llm_only_rank.py`, con questo commento
originale: *"letto dai log reali del 23-24/07, non stimato... non c'è un posto dove la pipeline di
produzione li salva da sola oggi."* **Riserva onesta**: ho cercato un log grezzo che li confermi
(cartelle `log/`, file `.txt`) e non esiste in questo checkout — l'unica fonte è questo dizionario
scritto a mano. Non posso verificarlo contro una fonte primaria, posso solo riportarlo come "l'unico
dato disponibile, dichiarato di produzione da chi l'ha scritto".

| Pipeline | Token input/articolo (gate, su 150) | Token output/articolo | Costo gate (150 art.) | Costo embedding corpus | Totale per run |
|---|---|---|---|---|---|
| OCX | 1540 | 70 | €0,735 | €0,163 | **€0,898** |
| AI | 1568 | 81 | €0,769 | €0,160 | **€0,930** |
| Startup Thinking | 1670 | 80 | €0,806 | €0,156 | **€0,963** |

Totale sulle 3 pipeline in un ciclo completo: **~€2,79**. Qualunque proiezione a 60 osservatori
parte da questi numeri (non verificabili oltre questo dizionario) moltiplicati per 20.

---

## 4. Tagging — inventario completo, eseguibilità verificata

Tutto il codice sotto elencato è in `src/` (pipeline generale multi-osservatorio), **isolato** da
ocx/ai/startup_thinking_relevance. L'unico punto di contatto REALE e verificato tra `src/` e le 3
pipeline è `src/fetch_feeds.py`, condiviso da `corpus/fetch_corpus.py` per il fetch RSS — nessun
collegamento di classificazione/tagging.

| File | Importa senza errori? | Eseguibile oggi così com'è? | Nota |
|---|---|---|---|
| `embed_classify.py` | Sì | Solo la parte osservatori (coseno) — CT/TA/macro_topics tornano `None`: nessun `.pkl` in `models/` | — |
| `confidence.py` | Sì | Sì, ma senza modelli locali ricade SEMPRE sull'LLM per ogni campo | orchestratore embed+classificatore+LLM |
| `train_classifiers.py` | Sì | **No**: richiede `--results output/results_*.csv`, `output/` è vuota in questo checkout | allena i modelli mancanti |
| `retrain.py` | Sì | **No**: richiede dati da `rlhf_store`, nessun file di store persistito trovato su disco | training incrementale |
| `rlhf_store.py` | Sì | Sì come libreria, ma store vuoto (0 esempi revisionati) | — |
| `classify.py` | Sì | Sì (LLM multi-campo: osservatori+CT+TA+macro_topics in un'unica chiamata) | richiede solo API key |
| `pipeline.py` | Sì | Sì — fetch→classifica→CSV piatto (confermato leggendo `main()`: nessuno step di scoring pesato o export Excel, a differenza di quanto descritto nel README, già segnalato il 19/08) | entrypoint |
| **`classify_ct_llm.py`** | Sì | Sì — **classificatore LLM few-shot dedicato SOLO al content_type**, non catalogato nelle ricognizioni precedenti | terza via per il CT, distinta sia da `embed_classify.py` che da `classify.py` |
| **`batch_classify_ct.py`** | Sì | Sì — wrapper CLI che applica `classify_ct_llm` a un file Excel (`Book-2.xlsx` di default, esiste) | — |
| **`evaluate_ct.py`** | Sì | Sì — produce un report Word di valutazione (**già eseguito**, vedi punto 5) | — |
| `compare_taxonomy.py` | Sì | **No**: richiede `output/results_*.csv`, assente | — |
| `reclassify_failed.py` | Sì | Dipende da un CSV con un formato specifico di errori pregressi, non verificato se presente | — |

**Tassonomia**: confermato ancora una volta (5 CT + 7 TA da `config/classification_taxonomy.json`)
allineata al Dizionario di `LABEL-Newsletter.xlsx` — nessuna novità rispetto al 19/08.

---

## 5. Cosa non era stato citato finora — e un risultato di valutazione già su disco

**Trovato un'evidenza reale, non richiesta ma già misurata, che va aggiunta al quadro**:

`Book-2_ct_eval.docx` (report Word, generato da `evaluate_ct.py`) — valutazione del
classificatore LLM few-shot per content_type (`classify_ct_llm.py`) su 50 articoli (lo stesso
sottoinsieme di `LABEL-Newsletter.xlsx`, ground truth identico: 24/5/3/4/14 per
analysis/case_study/regulation/research_report/vendor_announcement). Risultato:

- **Accuratezza globale: 84,0% (42/50)**
- Per label: regulation 100%, analysis 91,7%, case_study 80,0%, vendor_announcement 78,6%,
  research_report 50,0% (solo 4 casi, il più debole)
- Correlazione content_type × rilevanza OCX: regulation 0% "Sì", research_report 25%,
  vendor_announcement 50%, case_study 60%, analysis 62,5%

Questo è un dato concreto mai citato prima in nessuna delle analisi di questa sessione: esiste
già una misura reale di accuratezza per UN approccio di classificazione content_type (quello LLM
few-shot), diversa e più alta della semplice ipotesi "non_classificato pochi = tassonomia ok" usata
finora come proxy.

**Altri file non ancora citati, relativi a ranking/valutazione/confronto modelli**:
- `Book.xlsx` — stesse 12 colonne di `Book-2.xlsx` ma in ordine diverso; non verificato se
  contenuto identico o versione precedente.
- `20260608_news_ocx.xlsx` e `20260608_news_ocx-2.xlsx` — stessa struttura a 4 fogli di
  `LABEL-Newsletter.xlsx` (Config/Ranking/Dizionario/Dati), datati 8 giugno: sembrano snapshot
  precedenti dello stesso prototipo, non esaminati riga per riga.
- `results_20260608_130730.jsonl` (root, 1418 righe, 8MB) — stesso schema di
  `corpus/corpus_raw.jsonl` (title/link/published_date/feed_name/summary): uno snapshot del
  corpus condiviso datato 8 giugno, precedente al riferimento del 22 luglio.
- **`Fonti news (5).xlsx`** — foglio "Export", **3772 righe**, schema di export da un CMS
  editoriale (ID, Data inserimento, Data della news, Titolo, **Osservatorio**, Fonte, Link, Tag,
  Operatori). La colonna Osservatorio contiene voci come "Innovazione Digitale nel Retail" — **un
  osservatorio mai comparso in nessuna analisi precedente**, il che suggerisce che questo file
  copre una popolazione di osservatori e selezioni storiche più ampia dei 3 file usati per il
  46,8% di selezioni da fonti non tracciate (quelli erano solo OCX/AI/Startup). Non l'ho analizzato
  in profondità — lo segnalo perché è materiale rilevante per confronti storici mai preso in
  considerazione finora, e potrebbe cambiare quel numero se riletto includendo questo file.
- `ocx_relevance/prove_esperimenti/script/recall_newsletter.py` e `gpt_on_editorial.py` — citati
  nell'INDEX.md di quella cartella ma non letti riga per riga in questa sessione (solo i numeri
  di sintesi in INDEX.md sono stati usati finora).

**Nessuna contraddizione trovata nei punti 1 e 2** rispetto alle analisi precedenti — clustering e
gate si comportano esattamente come descritto il 19/08. La correzione principale è epistemica: il
costo "di produzione" del punto 3 non ha una fonte primaria verificabile in questo checkout, va
trattato come tale nelle proiezioni.
