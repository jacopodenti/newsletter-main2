# Documentazione tecnica — pipeline di classificazione e composizione newsletter

Verificata leggendo i sorgenti (codice + config + `git log`/`git show`), non i README — nel
repo i README si sono già rivelati sbagliati o superati più volte (vedi §11 e §12). Ogni
valore numerico riportato qui è citato con file e riga. Stato: commit `d411fbc` (25/08/2026).

## Panoramica

Tre pipeline paritetiche e isolate dalla produzione storica (`../src`, non importato da
nessun file qui sotto): `ocx_relevance/`, `ai_relevance/`, `startup_thinking_relevance/`.
Condividono un corpus unico (`corpus/corpus_raw.jsonl`, alimentato da `corpus/fetch_corpus.py`)
e tre moduli comuni:

- `ocx_relevance/common.py` — utility condivise (canonicalizzazione URL, fetch/pulizia testo,
  embedding, store persistente). Fisicamente vive solo in `ocx_relevance/`; `ai_relevance` e
  `startup_thinking_relevance` lo raggiungono con `sys.path.insert(0, str(ROOT/"ocx_relevance"))`.
- `ocx_relevance/ranker.py` — similarità e deduplicazione, stesso meccanismo di import.
- `corpus/tag_articles.py` — classificazione content_type/target_audience, importato allo
  stesso modo dai tre `06_rank.py`.

Ogni pipeline ha 9 script numerati (1→9): 2 costruiscono il dataset di riferimento
(negativi + dataset completo), poi embedding, valutazione, gate, la catena di produzione
(`06_rank.py`, che fa da solo ranking+gate+tagging+output), export/review Excel, e infine il
compositore. Le tre pipeline **non sono byte-identiche** come suggerirebbe il commento in cima
a `06_rank.py` ("concettualmente identiche") — Startup Thinking oggi scrive `voto_llm` in
`ranking.csv` invece di `confidence_llm` (cambiato oggi, commit `d411fbc`); il resto
(ranker, dedup, tagging) è davvero identico nei tre file.

---

## 1. Raccolta corpus — `corpus/fetch_corpus.py`

**Riceve**: nulla (interroga direttamente i feed). **Produce**: `corpus/corpus_raw.jsonl`
(un articolo per riga, condiviso dalle 3 pipeline) + `<out>_stats.json` + `<out>_feed_report.json`.

- **Fonti**: `corpus/feeds.yaml`, feed RSS/Atom. 87 voci totali, 7 marcate `disabilitato` →
  80 feed effettivamente interrogati (`src/fetch_feeds.py:17-28`, `load_feeds`).
- **Fetch**: `requests.get` con user-agent Chrome esplicito, `timeout=15`, **1 solo retry** su
  eccezione di rete (`MAX_RETRIES = 1`, `src/fetch_feeds.py:13-14`), poi `feedparser.parse`.
  **Sequenziale**, non parallelo — `fetch_all_with_diagnostics` itera i feed in un `for`
  semplice (`src/fetch_feeds.py:110-126`). Il `ThreadPoolExecutor(max_workers=16)` che si trova
  in `fetch_corpus.py` esiste solo nel percorso opzionale `--diagnose-fulltext`, non nel fetch
  di produzione.
- **Filtro d'età**: `MAX_AGE_DAYS = 30` (`corpus/fetch_corpus.py:57`), più `MAX_PER_FEED = 50`
  entry per feed (riga 56). Scarta se la data non è interpretabile, se `dt < cutoff`, **o se**
  `dt.year != now.year` (riga 189) — quest'ultimo vincolo non è menzionato nel docstring del
  modulo e a cavallo di fine anno può scartare articoli del 31/12 che sarebbero comunque entro
  i 30 giorni: comportamento più restrittivo di quanto documentato.
- **Deduplicazione**: solo per URL canonicalizzato (`corpus/fetch_corpus.py:147-157`), non per
  hash di contenuto. `canonicalize_url` (`common.py:150-161`): minuscolo dell'host, rimozione
  fragment e slash finale, rimozione dei parametri di tracking elencati in
  `corpus/tracking_params.yaml` (12 chiavi: `utm_*`, `traffic_source`, `at_medium`,
  `at_campaign`, `fbclid`, `gclid`, `mc_cid`, `mc_eid`). Non risolve redirect (nessuna
  richiesta di rete, dichiarato esplicitamente). La dedup per hash di contenuto esiste
  (`dedup_by_content_hash`, `common.py:164-204`) ma è usata solo a valle in `06_rank.py`, non qui.
- **Output**: ogni riga di `corpus_raw.jsonl` ha esattamente 5 campi — `title, link,
  published_date, summary, feed_name`. **Nessun `full_text`**: il testo integro non è mai
  scritto qui, solo il summary RSS.

---

## 2. Pulizia e recupero testo — `ocx_relevance/common.py` (condiviso)

**Riceve**: URL + summary RSS dal corpus. **Produce**: `full_text` + `text_source` per
articolo (`full` / `partial` / `fallback`), usati da tutte e 3 le pipeline dentro `06_rank.py`.

- **Fetch testo integro**: `fetch_fulltext(url, fallback)` (`common.py:472-491`), libreria
  **`trafilatura`** — `trafilatura.extract(r.text, include_comments=False,
  include_tables=False, favor_precision=True)`. `requests.get(timeout=15)`, stesso UA Chrome.
  Testo troncato a **8000 caratteri** se accettato (`body[:8000]`, riga 490).
- **Soglia di accettazione**: `if body and len(body) > len(fallback) + 50` (riga 489) — margine
  di **50 caratteri** sopra il summary RSS, altrimenti fallback. Dichiarata nel commento come
  euristica non calibrata su dati ("valore prudenziale, non testato sistematicamente").
- **Normalizzazione** (`normalize_text`, `common.py:134-140`): unescape HTML, rimozione tag via
  regex, **minuscolo**, collasso spazi multipli, trim. Usata per `content_hash`, per
  l'embedding e per il confronto di collisione testuale (§5).
- **Rilevamento lingua** (`detect_lang`, `common.py:63-75`): libreria `langdetect`, seed fisso
  `DetectorFactory.seed = 0` per determinismo, input troncato a 2000 caratteri. Testo
  vuoto/non rilevabile → fallback `"it"` (motivato come "lingua maggioritaria negli esempi di
  riferimento"). **Nota importante**: dopo la rimozione dei pool per lingua dal ranker (§4),
  questo calcolo resta usato solo per un log aggregato in `06_rank.py`, non incide più sul
  punteggio — vedi discrepanza in §12.
- **Cache full-text su disco**: **due file paralleli che oggi divergono**:
  - `<pipeline>/models/corpus_fulltext_cache.migrated.json` — chiave = URL canonicalizzato,
    scritto/letto da tutti e 3 i `06_rank.py` (fix del 24/08, commit `e533c8a`: la vecchia
    cache a chiave-titolo dava 0 hit su 1378 articoli in un run reale).
  - `<pipeline>/models/corpus_fulltext_cache.json` (senza suffisso) — ancora letto da
    `07_export_excel.py`, `08_build_review_sheet.py`, `build_review_file.py`, `llm_only_rank.py`.
  - **Nessun codice trovato che sincronizzi i due file** dopo il fix del 24/08: se si rilancia
    `06_rank.py`, solo `.migrated.json` si aggiorna — gli export possono mostrare testo/fonte
    diversi da quelli appena calcolati. Vedi §11.
  - Scrittura **non atomica** (`06_rank.py`, riscrive l'intero file con `json.dump` diretto) —
    a differenza dello store principale e dello store di embedding (§9), che usano
    tmp-file + `os.replace` proprio per questo motivo dal 24/08 (`d9cc691`).
- **Declassamento qualità testo** — tre controlli in sequenza, applicati **prima**
  dell'embedding (`06_rank.py`, tutte e 3 le pipeline), tutti descritti in dettaglio con date
  e casi reali in §10:
  1. `downgrade_known_bad_domains` — `config/known_bad_domains.json` (6 domini), declassa a
     `fallback` solo i domini `estensione:"sempre"` (theregister.com, ft.com, sifted.eu).
  2. `downgrade_colliding_fulltext` — soglia `TEXT_COLLISION_SIM_THRESHOLD = 0.6`
     (`common.py:338`), confronto a coppie per testata via `difflib.SequenceMatcher` su testo
     normalizzato; declassa entrambi gli articoli della coppia se il ratio è oltre soglia.
  3. `truncate_obfuscated_text` — soglia `soglia_testo_offuscato = 0.10`
     (`config/text_quality.json`), euristica language-agnostic su pattern di lettere
     impossibili; taglia al primo paragrafo pulito (`text_source="partial"`) invece di
     scartare, pensata per sifted.eu.

---

## 3. Embedding — `<pipeline>/03_embed.py` + `embed_texts` (`common.py`)

**Riceve**: `title + full_text` per riga di `dataset.jsonl` (dataset di riferimento/valutazione
— per il corpus di produzione lo stadio equivalente è `embed_texts_cached`, vedi §9).
**Produce**: `<pipeline>/models/dataset_vectors.npz` (`X`, `y`, `ids`).

- **Modello**: `EMBED_MODEL = "text-embedding-3-large"`, `EMBED_DIM = 3072`
  (`common.py:31-32`, unico punto di definizione), via **Azure OpenAI**
  (`_embed_client()`, `common.py:494-499`). `dimensions=3072` è imposto esplicitamente nella
  chiamata (`common.py:518`), non solo atteso.
- **Testo passato**: `f"{title}. {full_text}"`, poi dentro `embed_texts`: `normalize_text(t)`
  (minuscolo + pulizia HTML) **troncato a 6000 caratteri** (`max_chars=6000`, default della
  firma, `common.py:502,513`) — **non il testo intero grezzo**, contrariamente a quanto
  suggerisce il docstring dei tre `03_embed.py` ("embedda il testo intero"). Se il risultato è
  vuoto, sostituito dalla stringa `"vuoto"` per non mandare stringhe vuote all'API.
- **Batch/parallelismo**: batch da 8 testi (`batch=8`, `common.py:502,507`),
  `ThreadPoolExecutor(max_workers=4)` — deliberatamente moderato per non sforare i limiti
  Azure 429 (commento `common.py:504`).
- **Retry**: 6 tentativi totali, backoff `2^attempt` secondi, `except Exception:` generico (non
  distingue 429 da altri errori) — `common.py:515-523`. **Discrepanza**: il docstring dichiara
  "aspetta 1,2,4,8,16,32s, fino a 6 tentativi", ma il 6° tentativo (`attempt==5`) fallisce con
  `raise` immediato, senza sleep: gli sleep reali sono solo 5 (1,2,4,8,16s ≈ 31s totali), il
  "32s" citato non si verifica mai.
- **Persistito**: solo `X` (vettori), `y` (label), `ids` — **non** url né lingua, in
  `.npz` compresso, un file per pipeline, non condiviso.
- **Store cumulativo separato**: `store/embeddings.npz` alla radice (unico file fisico per le
  3 pipeline, non toccato da `03_embed.py` — vedi §9).

---

## 4. Ranker (similarità) — `ocx_relevance/ranker.py`

**Riceve**: embedding di un candidato + `reference_set.npz` (esempi Sì/No). **Produce**: un
punteggio scalare di similarità per candidato, usato per ordinare `ranking.csv`.

> **ATTENZIONE — cambiamento recente**: qualunque documentazione o audit precedente al 24/08
> che parli di "pool separati per lingua con fallback italiano sotto `MIN_PER_CLASS`" descrive
> un meccanismo **rimosso** (commit `5ea40df`). Vedi §10 e §12.

- **Formula esatta** (`score_similarity`, `ranker.py:39-48`): **non** è kNN classico (nessun
  voto/maggioranza). Coseno con top-k esempi per classe, mediati:
  ```
  score = media(coseno con i k "Sì" più vicini) − media(coseno con i k "No" più vicini)
  ```
  Vettori L2-normalizzati prima del prodotto scalare (`_l2`, righe 27-31); se una classe ha
  meno di k esempi disponibili, `_topk_mean` (righe 34-36) ne usa tutti quelli che ci sono,
  senza errore. Combinazione Sì/No: sottrazione diretta, nessun peso — non è una probabilità,
  può essere negativo.
- **k = 3**, non è una costante di modulo: hardcoded a ogni sito di chiamata (3× `06_rank.py`,
  3× `04_valuta.py`, 6 punti in totale) — rischio di disallineamento se qualcuno lo cambia in
  un solo file, anche se oggi sono coerenti.
- **Esempi di riferimento**: `reference_set.npz`, generato da `04_valuta.py` (STEP 4) a partire
  da `dataset_vectors.npz` (STEP 3) costruito da `02_build_dataset.py` (STEP 2). Provenienza
  per osservatorio (positivi/negativi):
  - **OCX**: 26 "Sì" + 24 "No" dal set-50 etichettato a mano, + 106 editoriali già pubblicati
    (positivi) + 3 editoriali scartati con nota redazionale esplicita (negativi) + 25 off-topic
    per parola-chiave dal corpus.
  - **AI**: 500 "editorial_ai" (tutti trattati come pertinenti per costruzione) + 100
    "llm_judged" **mai rivisti da un umano** (etichetta presa dal giudizio di un LLM).
  - **Startup Thinking**: 161 "editorial_startup" da 25 numeri storici + 100 "llm_judged",
    stessa riserva sulla non-revisione umana.
  - `06_rank.py` esclude dal ranking gli articoli già presenti in `dataset.jsonl` (match per
    URL canonicalizzato) per evitare l'auto-confronto a coseno ~1.
- **Valutazione** (`04_valuta.py`, tutte e 3 le pipeline): `RepeatedStratifiedKFold(n_splits=5,
  n_repeats=20, random_state=0)`, AUC out-of-fold media su 20 ripetizioni; precision@{10,20,30,50}
  su un singolo giro a 5 fold. Nessun training: "gli esempi SONO il modello" (docstring,
  confermato dal codice — nessun `.fit()` da nessuna parte).
- **Codice morto legato alla rimozione della lingua**: `04_valuta.py` calcola ancora `langs`
  con `detect_lang` e lo salva in `reference_set.npz` (riga 77 in ocx, equivalenti in ai/startup)
  — **nessun consumer lo legge più** (grep sull'intero repo: 0 occorrenze di `ref["langs"]`
  fuori da questo salvataggio). Stesso discorso per `arts_langs` in `06_rank.py`, calcolato
  solo per un `Counter` di log, mai passato a `score_similarity`.

---

## 5. Deduplicazione / clustering — `find_similar_clusters` (`ranker.py:51-100`)

**Riceve**: matrice di embedding di tutti i candidati di un run. **Produce**: un `cluster_id`
(o `None`) per articolo + una matrice di similarità riusata per i metadati cluster.

- **Algoritmo esatto**: linkage gerarchico agglomerativo **complete-linkage**
  (`scipy.cluster.hierarchy.linkage(condensed, method="complete")`, riga 86) — **non**
  union-find. Matrice di distanza `1 - coseno`, simmetrizzata esplicitamente
  (`(dist+dist.T)/2`, riga 81, per arrotondamenti float), diagonale a 0, clip a ≥0. Taglio
  dell'albero: `fcluster(Z, t=1-threshold, criterion="distance")`. Cluster con un solo membro
  scartati (restano `None`); id progressivo assegnato solo ai cluster con 2+ membri.
- **Soglia**: `CLUSTER_SIM_THRESHOLD = 0.72` (`ranker.py:24`), calibrata su un caso reale del
  22/07 (coppia confermata di articoli sullo stesso evento a coseno 0,744; coppie sullo stesso
  tema ma notizie diverse per lo più sotto 0,66). Filtro per **sospetti** ad alto richiamo, non
  precisione garantita — la decisione resta al gate/revisore umano.
- **Perché complete-linkage e non union-find**: commento esplicito (righe 55-60) — con
  union-find, su dati reali del 22/07, 44 articoli su 150 finivano incollati in un unico
  cluster per effetto catena (A~B, B~C ⇒ stesso gruppo anche se A e C non si somigliano).
  Complete-linkage impone che ogni coppia dentro un cluster superi la soglia.
- **Uso a valle**:
  - **Gate LLM**: `cluster_peers` (titolo + estratto di 240 caratteri per ogni altro membro
    dello stesso cluster) iniettati nel prompt via `CLUSTER_NOTE_TMPL` (§6) — il gate commenta
    nel `motivo` se conferma o smentisce che sia la stessa notizia, non decide un downgrade
    automatico.
  - **Metadati cluster** (`cluster_metadata.json`): `n_articoli, n_testate, testate, sim_media/
    min/max, spread_ore` — riusa la matrice già calcolata, nessun effetto sull'ordine.
  - **Fogli Excel**: colore per cluster (palette di 8 colori, `CLUSTER_COLORS`, ciclica) +
    numero esplicito `[cluster N]` nel titolo (aggiunto il 24/08, commit `7baeae7`, perché con
    153 cluster in un run reale il colore da solo si ripete e può ingannare — il numero è
    l'informazione affidabile, il colore resta un aiuto visivo).

---

## 6. Gate LLM — `<pipeline>/05_llm_gate.py`

**Riceve**: i primi N candidati per rank (`--topn`, default 150) con `title`, `full_text`/
`summary`, eventuali `cluster_peers`. **Produce**: `llm` (1/0), un secondo campo di punteggio
(`voto` 0-100 per Startup Thinking, `confidence` 0-1 per OCX/AI), `motivo`.

**Assemblaggio del prompt** (`PROMPT`, stesso ordine di blocchi nelle 3 pipeline):
1. Riga fissa di ruolo ("Sei un classificatore di pertinenza per l'Osservatorio ... del
   Politecnico di Milano").
2. `DESCRIZIONE OSSERVATORIO` — costante statica per osservatorio (`STARTUP_DESCRIPTION`,
   `OCX_DESCRIPTION`, ecc.), non cambia mai a runtime.
3. `COMPITO` + criteri RIENTRA/NON RIENTRA — costante statica per osservatorio
   (`STARTUP_CRITERIA` ecc.), separata dal resto del prompt apposta perché riusata da altri
   script (es. `llm_only_rank.py`) senza duplicarla.
4. Istruzione di formato JSON — statica, tranne il campo di punteggio: per Startup Thinking
   oggi è `"voto": 0-100 (quanto lo sceglieresti davvero per la prossima newsletter, non
   quanto parla del nostro tema)`; per OCX/AI è ancora `"confidence": 0.0-1.0` (non cambiato,
   vedi §10).
5. `ARTICOLO` — **variabile**: titolo + `body` (`full_text` o `summary`, troncato a **6000
   caratteri**, `body[:6000]`) + `cluster_note` (vuoto se nessun sospetto duplicato in questo run).
6. Marcatore `JSON:` finale.

**Chiamata**: `client.responses.create(model=LLM_MODEL, input=prompt)` — un articolo per
chiamata, `ThreadPoolExecutor(max_workers=8)`. `LLM_MODEL = "gpt-5.4"` per il gate in tutte e
3 le pipeline (confermato: `config/models.json` ha `gate_model: "gpt-5.4"`, e nessun'altra
chiamata usa un modello diverso per il gate).

**Parsing**: `resp.output_text.strip()`, se inizia con ```` ``` ```` rimuove il code-fence
(gestisce esplicitamente il caso in cui il modello avvolge il JSON in un blocco markdown),
poi `json.loads`.

**Se la chiamata fallisce**: retry con backoff esponenziale, **5 tentativi**
(`for attempt in range(5)`, `time.sleep(2**attempt)`), poi `{"llm": None, "voto"/"confidence":
None, "motivo": ""}` — nessuna eccezione propagata, il candidato resta nel run con campi vuoti.

**Persistito**: `gate_cost_log.jsonl` per pipeline (token reali in/out per chiamata, via
`log_gate_cost`). Il verdetto/voto finisce in `ranking.csv` (persistito) — i `cluster_peers`
passati nel prompt **non sono mai persistiti**, sono ricalcolati ad ogni run da
`find_similar_clusters` sul run corrente: non è possibile ricostruire il prompt esatto usato
in un run passato per rigiocarlo identico in un secondo momento (rilevante per qualunque test
retrospettivo, vedi commento nell'esperimento di §10 sul cambio voto).

---

## 7. Tagging (content_type / target_audience) — `corpus/tag_articles.py`

**Riceve**: `title` + `body` (`full_text`/`summary`, troncato a **6000 caratteri**,
`BODY_MAX_CHARS`) per **tutti** i candidati visti dal gate — Sì **e** No, non solo gli
approvati. **Produce**: `content_type` e `target_audience` per articolo.

**Perché su Sì e No**: dichiarato esplicitamente nel commento di `06_rank.py` (step "5b") —
il compositore a valle ripesca dai bocciati per riempire slot prima di allentare vincoli, e
per ripescare deve sapere che tipo di contenuto è. Chiamata **separata** dal gate: testato il
20/08 che nella stessa chiamata di pertinenza l'accordo con Filippo scendeva.

**Modello**: `tagging_model()` legge `config/models.json["tagging_model"] = "gpt-5.4-nano"` —
diverso dal gate. Scelto il 21/08 dopo test su 150 articoli reali: distribuzione content_type
simile a gpt-5.4 (differenza massima 9 punti), a 1/20 del costo.

**Categorie**:
- `content_type` (`config/classification_taxonomy.json`): `analysis, research_report,
  regulation, vendor_announcement, case_study`, + `non_classificato` (aggiunto solo nel
  prompt, non nel file di tassonomia).
- `target_audience`: `executive, marketing, sales, customer_care, it_data, cross_functional,
  research`, + `non_classificato`. Le descrizioni delle categorie sono scritte ad hoc in
  `AUDIENCE_DESCRIPTIONS` (`tag_articles.py:72-93`), non lette dal file di tassonomia (che dà
  solo la lista degli slug).

**Parsing**: JSON stretto, `json.loads(resp.output_text.strip())` — **a differenza del gate,
qui non c'è rimozione di eventuali code-fence** ```` ```json ````: se il modello li include, il
parsing fallisce.

**Se la chiamata o il parsing falliscono**: **nessun retry** — `classify_content_type`/
`classify_audience` non hanno un loro try/except, l'eccezione si propaga al chiamante
(`_tag()` in `06_rank.py`), che la cattura una sola volta e assegna `None` (non
`"non_classificato"`, che è invece una scelta esplicita del modello). È l'unica chiamata LLM
della pipeline **senza** retry/backoff (il gate ne ha 5, l'embedding 6). Nel log di riepilogo a
schermo, `None` e `"non_classificato"` vengono **conflati** nello stesso conteggio
(`ct in (None, "non_classificato")`), anche se restano distinguibili nel CSV (vuoto vs testo).

**Persistito**: `tag_cost_log.jsonl` per pipeline (token/costo reale per chiamata, per campo).
Da metà agosto anche nello store cumulativo (`store/articles.jsonl`, campi aggiunti in
`ea5ba57`, §10) — un `None` in un run non sovrascrive un tag già presente da un run precedente.

**Nota**: il docstring in cima a `tag_articles.py` (righe 35-37) dichiara ancora "non eseguito
ancora da nessuno step" — falso, è collegato in produzione da tutti e 3 i `06_rank.py` dal
21/08. Non aggiornato dopo il collegamento.

---

## 8. Selettore (compositore) — `<pipeline>/09_compose.py` + `select_for_newsletter` (`common.py`)

**Stato attuale**: esiste solo il compositore a quote, descritto qui sotto. Un selettore
alternativo basato su LLM (voto del gate anziché quote fisse) è stato pianificato in questa
stessa sessione ma **non ancora scritto** — non documentato oltre perché non esiste come codice.

**Riceve**: `ranking.csv` (righe con `pertinenza_llm` non vuota — le sole taggate) + lo store
(`store/articles.jsonl`) per sapere cosa è già uscito in newsletter. **Produce**:
`data/composizione.xlsx`, due fogli: "Selezionati" e "Esclusi".

**Algoritmo** (`select_for_newsletter`, `common.py:700` in poi), in 3 fasi rigide:
1. Riempimento con i candidati `pertinenza_llm=="Si"`, ordinati per `rank` (non per
   confidence/voto), sotto i vincoli base (`max_per_testata`, `max_per_gruppo_editoriale`,
   `max_per_content_type` da `config/composer_<osservatorio>.json`).
2. Ripescaggio dal pool dei "No", ordinato per `gate_confidence` **crescente** (il "No" meno
   sicuro/più borderline per primo), **senza allentare nessun vincolo**. Se `gate_confidence`
   manca, fallback sul `rank`.
3. Fino a `max_round_allentamento` round (default 2), +1 alla volta ai vincoli, nell'ordine
   `max_per_testata → max_per_gruppo_editoriale → max_per_content_type`.

Gli articoli già in store con `selected_for_newsletter=true` sono esclusi **prima** di
costruire i pool (mai allentato). Ogni escluso porta il `motivo_esclusione` esatto registrato
al momento in cui è stato scartato, non ricostruito a posteriori.

**Parametri attuali per osservatorio** (`config/composer_*.json`, derivati il 21/08 da storico
reale):

| | OCX | AI | Startup Thinking |
|---|---|---|---|
| n_articoli_totali | 10 | 9 (stimato, cadenza non misurabile) | 6 |
| max_per_testata | 2 | 2 (non misurato per AI, riusa default) | 2 |
| max_per_gruppo_editoriale | 3 | 3 | 3 |
| max_round_allentamento | 2 | 2 | 2 |

`max_per_content_type` diverso per osservatorio (quote arrotondate **con margine** sopra la
proporzione storica esatta, non congelate al valore osservato — dichiarato esplicitamente nei
`_nota` dei config).

`config/editorial_groups.json`: mappa testata→gruppo per **~19 testate su ~85** configurate in
`feeds.yaml` (4 gruppi noti: Foundry/ex-IDG, Digital360, Bloomberg L.P., Conde Nast). Una
testata non mappata vale come gruppo di sé stessa — il vincolo `max_per_gruppo_editoriale`
oggi agisce solo su queste ~19, non sulla concentrazione delle fonti in generale (limite
dichiarato nel `_nota` del file).

**Output**: `write_output()` (`09_compose.py`) — foglio "Selezionati" (Rank, Via — gate-si/
ripescato/roundN, Content type, Testata, Gruppo editoriale, Target audience informativo,
Titolo, Link) e "Esclusi" (stessa struttura + motivo). L'header A1 dichiara esplicitamente se
sono stati allentati vincoli o se il compositore è rimasto bloccato. Non aggiorna lo store: è
una proposta da rivedere, non un invio confermato.

---

## 9. Persistenza — quadro d'insieme

| Dato | File | Ambito | Scrittura atomica? |
|---|---|---|---|
| Corpus grezzo | `corpus/corpus_raw.jsonl` | condiviso | no (riscritto per intero) |
| Cache full-text (uso produzione) | `<pipeline>/models/corpus_fulltext_cache.migrated.json` | per pipeline | **no** |
| Cache full-text (uso export/review) | `<pipeline>/models/corpus_fulltext_cache.json` | per pipeline | **no**, e disallineata dalla precedente (§11) |
| Dataset di valutazione | `<pipeline>/data/dataset.jsonl` | per pipeline | statico, costruito una volta |
| Vettori dataset | `<pipeline>/models/dataset_vectors.npz` | per pipeline | n/d (rigenerato da zero) |
| Set di riferimento (ranker) | `<pipeline>/models/reference_set.npz` | per pipeline | n/d |
| Store articoli (metadati, selezioni storiche) | `store/articles.jsonl` | **condiviso tra le 3 pipeline** | **sì**, tmp+`os.replace` (dal 24/08, `d9cc691`) |
| Store embedding cumulativo | `store/embeddings.npz` | **condiviso** | **sì**, stesso fix |
| Output di un run | `<pipeline>/data/ranking.csv`, `composizione.xlsx`, fogli Excel | per pipeline | rigenerati per intero ad ogni run |
| Costo gate/tagging | `<pipeline>/data/gate_cost_log.jsonl`, `tag_cost_log.jsonl` | per pipeline | append, sotto lock |
| `cluster_peers` nel prompt del gate | — | **effimero**, mai persistito | ricalcolato ad ogni run |

Lo store condiviso (`store/articles.jsonl`) è la sola persistenza cross-pipeline oltre al
corpus grezzo: chiave `url_canonico` + `osservatorio`, campi `content_hash, published_date,
text_source, last_rank, selected_for_newsletter, first_seen`, estesi il 24/08 con
`content_type, target_audience` (`ea5ba57`) — un `None` in input non sovrascrive un valore già
presente, quindi un articolo non ripescato in un run non perde il tag preso in un run precedente.

Retention dello store embedding: **60 giorni** da `last_seen`, con eccezione per gli articoli
`selected_for_newsletter=true` (mai potati). Nota di correttezza esplicita nel codice:
l'embedding Azure non è deterministico bit-per-bit tra chiamate indipendenti (deriva coseno
0,999992-0,999999); il criterio dichiarato non è "ordine identico" ma "nessun attraversamento
di soglia operativa" — tasso osservato ~2% di scambi in una fascia di 101 articoli attorno al
rank 150.

---

## 10. Cosa è cambiato nell'ultima settimana

Tre filoni riconoscibili nei commit dell'ultima settimana, più il lavoro di questa sessione.

### Filone A — la catena tagging → compositore (21/08, un unico pomeriggio)

1. **`66c6de9`** — Prima: nessuna configurazione di composizione esisteva. Dopo: 4 file JSON
   nuovi (`editorial_groups.json`, `composer_ocx/ai/startup.json`) con vincoli derivati da
   storico reale (numeri di newsletter passati). Perché: formalizzare quote prima invisibili.
2. **`29e465b`** — Prima: nessuna funzione di selezione. Dopo: `select_for_newsletter()` in
   `common.py`, condivisa tra le 3 pipeline invece di triplicata. Verificata contro una
   simulazione manuale sui 150 articoli reali di Startup Thinking (stesso risultato).
3. **`67bac93`** — Prima: `ranking.csv` non aveva `content_type`/`target_audience`. Dopo: step
   "5b" in `06_rank.py`, tagging su **tutti** i candidati visti dal gate (Si e No), non solo
   gli approvati — "correzione rispetto al progetto iniziale", perché il ripescaggio del
   compositore pesca anche dai bocciati.
4. **`4b87aa6`** — Crea `09_compose.py` in tutte e 3 le pipeline, chiude la catena end-to-end.
5. **`adea4a4`** — Prima: il tagging usava lo stesso modello hardcoded del gate (`LLM_MODEL`
   importato da `common`). Dopo: `config/models.json` separa `gate_model`/`tagging_model`;
   gpt-5.4-nano promosso a default del tagging dopo test comparativo (differenza max 9 punti
   nella distribuzione content_type, 1/20 del costo). Il gate resta su gpt-5.4: nano e mini
   testati e scartati per quel ruolo (nano: recall 35,1% contro 81-87%; mini: kappa 0,384,
   sotto la soglia 0,45 fissata).
6. **`96315b1`** — Prima: il costo reale del tagging veniva scartato subito dopo la chiamata
   (stesso problema già risolto per il gate). Dopo: `_tokens_in`/`_tokens_out` nel risultato +
   `log_tag_cost()` → `tag_cost_log.jsonl`. "Trovato e corretto prima di eseguire il run reale
   end-to-end, non dopo" (messaggio di commit).

### Filone B — qualità del testo estratto (24-25/08, indagine con più iterazioni)

7. **`7baeae7`** — Prima: solo il colore (8 colori ciclici) segnalava i cluster sospetti nei
   fogli Excel. Dopo: numero `[cluster N]` esplicito nel titolo — un run reale con 153 cluster
   rendeva il colore ambiguo oltre l'ottavo.
8. **`e533c8a`** — Prima: `06_rank.py` leggeva la cache full-text a chiave-titolo, creata prima
   della migrazione a chiave-URL. Dopo: punta al file `.migrated.json`. Perché: la cache
   vecchia dava **0 hit su 1378 articoli** in un run reale (corpus riscaricato ogni volta).
9. **`d9cc691`** — Prima: `articles.jsonl` ed `embeddings.npz` scritti con `open(..., "w")`/
   `np.savez` diretti sul path finale — un crash a metà scrittura tronca il file. Dopo:
   scrittura su tmp-file + `os.replace` atomico. Perché: 1377 righe da run reali non
   ricostruibili in caso di perdita (le 773 storiche sì, con backfill).
10. **`ea5ba57`** — Prima: `content_type`/`target_audience` vivevano solo in `ranking.csv`,
    sovrascritto ad ogni run. Dopo: persistiti anche nello store condiviso.
11. **`2d32a94`** — Prima: nessun controllo incrociato tra articoli della stessa testata. Dopo:
    `downgrade_colliding_fulltext`, soglia 0.6. Perché: scoperto che theregister.com estraeva
    il widget "Most Popular" al posto dell'articolo — un cluster di 47 articoli scollegati
    (multe AliExpress, anniversario Viking 1, chip Nvidia...) letto inizialmente come "rubrica
    ricorrente" era in realtà questo bug. Stesso pattern trovato su ft.com (paywall), bebeez.it
    (widget "cerco soci"), cybersecitalia.it (banner cookie).
12. **`81ef281`** (solo doc) — Crea `revisione/audit_2026-08-22.md`, registra la correzione del
    punto 11 come finding #12, più un secondo cluster BeBeez confermato bug e un cluster Sifted
    con un problema **diverso e non ancora spiegato** (sarà il punto 13).
13. **`7cac581`** — Prima: nessun controllo per testo degradato **dentro** un singolo articolo
    (la collisione del punto 11 intercetta solo testo duplicato **tra** articoli). Dopo:
    `truncate_obfuscated_text`, soglia 0.10 su `_suspicious_frac` (frazione di token con
    pattern impossibili in italiano/inglese). Perché: sifted.eu rimappa i glifi con un font
    custom anti-scraping oltre un certo punto — testati tutti i 26 shift Caesar/ROT, non è un
    cifrario a chiave fissa. Testato su 14/14 articoli Sifted affetti e 5 Wired di controllo
    (0 falsi positivi).
14. **`b730663`** (solo doc) — Applicando lo stesso criterio del punto 11 al **reference set**
    di Startup Thinking, trova 13/238 esempi con testo pieno contaminati dallo stesso bug
    (3 theregister.com + 2 bebeez.it nel pool "No", 8 forbes.it nel pool "Sì"). Documenta per
    intero il caso BeBeez: 6 candidati finiti a rank 1369-1377/1377 per colpa di 2 esempi di
    riferimento "No" contaminati dallo stesso widget — corretti quei 2 esempi (zero nuove
    chiamate API), gli stessi 6 articoli risalgono a rank 458-755. Nello stesso commit,
    scoperta collegata: il pool inglese di Startup Thinking ha solo 3 esempi "Sì" contro ~70
    "No" — sotto qualunque soglia minima sensata.
15. **`5ea40df`** — Prima: `ranker.py` separava il confronto per lingua
    (`build_lang_pools`/`score_similarity_multilang`, `MIN_PER_CLASS=10`), con fallback
    italiano sotto soglia. Dopo: **rimossi**, confronto unico su tutto il reference set. Perché:
    diretta conseguenza del punto 14 — con solo 3 esempi inglesi, il pool inglese di Startup
    Thinking ricadeva **sempre** sull'italiano (il 70% del corpus è inglese): "un meccanismo
    che sembrava confrontare per lingua e in pratica non lo faceva mai". Motivazione tecnica
    aggiuntiva: `text-embedding-3-large` è multilingua per costruzione. Verificato via grep che
    gate/tagging/compositore non usassero queste funzioni prima di rimuoverle.
16. **`75526b5`** — Prima: solo controlli generici (collisione, offuscamento) intercettavano
    testo mal estratto. Dopo: `config/known_bad_domains.json`, 6 domini verificati a mano,
    declassamento incondizionato. Perché: due tentativi di rilevatore generico (misure
    sintattiche di superficie, coerenza per embedding tra prima/seconda metà del testo) avevano
    fallito — troppa sovrapposizione con articoli brevi ma genuini: "il problema è semantico,
    non sintattico".
17. **`dfa8ab2`** — Prima (dal punto 16): declassamento incondizionato su tutti e 6 i domini.
    Dopo: campo `estensione` (`sempre`/`parziale`) — solo 3 domini restano a declassamento
    totale. Perché: misurato che il danno collaterale costava più del beneficio — un articolo
    vero di cybersecitalia.it era **rank 1 su tutto il corpus AI** prima di essere declassato
    per il solo fatto di provenire da quel dominio (contaminazione reale: 6/45 su bebeez.it,
    2/20 su cybersecitalia.it, non l'intero dominio).
18. **`b600439`** (solo doc) — Registra il punto 17 come finding #14, incluso il limite
    accettato su forbes.it (8/8 contaminato ma la collisione ne intercetta solo 2/8, perché la
    lista "Trending" catturata cambia nel tempo — nessun fix ulteriore, documentato come noto).

### Filone C — questa sessione (25/08)

19. **`d411fbc`** — Prima: il gate di Startup Thinking chiedeva un campo `confidence` 0-1 (poi
    scoperto: non discriminava tra i "Sì" buoni e i "Sì" sbagliati, media 0,852 vs 0,830 anche
    col segno corretto). Dopo: campo `voto` 0-100 esplicito. Test di non regressione sui 143
    giudizi netti di Filippo: accordo/kappa dentro la banda di rumore storica, ma recall
    -13,5pt — dettagli completi nel commento sopra `run_gate()` in `05_llm_gate.py`. Propagato
    il rename a `06_rank.py`/`07_export_excel.py`/`08_build_review_sheet.py`/`09_compose.py`
    (quest'ultimo con inversione di segno nel ripescaggio, perché voto e confidence hanno
    verso opposto — vedi commento in `09_compose.py`).

---

## 11. Punti fragili — dove il sistema può degradare in silenzio

Elencati per gravità percepita (non per stadio), con il meccanismo esatto per cui l'errore
**non produce un'eccezione** ma un risultato sbagliato che passa inosservato.

1. **Due cache full-text disallineate, nessuna sincronizzazione**. `06_rank.py` scrive/legge
   `corpus_fulltext_cache.migrated.json`; `07_export_excel.py`, `08_build_review_sheet.py`,
   `build_review_file.py`, `llm_only_rank.py` leggono `corpus_fulltext_cache.json` (senza
   suffisso). Se si rilancia `06_rank.py` dopo un cambiamento di testo (es. un dominio
   aggiunto a `known_bad_domains.json`), solo il primo file si aggiorna: gli export possono
   mostrare per settimane un testo/fonte diverso da quello realmente usato per rank/gate/voto
   nell'ultimo run, senza che nulla segnali la divergenza. È lo stesso tipo di bug già
   diagnosticato una volta (0 hit su 1378, `e533c8a`) e riapparso all'altro capo della catena.
2. **Cache full-text non scritta atomicamente**. A differenza di `articles.jsonl` ed
   `embeddings.npz` (fix esplicito del 24/08), la cache full-text usa ancora `json.dump`
   diretto sul path finale: un'interruzione a metà scrittura la lascia troncata/corrotta, e il
   run successivo la leggerebbe silenziosamente parziale (JSON troncato → eccezione al
   caricamento, oppure — se il troncamento cade su un confine di record — una cache
   silenziosamente incompleta con alcune chiavi mancanti, trattate come "mai viste" e
   riscaricate, non come errore).
3. **Tagging senza retry, unica chiamata LLM della pipeline così**. Un singolo timeout o una
   risposta con code-fence markdown (che qui, a differenza del gate, non viene ripulita prima
   del parsing) fa fallire silenziosamente **quell'articolo solo**: `content_type`/
   `target_audience` diventano `None`, il compositore lo tratta come "non tra i vincoli di
   quota" (il controllo è `if ct != "non_classificato"`, quindi un `None` tecnico si comporta
   come categoria libera, non come "sconosciuto da trattare con cautela") — un fallimento di
   rete può quindi cambiare quali articoli riempiono gli slot, senza log di errore visibile a
   valle del run.
4. **`None` e `"non_classificato"` conflati nel log di riepilogo, ma non nei dati**. Il
   conteggio a schermo (`ct in (None, "non_classificato")`) non permette di distinguere "il
   modello ha detto esplicitamente che non sa classificarlo" da "la chiamata è fallita" — chi
   legge solo il log a schermo non può notare un aumento di errori tecnici mascherato da un
   aumento di incertezza editoriale.
5. **Ripescaggio del compositore silenziosamente meno preciso senza `gate_confidence`/voto**.
   `select_for_newsletter` ricade sul `rank` come proxy se il campo manca (`c["rank"]`) — non
   è un errore, ma un run in cui il campo è vuoto per un problema a monte (es. i due punti
   sopra) sposta silenziosamente l'ordine di ripescaggio dal "No meno sicuro" al "rank più
   alto", cambiando quali articoli bocciati vengono recuperati, senza che il foglio Excel di
   output lo segnali in alcun modo (il campo "Via" dice solo "ripescato", non su che base).
6. **k=3 duplicato in 6 punti** (non un bug oggi, ma un rischio strutturale): cambiare il
   numero di vicini del ranker richiede modificare 3× `06_rank.py` + 3× `04_valuta.py`
   coerentemente; un aggiornamento parziale renderebbe silenziosamente la valutazione (§4) non
   rappresentativa del comportamento reale in produzione, senza nessun controllo che lo segnali.
7. **`langs` calcolato e persistito ma morto**. Non causa errori, ma chiunque legga
   `reference_set.npz` in futuro e trovi il campo `langs` presuppone ragionevolmente che sia
   ancora usato dal ranker — non lo è da 5ea40df in poi. Lo stesso vale per `arts_langs` in
   `06_rank.py`: costa un `detect_lang` per articolo (tempo CPU) per un log che non influenza
   nulla a valle.
8. **forbes.it: limite di rilevamento noto e accettato, non un bug ma un buco reale**. Solo
   2/8 pagine contaminate note vengono intercettate dal controllo di collisione, perché la
   lista "Trending" catturata cambia nel tempo — un articolo forbes.it genuinamente
   contaminato ma non intercettato entra nel ranking/gate con testo sbagliato, senza alcun
   segnale visibile (a differenza dei domini "sempre", qui non c'è declassamento automatico).
9. **`cluster_peers` mai persistito**: qualunque tentativo futuro di rigiocare un prompt del
   gate esattamente come fu eseguito in un run passato (per debug o per un test A/B
   retrospettivo) non può includere la nota sui duplicati sospetti — l'esperimento di questa
   sessione (§10, punto 19) ha dovuto usare `cluster_note=""` per entrambe le varianti,
   disclosurato esplicitamente come semplificazione, non come comportamento di produzione.
10. **Stampa/scrittura non atomica del corpus grezzo**: `corpus_raw.jsonl` è riscritto per
    intero da `fetch_corpus.py` senza tmp-file — un'interruzione a metà lascia un jsonl
    troncato, che le pipeline a valle leggerebbero riga per riga fino all'errore di parsing
    sull'ultima riga incompleta (fallimento rumoroso qui, non silenzioso — l'unico della lista
    che si autodenuncia).

---

## 12. Contraddizioni con analisi precedenti — riepilogo esplicito

- **Pool per lingua nel ranker**: qualunque nota, audit o spiegazione precedente al 24/08 che
  descriva "pool separati per lingua, fallback italiano sotto `MIN_PER_CLASS=10`" descrive un
  meccanismo **rimosso**. In particolare `revisione/audit_2026-08-22.md` (righe 169, 184, 216)
  cita `MIN_PER_CLASS=10` come vivo: è **superato** dai commit del 24-25/08, non va usato come
  fonte per lo stato attuale del ranker (resta valido invece come registro storico dei finding
  che hanno *portato* a quella rimozione — BeBeez, cybersecitalia.it, ecc., riportati per intero
  in §10).
- **"Le tre pipeline `06_rank.py` sono concettualmente identiche"** (commento in cima al file):
  vero per ranker/dedup/tagging, ma da oggi (`d411fbc`) Startup Thinking scrive `voto_llm`
  mentre OCX/AI scrivono `confidence_llm` — un diff riga per riga dei tre file non è più a
  costo zero, va tenuto a mente citando quel commento.
- **`corpus/tag_articles.py` "non ancora eseguito da nessuno step"** (docstring in cima al
  file): falso da fine agosto, collegato in produzione in tutti e 3 i `06_rank.py`.
- **La confidence del gate come segnale di pertinenza**: l'audit del 22/08 elencava già tra i
  suoi finding "la confidence del gate non è mai stata validata come segnale" — questa sessione
  lo ha misurato direttamente: confidence grezza dà AUC 0,404 (peggio del caso) contro
  l'etichetta umana per un problema di segno (una "No" sicura ha confidence alta quanto un "Sì"
  sicuro); con il segno corretto sale a 0,81 ma non separa i "Sì" buoni dai "Sì" mediocri
  (0,852 vs 0,830). Il voto introdotto oggi risolve la seconda parte (separazione reale, 70,2
  vs 65,2) ma non è indenne da effetti collaterali: la stessa riformulazione che rende il voto
  utile per ordinare ha spostato anche il criterio di sì/no, con un calo di recall di 13,5
  punti su 143 giudizi netti — non c'è oggi una banda di rumore storica per precision/recall
  (solo per accordo/kappa) con cui giudicare se quello spostamento sia rumore o un effetto
  reale della formulazione.
- **Nessuna contraddizione trovata**, invece, tra questo documento e la metodologia di
  valutazione del ranker (`RepeatedStratifiedKFold`, 5×20, k=3): confermata leggendo
  `04_valuta.py` per intero in tutte e 3 le pipeline, invariata dal refactor del 24/08 a parte
  la rimozione dei pool per lingua a monte del calcolo.
