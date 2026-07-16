# Prove degli esperimenti — OCX relevance

Log grezzi (cartella `log/`) e script (cartella `script/`) degli esperimenti fatti.
Ogni log è l'output **reale** dello script corrispondente, girato su dati reali.
Sono la traccia dei numeri riportati. Riproducibili rilanciando gli script.

⚠️ Nota: questi sono numeri di **PROTOTIPO** (embedding + ML + LLM), misurati su dati
pochi e in parte distorti. NON sono numeri del sistema di produzione (che è un prompt LLM
singolo). Vanno letti come esperimento, non come metrica certificata.

| Log | Script | Cosa dimostra (numeri chiave) |
|---|---|---|
| `gpt_eval_out.txt` | `eval_relevance_gpt.py` | GPT-5.4 sui 50 etichettati: **precisione 100%, recall 42%, 0 falsi positivi** |
| `gpt_all_out.txt` | `gpt_all_1418.py` | GPT-5.4 su tutti i 1418: **53 Si / 1365 No, 0 errori**; incrocio con kNN |
| `train_exp_out.txt` | `train_expanded.py` | Set espanso (159 esempi): **AUC kNN 0.904 / LogReg 0.905**, soglia 0.179 |
| `rescore_out.txt` | `rescore_expanded.py` | Ranking dei 1418 col modello espanso: soglia, top-15 (tutto CX in cima) |
| `gpt56_out.txt` | `gpt56_vs_54.py` | GPT-5.6-sol vs 5.4: **47 vs 53 Si, accordo 98.5%, stessa precisione 100%** |
| `fulltext_out.txt` | `fetch_fulltext.py` | Scraping testo intero: **1218/1418 (86%)**, mediana ~4000 car (vs ~150) |
| `embed_ft_out.txt` | `embed_fulltext.py` | Sommario vs testo intero (AUC sui 50): **0.894 → 0.910** |
| `reembed_out.txt` | `reembed_corpus_full.py` | Ri-embedding del corpus sul testo intero (1418 × 3072) |
| — | `recall_newsletter.py` | Recall sui 110 editoriali: 94% via coseno OCX |
| — | `gpt_on_editorial.py` | GPT sui 110 editoriali: severo 83% / morbido 97% d'accordo coi redattori |

## Riepilogo del percorso (AUC = quanto separa rilevante da non-rilevante)
- Vecchio score (coseno-descrizione): **0.48** (≈ caso)
- Embedding su descrizione: **0.75**
- Impara dagli esempi (50): **0.87**
- Impara dagli esempi editoriali (159): **0.90**
- Testo intero invece del sommario: **+0.016** (0.894 → 0.910 sui 50)
- Filtro LLM: **0 falsi positivi**, ~53/1418 candidati puliti
