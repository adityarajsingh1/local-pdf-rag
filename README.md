# local-pdf-rag

Chat with your PDFs without sending them anywhere. Everything runs on your own machine: [Ollama](https://ollama.com) makes the embeddings and writes the answers, chromadb keeps the vectors, pypdf reads the files. Every answer cites the file and page it came from, so you can check it.

I already have [chat-with-pdfs](https://github.com/adityarajsingh1/chat-with-pdfs), which uses OpenAI for embeddings. This one is the fully local version, for documents I don't want to upload anywhere, and it keeps working offline.

## Setup

You need Python 3.10+ and Ollama.

```bash
pip install -r requirements.txt

ollama serve                 # if it isn't already running
ollama pull nomic-embed-text # embeddings
ollama pull llama3.2         # answers
```

Put your PDFs in `docs/` (or any folder, see below).

## Usage

### 1. Index

```bash
python ingest.py           # everything in docs/
python ingest.py ~/papers  # or any other folder
python ingest.py --force   # re-embed everything, even unchanged files
python ingest.py --reset   # start the index from scratch
```

It reads each PDF page by page, splits the text into overlapping chunks, embeds every chunk with Ollama and stores it in chromadb (in `chroma_db/`, so it survives restarts).

Running it again is cheap. Each chunk remembers a hash of its PDF and the chunk settings, so:

- files that didn't change are skipped,
- edited files are re-embedded,
- PDFs you deleted from the folder are dropped from the index.

### 2. Ask from the terminal

```bash
python ask.py "what does the warranty cover?"          # one question
python ask.py                                           # keep asking until /quit
python ask.py -k 8 --show-chunks "summarise chapter 2"
```

Answers stream in as the model writes them, so a slow local model doesn't look frozen. Under each answer it lists the pages it used, and `--show-chunks` prints the actual text behind the answer.

The interactive mode has a few commands:

- `/k 8` changes how many chunks go to the model.
- `/chunks` shows or hides the text behind each answer.
- `/files` lists the indexed PDFs.
- `/quit` exits.

### 3. Or in the browser

```bash
streamlit run app.py
```

Upload PDFs in the sidebar and press **index**. Only new or changed files get embedded. Then chat: answers stream in, the source pages are listed under each answer and you can open the exact chunks that were used. It's the same index as the command line tools, so anything indexed with `ingest.py` is already there.

## How it works

```
PDFs ──pypdf──> pages ──> chunks (1000 chars, 200 overlap, never across pages)
                              │
                     Ollama /api/embed
                              │
                          chromadb  (one collection per embedding model)

question ──embed──> 5 closest chunks ──> prompt with [file p.N] labels ──> Ollama /api/chat ──> answer + sources
```

A few choices I made along the way:

- **Chunks never cross a page.** That way every chunk has exactly one page number, and the citations stay exact. Chunks break at spaces, so words aren't cut in half.
- **The prompt is strict.** The model gets only the retrieved excerpts, each labelled like `[manual.pdf p.12]`. It's told to answer from those only, to cite them, and to say so when the answer isn't there instead of guessing. Small local models don't always follow this perfectly, which is why the sources and `--show-chunks` are always there to check against.
- **One collection per embedding model.** Vectors from different models can't be compared, so switching `EMBED_MODEL` starts a fresh collection instead of mixing them.
- **New embeddings first, old chunks removed after.** If Ollama dies halfway through re-indexing a file, the old version is still searchable.
- **Errors you can act on.** If Ollama isn't running you get "Start it with `ollama serve`". If a model is missing you get the exact `ollama pull` command.

## Settings

Everything comes from environment variables or a `.env` file (copy `.env.example`):

| variable | default | |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | where Ollama runs |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `CHAT_MODEL` | `llama3.2` | model that writes the answers |
| `DOCS_DIR` | `docs` | where the web app saves uploads |
| `DB_PATH` | `chroma_db` | where chromadb keeps its files |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | in characters |

Changing the chunk settings changes every file's fingerprint, so the next `ingest.py` re-embeds everything.

## Project layout

```
localrag/
  pdfs.py     read PDFs page by page, clean up the text
  chunks.py   overlapping chunks that keep their page number
  ollama.py   tiny client for /api/embed and /api/chat (plain requests)
  store.py    chromadb wrapper: add, search, per-file fingerprints
  qa.py       retrieval, the prompt, answers with sources
  config.py   settings from env vars / .env
ingest.py     index a folder
ask.py        terminal Q&A
app.py        Streamlit app
tests/
```

## Tests

```bash
python -m pytest tests
```

You don't need Ollama for the tests. The unit tests use a fake client with bag-of-words embeddings. `tests/test_pipeline.py` runs the real `ingest.py` and `ask.py` on generated PDFs against a small fake Ollama HTTP server, so the whole path from PDF to cited answer is covered, plus the "Ollama isn't running" and "model not pulled" errors. The Streamlit app is tested with `AppTest`.

## Ideas for later

- Scanned PDFs: there's no OCR, so pages that are just images come out empty.
- Hybrid search: mix keyword search (BM25) with the embeddings. Pure vector search sometimes misses exact terms like part numbers.
- Reranking the retrieved chunks before they go to the model.
- Chat history, so follow-up questions like "and what about chapter 3?" work.
- Highlighting the cited passage in the PDF itself in the web app.
- An evaluation set of questions with known answers, to compare chunk sizes and models properly instead of by feel.
