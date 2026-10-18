# local-pdf-rag

Chat with your PDFs without sending them anywhere. Everything runs on your own machine: [Ollama](https://ollama.com) for the embeddings and the answers, chromadb for the vector store.

I already have [chat-with-pdfs](https://github.com/adityarajsingh1/chat-with-pdfs), which uses OpenAI for embeddings. This one is the fully local version, for documents I don't want to upload anywhere and so it works offline.

**Work in progress.** I'm building it step by step. Right now it can read PDFs page by page and split them into overlapping chunks (each chunk stays on one page, so answers can cite the page later):

```bash
pip install -r requirements.txt
mkdir docs   # put your PDFs here
python -m localrag.pdfs docs/
```

There's also a small Ollama client now (embeddings and chat). To get Ollama ready:

```bash
ollama serve                 # if it isn't already running
ollama pull nomic-embed-text # for embeddings
ollama pull llama3.2         # for the answers
```

The models and the Ollama address can be changed with `EMBED_MODEL`, `CHAT_MODEL` and `OLLAMA_URL` (see `.env.example`). If Ollama isn't running or a model is missing, it tells you what to run instead of dumping a stack trace.

### Indexing

```bash
python ingest.py           # indexes everything in docs/
python ingest.py ~/papers  # or any other folder
python ingest.py --reset   # start the index from scratch
```

It reads each PDF, splits it into chunks, gets an embedding for every chunk from Ollama and stores them in chromadb (in `chroma_db/`, so it survives restarts). Re-running replaces a file's chunks instead of adding duplicates. Each embedding model gets its own collection, because vectors from different models can't be compared.

Tests: `python -m pytest tests` (they fake Ollama, so you don't need it running)

### Asking

```bash
python ask.py "what does the warranty cover?"          # one question
python ask.py                                           # keep asking until /quit
python ask.py -k 8 --show-chunks "summarise chapter 2"
```

In the interactive mode there are a few commands: `/k 8` to change how many chunks go to the model, `/chunks` to show the text behind each answer, `/files` to see what's indexed and `/quit`. Answers stream in as the model writes them, so a slow local model doesn't look frozen.

It embeds the question, pulls the 5 closest chunks (`-k` to change that) out of chromadb and sends them to the chat model, each labelled with its file and page. The prompt tells the model to answer only from those excerpts, to say so when the answer isn't there, and to cite like `[manual.pdf p.12]`. Under the answer it lists every page it used, and `--show-chunks` prints the actual text so you can check it.

Next up: only re-indexing PDFs that actually changed, so `ingest.py` gets fast on big folders.
