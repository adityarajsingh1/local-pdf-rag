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

Tests: `python -m pytest tests` (they fake Ollama, so you don't need it running)

Next up: the vector store and then actually asking questions.
