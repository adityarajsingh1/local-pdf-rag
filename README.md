# local-pdf-rag

Chat with your PDFs without sending them anywhere. Everything runs on your own machine: [Ollama](https://ollama.com) for the embeddings and the answers, chromadb for the vector store.

I already have [chat-with-pdfs](https://github.com/adityarajsingh1/chat-with-pdfs), which uses OpenAI for embeddings. This one is the fully local version, for documents I don't want to upload anywhere and so it works offline.

**Work in progress.** I'm building it step by step. Right now it can read PDFs page by page and split them into overlapping chunks (each chunk stays on one page, so answers can cite the page later):

```bash
pip install -r requirements.txt
mkdir docs   # put your PDFs here
python -m localrag.pdfs docs/
```

Tests: `python -m pytest tests`

Next up: embeddings through Ollama, the vector store and then actually asking questions.
