"""
The actual question answering: find the closest chunks, put them in the
prompt with their file and page, and ask the local model to answer from
them only.
"""

SYSTEM = """You answer questions about the user's PDF documents.
Use only the excerpts you are given. If they don't contain the answer, say you couldn't find it in the documents, don't guess.
After each fact, cite where it came from like [report.pdf p.3], using the labels given with the excerpts.
Keep answers short and to the point."""


def label(chunk):
    return f"{chunk.file} p.{chunk.page}"


def retrieve(question, store, client, k=5):
    vector = client.embed([question])[0]
    return store.search(vector, k=k)


def build_messages(question, hits):
    excerpts = "\n\n".join(f"[{label(chunk)}]\n{chunk.text}" for chunk, _ in hits)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]


def sources(hits):
    """File + page of every chunk used, in order, without repeats."""
    seen = []
    for chunk, _ in hits:
        if (chunk.file, chunk.page) not in seen:
            seen.append((chunk.file, chunk.page))
    return seen


def answer(question, store, client, k=5):
    hits = retrieve(question, store, client, k)
    if not hits:
        return "The index is empty, run `python ingest.py` first.", [], []
    reply = client.chat(build_messages(question, hits))
    return reply, sources(hits), hits


def answer_stream(question, store, client, k=5):
    """Like answer(), but the reply comes back piece by piece. Returns (pieces, sources, hits)."""
    hits = retrieve(question, store, client, k)
    if not hits:
        return iter(["The index is empty, run `python ingest.py` first."]), [], []
    return client.chat_stream(build_messages(question, hits)), sources(hits), hits
