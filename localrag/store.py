"""
The vector store. chromadb keeps everything in a folder on disk (chroma_db/
by default), so you only pay for embedding a PDF once.

We always hand chroma our own vectors from Ollama. Its built-in embedding
function would download a model from the internet, which is exactly what
this project tries to avoid.
"""

import re

import chromadb
from chromadb.config import Settings

from localrag import config
from localrag.chunks import Chunk


def collection_name(embed_model):
    # vectors from different models can't be mixed, so each model gets its own collection
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", embed_model).strip("-")
    return f"pdfs-{name}"[:60]


class Store:
    def __init__(self, path=None, embed_model=None):
        self.client = chromadb.PersistentClient(
            path=path or config.DB_PATH, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            collection_name(embed_model or config.EMBED_MODEL),
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self):
        return self.collection.count()

    def files(self):
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["file"] for m in metas})

    def remove_file(self, name):
        self.collection.delete(where={"file": name})

    def indexed(self):
        """{file name: (fingerprint, folder it came from)} for everything in the index."""
        out = {}
        for m in self.collection.get(include=["metadatas"])["metadatas"]:
            out[m["file"]] = (m.get("fingerprint", ""), m.get("folder", ""))
        return out

    def add(self, chunks, vectors, fingerprint="", folder=""):
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"file": c.file, "page": c.page, "index": c.index,
                        "fingerprint": fingerprint, "folder": folder} for c in chunks],
        )

    def search(self, vector, k=5):
        """The k closest chunks as (chunk, distance). Lower distance = more similar."""
        if self.count() == 0:
            return []
        res = self.collection.query(query_embeddings=[vector], n_results=min(k, self.count()),
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append((Chunk(meta["file"], meta["page"], meta["index"], text), dist))
        return hits
