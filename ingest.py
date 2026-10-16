#!/usr/bin/env python3
"""
Index a folder of PDFs so you can ask questions about them.

    python ingest.py            # everything in docs/
    python ingest.py ~/papers
    python ingest.py --reset    # throw the index away and start over
"""

import argparse
import sys
import time

from localrag import config
from localrag.chunks import chunk_pages
from localrag.ollama import Ollama, OllamaError
from localrag.pdfs import find_pdfs, load_pdf
from localrag.store import Store


def index_folder(folder, store, client, size=None, overlap=None):
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    total = 0
    for path in find_pdfs(folder):
        try:
            pages = load_pdf(path)
        except Exception as e:  # pypdf raises all sorts of things on broken files
            print(f"  skipping {path.name}: {e}")
            continue
        if not pages:
            print(f"  skipping {path.name}: no text (maybe a scanned PDF?)")
            continue
        chunks = chunk_pages(pages, size, overlap)
        start = time.time()
        vectors = client.embed([c.text for c in chunks])
        # replace whatever we had for this file, so re-running doesn't leave old chunks around
        store.remove_file(path.name)
        store.add(chunks, vectors)
        total += len(chunks)
        print(f"  {path.name}: {len(pages)} pages, {len(chunks)} chunks ({time.time() - start:.1f}s)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into the local vector store.")
    parser.add_argument("folder", nargs="?", default="docs")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    if not find_pdfs(args.folder):
        sys.exit(f"no PDFs found in {args.folder}/")

    store = Store()
    if args.reset:
        store.client.delete_collection(store.collection.name)
        store = Store()

    client = Ollama()
    print(f"indexing {args.folder}/ with {client.embed_model}")
    try:
        n = index_folder(args.folder, store, client)
    except OllamaError as e:
        sys.exit(f"error: {e}")
    print(f"done, {n} chunks added. the index now has {store.count()} chunks from {len(store.files())} files")


if __name__ == "__main__":
    main()
