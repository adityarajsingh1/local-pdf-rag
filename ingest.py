#!/usr/bin/env python3
"""
Index a folder of PDFs so you can ask questions about them.

    python ingest.py            # everything in docs/
    python ingest.py ~/papers
    python ingest.py --force    # re-embed everything, even files that didn't change
    python ingest.py --reset    # throw the index away and start over

Files that haven't changed since the last run are skipped, and PDFs you
deleted from the folder are removed from the index too.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from localrag import config
from localrag.chunks import chunk_pages
from localrag.ollama import Ollama, OllamaError
from localrag.pdfs import find_pdfs, load_pdf
from localrag.store import Store


def fingerprint(path, size, overlap):
    """Hash of the file plus the chunk settings. If either changes, the file gets re-indexed."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return f"{h}-{size}-{overlap}"


def index_folder(folder, store, client, size=None, overlap=None, force=False):
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    folder_key = str(Path(folder).resolve())
    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 0}
    known = store.indexed()
    seen = set()

    for path in find_pdfs(folder):
        seen.add(path.name)
        fp = fingerprint(path, size, overlap)
        if not force and path.name in known and known[path.name][0] == fp:
            stats["unchanged"] += 1
            continue
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
        # embed first, then swap. if Ollama fails halfway the old chunks are still there
        vectors = client.embed([c.text for c in chunks])
        store.remove_file(path.name)
        store.add(chunks, vectors, fingerprint=fp, folder=folder_key)
        stats["updated" if path.name in known else "added"] += 1
        stats["chunks"] += len(chunks)
        print(f"  {path.name}: {len(pages)} pages, {len(chunks)} chunks ({time.time() - start:.1f}s)")

    # PDFs that were indexed from this folder but aren't there anymore
    for name, (_, from_folder) in known.items():
        if from_folder == folder_key and name not in seen:
            store.remove_file(name)
            stats["removed"] += 1
            print(f"  removed {name} (not in the folder anymore)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into the local vector store.")
    parser.add_argument("folder", nargs="?", default="docs")
    parser.add_argument("--force", action="store_true", help="re-index every file, changed or not")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    if not Path(args.folder).is_dir():
        sys.exit(f"{args.folder}/ doesn't exist. make it and put some PDFs in it")
    if not find_pdfs(args.folder):
        # still worth running: PDFs that were deleted get dropped from the index
        print(f"no PDFs in {args.folder}/")

    store = Store()
    if args.reset:
        store.client.delete_collection(store.collection.name)
        store = Store()

    client = Ollama()
    print(f"indexing {args.folder}/ with {client.embed_model}")
    try:
        stats = index_folder(args.folder, store, client, force=args.force)
    except OllamaError as e:
        sys.exit(f"error: {e}")
    print(f"done: {stats['added']} new, {stats['updated']} updated, {stats['unchanged']} unchanged, "
          f"{stats['removed']} removed. the index has {store.count()} chunks from {len(store.files())} files")


if __name__ == "__main__":
    main()
