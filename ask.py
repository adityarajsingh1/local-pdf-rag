#!/usr/bin/env python3
"""
Ask a question about the PDFs you indexed with ingest.py.

    python ask.py "what does the warranty cover?"
    python ask.py -k 8 --show-chunks "summarise chapter 2"
"""

import argparse
import sys

from localrag.ollama import Ollama, OllamaError
from localrag.qa import answer
from localrag.store import Store


def main():
    parser = argparse.ArgumentParser(description="Ask your PDFs a question, answered by a local model.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to give the model (default 5)")
    parser.add_argument("--show-chunks", action="store_true", help="print the chunks the answer was based on")
    args = parser.parse_args()

    store = Store()
    if store.count() == 0:
        sys.exit("nothing indexed yet. put PDFs in docs/ and run `python ingest.py` first")

    try:
        reply, cited, hits = answer(args.question, store, Ollama(), k=args.k)
    except OllamaError as e:
        sys.exit(f"error: {e}")

    print(reply.strip())
    print("\nSources:")
    for file, page in cited:
        print(f"  - {file}, page {page}")
    if args.show_chunks:
        for chunk, dist in hits:
            print(f"\n--- {chunk.file} p.{chunk.page} (distance {dist:.3f}) ---\n{chunk.text}")


if __name__ == "__main__":
    main()
