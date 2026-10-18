#!/usr/bin/env python3
"""
Ask questions about the PDFs you indexed with ingest.py.

    python ask.py "what does the warranty cover?"    # one question
    python ask.py                                     # keep asking, until you type /quit
    python ask.py -k 8 --show-chunks "summarise chapter 2"
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from localrag.ollama import Ollama, OllamaError
from localrag.qa import answer_stream
from localrag.store import Store

console = Console()

HELP = """[bold]commands[/bold]
  /k 8        use 8 chunks per answer
  /chunks     show or hide the chunks behind each answer
  /files      list the indexed PDFs
  /quit       exit (ctrl+c works too)"""


def ask_once(question, store, client, k, show_chunks):
    with console.status("searching your PDFs..."):
        pieces, cited, hits = answer_stream(question, store, client, k=k)
        first = next(pieces, "")  # wait for the model to start writing before the spinner goes
    # print the answer as the model writes it, so long answers don't feel stuck
    text = first
    # markup=False matters: rich would otherwise eat citations like [manual.pdf p.3] as style tags
    console.print(first, end="", style="bold", markup=False, highlight=False)
    for piece in pieces:
        text += piece
        console.print(piece, end="", style="bold", markup=False, highlight=False)
    console.print()

    if cited:
        pages = ", ".join(f"{f} p.{p}" for f, p in cited)
        console.print(Text(f"sources: {pages}", style="dim"))
    if show_chunks:
        for chunk, dist in hits:
            console.print(Panel(Text(chunk.text), title=f"{chunk.file} p.{chunk.page}",
                                subtitle=f"distance {dist:.3f}", border_style="dim"))
    return text


def repl(store, client, k, show_chunks):
    console.print(f"[bold]local-pdf-rag[/bold] · {store.count()} chunks from {len(store.files())} PDFs · "
                  f"answers by {client.chat_model}. type /help for commands\n")
    while True:
        try:
            q = console.input("[bold cyan]? [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nbye!")
            return
        if not q:
            continue
        if q in ("/quit", "/exit"):
            console.print("bye!")
            return
        if q == "/help":
            console.print(HELP)
        elif q == "/chunks":
            show_chunks = not show_chunks
            console.print(f"[dim]chunks {'on' if show_chunks else 'off'}[/dim]")
        elif q == "/files":
            for name in store.files():
                console.print(Text(f"  {name}"))
        elif q.startswith("/k"):
            try:
                k = max(1, int(q.split()[1]))
                console.print(f"[dim]using {k} chunks per answer[/dim]")
            except (IndexError, ValueError):
                console.print("[yellow]usage: /k 8[/yellow]")
        elif q.startswith("/"):
            console.print("[yellow]unknown command, try /help[/yellow]")
        else:
            try:
                ask_once(q, store, client, k, show_chunks)
            except OllamaError as e:
                console.print(f"[red]{e}[/red]")
            console.print()


def main():
    parser = argparse.ArgumentParser(description="Ask your PDFs questions, answered by a local model.")
    parser.add_argument("question", nargs="?", help="leave it out to keep asking questions")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to give the model (default 5)")
    parser.add_argument("--show-chunks", action="store_true", help="print the chunks the answer was based on")
    args = parser.parse_args()

    store = Store()
    if store.count() == 0:
        sys.exit("nothing indexed yet. put PDFs in docs/ and run `python ingest.py` first")
    client = Ollama()

    if args.question is None:
        repl(store, client, args.k, args.show_chunks)
        return
    try:
        ask_once(args.question, store, client, args.k, args.show_chunks)
    except OllamaError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
