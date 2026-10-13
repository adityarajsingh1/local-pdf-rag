"""
Reading PDFs. Every page is kept separately so answers can later say
which page something came from.

    python -m localrag.pdfs docs/     # quick check of what got extracted
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class Page:
    file: str     # just the file name, that's what shows up in citations
    number: int   # 1-based, like a PDF viewer shows it
    text: str


def clean(text):
    # pypdf leaves lots of odd whitespace and hyphenated line breaks
    text = text.replace("-\n", "")
    return " ".join(text.split())


def load_pdf(path):
    path = Path(path)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if text:  # blank pages and scanned images give nothing, skip them
            pages.append(Page(path.name, i, text))
    return pages


def find_pdfs(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() == ".pdf")


def load_folder(folder):
    """Load every PDF under a folder. Broken files are reported and skipped, not fatal."""
    pages = []
    for path in find_pdfs(folder):
        try:
            found = load_pdf(path)
        except (PdfReadError, OSError) as e:
            print(f"skipping {path.name}: {e}")
            continue
        if not found:
            print(f"skipping {path.name}: no text (maybe a scanned PDF?)")
        pages.extend(found)
    return pages


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    pages = load_folder(folder)
    by_file = {}
    for p in pages:
        by_file.setdefault(p.file, []).append(p)
    for name, ps in by_file.items():
        words = sum(len(p.text.split()) for p in ps)
        print(f"{name}: {len(ps)} pages with text, {words:,} words")
        print(f"   page {ps[0].number}: {ps[0].text[:120]}...")
    if not pages:
        print(f"no PDFs with text found in {folder}/")
