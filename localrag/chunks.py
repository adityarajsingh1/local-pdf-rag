"""
Split pages into overlapping chunks for embedding.

Chunks never cross a page boundary. That way every chunk has exactly one
page number to cite. The overlap means a sentence that falls on a chunk
border still shows up whole in one of the two chunks.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    file: str
    page: int
    index: int   # position of the chunk within its page, keeps ids unique
    text: str

    @property
    def id(self):
        return f"{self.file}:p{self.page}:{self.index}"


def split_text(text, size=1000, overlap=200):
    """Cut text into pieces of about `size` characters, preferring to break at a space."""
    if overlap >= size:
        raise ValueError("overlap has to be smaller than the chunk size")
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # back up to the last space so we don't cut a word in half
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # step back by the overlap, but start the next chunk at a word, not halfway through one
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        if space != -1:
            next_start = space + 1
        # always move forward, even if the overlap would put us back where we started
        start = max(next_start, start + 1)
    return pieces


def chunk_pages(pages, size=1000, overlap=200):
    chunks = []
    for page in pages:
        for i, piece in enumerate(split_text(page.text, size, overlap)):
            chunks.append(Chunk(page.file, page.number, i, piece))
    return chunks
