import pytest

from localrag.chunks import chunk_pages, split_text
from localrag.pdfs import Page

WORDS = " ".join(f"word{i}" for i in range(500))  # ~4,400 characters


def test_short_text_is_one_chunk():
    assert split_text("just a short page", size=100, overlap=20) == ["just a short page"]


def test_chunks_respect_the_size_and_dont_cut_words():
    pieces = split_text(WORDS, size=300, overlap=50)
    assert len(pieces) > 10
    assert all(len(p) <= 300 for p in pieces)
    vocab = set(WORDS.split())
    assert all(set(p.split()) <= vocab for p in pieces)  # no half words


def test_neighbouring_chunks_overlap():
    pieces = split_text(WORDS, size=300, overlap=50)
    for a, b in zip(pieces, pieces[1:]):
        assert a.split()[-1] in b.split()[:10]


def test_nothing_is_lost():
    pieces = split_text(WORDS, size=300, overlap=50)
    assert set(" ".join(pieces).split()) == set(WORDS.split())


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text(WORDS, size=100, overlap=100)


def test_text_without_spaces_still_terminates():
    pieces = split_text("x" * 1000, size=300, overlap=100)
    assert "".join(pieces).count("x") >= 1000


def test_chunks_keep_their_page_and_unique_ids():
    pages = [Page("a.pdf", 1, WORDS), Page("a.pdf", 2, "second page"), Page("b.pdf", 1, "other file")]
    chunks = chunk_pages(pages, size=500, overlap=100)
    assert {c.page for c in chunks if c.file == "a.pdf"} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[-1].id == "b.pdf:p1:0"
