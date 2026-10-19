from conftest import fake_embed, make_pdf

from ingest import index_folder
from localrag.chunks import Chunk
from localrag.store import Store, collection_name


def test_add_and_search(tmp_path):
    store = Store(path=str(tmp_path / "db"), embed_model="fake")
    chunks = [Chunk("a.pdf", 1, 0, "monopiles are steel tubes driven into the seabed"),
              Chunk("a.pdf", 2, 0, "floating turbines are moored with chains"),
              Chunk("b.pdf", 1, 0, "the cake needs flour sugar and eggs")]
    store.add(chunks, fake_embed([c.text for c in chunks]))
    hit, dist = store.search(fake_embed(["what holds floating turbines in place? chains"])[0], k=1)[0]
    assert (hit.file, hit.page) == ("a.pdf", 2)
    assert store.files() == ["a.pdf", "b.pdf"]


def test_empty_store_search(tmp_path):
    assert Store(path=str(tmp_path / "db"), embed_model="fake").search([0.1] * 64) == []


def test_collection_name_is_safe():
    assert collection_name("nomic-embed-text:latest") == "pdfs-nomic-embed-text-latest"


def test_ingest_twice_does_not_duplicate(tmp_path, fake_ollama):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "wind.pdf", ["Offshore wind farms use monopile foundations.", "Export cables bring power ashore."])
    make_pdf(docs / "cake.pdf", ["Mix the flour, sugar and eggs."])
    store = Store(path=str(tmp_path / "db"), embed_model="fake")
    first = index_folder(docs, store, fake_ollama)
    index_folder(docs, store, fake_ollama, force=True)
    assert first["chunks"] == 3 and store.count() == 3
    hit, _ = store.search(fake_embed(["how does the power get ashore? export cables"])[0], k=1)[0]
    assert (hit.file, hit.page) == ("wind.pdf", 2)
