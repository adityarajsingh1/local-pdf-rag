import pytest
from conftest import FakeOllama, make_pdf

from ingest import index_folder
from localrag.ollama import OllamaError
from localrag.store import Store


class CountingOllama(FakeOllama):
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


@pytest.fixture
def setup(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "a.pdf", ["first file about turbines"])
    make_pdf(docs / "b.pdf", ["second file about cables"])
    return docs, Store(path=str(tmp_path / "db"), embed_model="fake"), CountingOllama()


def test_unchanged_files_are_skipped(setup):
    docs, store, client = setup
    assert index_folder(docs, store, client)["added"] == 2
    stats = index_folder(docs, store, client)
    assert stats["unchanged"] == 2 and client.calls == 2  # nothing new was embedded


def test_changed_file_is_reindexed(setup):
    docs, store, client = setup
    index_folder(docs, store, client)
    make_pdf(docs / "a.pdf", ["first file, now about foundations", "and a second page"])
    stats = index_folder(docs, store, client)
    assert stats["updated"] == 1 and stats["unchanged"] == 1
    assert store.count() == 3


def test_deleted_file_is_removed(setup):
    docs, store, client = setup
    index_folder(docs, store, client)
    (docs / "b.pdf").unlink()
    assert index_folder(docs, store, client)["removed"] == 1
    assert store.files() == ["a.pdf"]


def test_other_folders_are_left_alone(setup, tmp_path):
    docs, store, client = setup
    other = tmp_path / "other"
    other.mkdir()
    make_pdf(other / "c.pdf", ["from another folder"])
    index_folder(docs, store, client)
    index_folder(other, store, client)
    assert store.files() == ["a.pdf", "b.pdf", "c.pdf"]


def test_chunk_settings_change_means_reindex(setup):
    docs, store, client = setup
    index_folder(docs, store, client, size=1000, overlap=200)
    assert index_folder(docs, store, client, size=500, overlap=100)["updated"] == 2


def test_failed_embedding_keeps_the_old_chunks(setup):
    docs, store, client = setup
    index_folder(docs, store, client)
    make_pdf(docs / "a.pdf", ["changed"])

    class Broken(FakeOllama):
        def embed(self, texts):
            raise OllamaError("down")

    with pytest.raises(OllamaError):
        index_folder(docs, store, Broken())
    assert store.files() == ["a.pdf", "b.pdf"]
