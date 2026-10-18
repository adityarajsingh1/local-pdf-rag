import ask
from conftest import FakeOllama, fake_embed
from localrag.chunks import Chunk
from localrag.store import Store


class StreamingOllama(FakeOllama):
    chat_model = "fake-chat"

    def chat_stream(self, messages):
        yield from ["Steel ", "tubes ", "[wind.pdf p.2]."]


def _store(tmp_path):
    store = Store(path=str(tmp_path / "db"), embed_model="fake")
    chunks = [Chunk("wind.pdf", 2, 0, "monopiles are steel tubes"), Chunk("cake.pdf", 1, 0, "flour and eggs")]
    store.add(chunks, fake_embed([c.text for c in chunks]))
    return store


def test_ask_once_streams_and_lists_sources(tmp_path, capsys):
    text = ask.ask_once("what are monopiles?", _store(tmp_path), StreamingOllama(), k=1, show_chunks=True)
    out = capsys.readouterr().out
    assert text == "Steel tubes [wind.pdf p.2]."
    assert "sources: wind.pdf p.2" in out and "monopiles are steel tubes" in out
    assert "[wind.pdf p.2]" in out  # the citation has to survive rich's markup parsing


def test_repl_commands(tmp_path, capsys, monkeypatch):
    typed = iter(["/help", "/k 2", "/k nope", "/files", "/chunks", "/what", "what are monopiles?", "/quit"])
    monkeypatch.setattr(ask.console, "input", lambda prompt="": next(typed))
    ask.repl(_store(tmp_path), StreamingOllama(), k=5, show_chunks=False)
    out = capsys.readouterr().out
    assert "using 2 chunks" in out and "usage: /k 8" in out
    assert "cake.pdf" in out and "chunks on" in out and "unknown command" in out
    assert "Steel tubes" in out and "bye!" in out
