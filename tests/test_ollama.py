import json as jsonlib

import pytest
import requests

from localrag import ollama
from localrag.ollama import Ollama, OllamaError


class FakeResponse:
    def __init__(self, status=200, data=None, lines=None):
        self.status_code = status
        self.ok = status < 400
        self._data = data or {}
        self._lines = lines or []
        self.text = jsonlib.dumps(self._data)

    def json(self):
        return self._data

    def iter_lines(self):
        return iter(self._lines)


@pytest.fixture
def calls(monkeypatch):
    """Records every request and answers like Ollama would."""
    seen = []

    def fake_post(url, json=None, timeout=None, stream=False):
        seen.append((url, json))
        if url.endswith("/api/embed"):
            return FakeResponse(data={"embeddings": [[float(len(t)), 1.0] for t in json["input"]]})
        if json.get("stream"):
            parts = [{"message": {"content": "Hel"}}, {"message": {"content": "lo"}}, {"done": True}]
            return FakeResponse(lines=[jsonlib.dumps(p).encode() for p in parts])
        return FakeResponse(data={"message": {"content": "Hello"}})

    monkeypatch.setattr(ollama.requests, "post", fake_post)
    return seen


def test_embed_batches_and_keeps_order(calls):
    vecs = Ollama(url="http://x").embed(["a", "bb", "ccc"], batch_size=2)
    assert vecs == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert len(calls) == 2 and calls[0][0] == "http://x/api/embed"


def test_chat_and_stream(calls):
    client = Ollama(url="http://x", chat_model="llama3.2")
    msgs = [{"role": "user", "content": "hi"}]
    assert client.chat(msgs) == "Hello"
    assert "".join(client.chat_stream(msgs)) == "Hello"
    assert calls[0][1]["model"] == "llama3.2"


def test_clear_error_when_ollama_is_not_running(monkeypatch):
    def refuse(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama.requests, "post", refuse)
    with pytest.raises(OllamaError, match="ollama serve"):
        Ollama(url="http://localhost:1").embed(["x"])


def test_missing_model_says_how_to_pull_it(monkeypatch):
    monkeypatch.setattr(ollama.requests, "post", lambda *a, **kw: FakeResponse(status=404))
    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        Ollama(url="http://x", embed_model="nomic-embed-text").embed(["x"])
