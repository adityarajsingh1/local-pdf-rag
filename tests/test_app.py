from pathlib import Path

import pytest
import streamlit as st
from conftest import fake_embed, make_pdf
from streamlit.testing.v1 import AppTest

from localrag import config
from localrag.ollama import Ollama, OllamaError

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def env(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "wind.pdf", ["Monopiles are steel tubes driven into the seabed."])
    st.cache_resource.clear()  # the app caches the store, and each test has its own folder
    monkeypatch.setattr(config, "DOCS_DIR", str(docs))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "db"))
    monkeypatch.setattr(Ollama, "embed", lambda self, texts, batch_size=32: fake_embed(texts))
    monkeypatch.setattr(Ollama, "chat_stream", lambda self, msgs: iter(["Steel ", "tubes [wind.pdf p.1]."]))
    return docs


def test_index_then_ask(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.sidebar.info[0].value == "nothing indexed yet"

    at.sidebar.button[0].click().run()  # index
    assert "1 new" in at.sidebar.success[0].value

    at.chat_input[0].set_value("what are monopiles?").run()
    assert not at.exception
    assert "Steel tubes" in at.chat_message[1].markdown[0].value
    assert "wind.pdf p.1" in at.chat_message[1].caption[0].value


def test_ollama_down_shows_an_error(env, monkeypatch):
    def down(self, texts, batch_size=32):
        raise OllamaError("Couldn't connect to Ollama")

    monkeypatch.setattr(Ollama, "embed", down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    assert "Couldn't connect" in at.sidebar.error[0].value
