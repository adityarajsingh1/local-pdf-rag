from conftest import FakeOllama, fake_embed

from localrag.chunks import Chunk
from localrag.qa import answer, build_messages, sources
from localrag.store import Store


class ChatOllama(FakeOllama):
    """Fake Ollama that remembers the prompt and answers with a canned reply."""

    def __init__(self):
        self.sent = None

    def chat(self, messages):
        self.sent = messages
        return "Monopiles are steel tubes [wind.pdf p.2]."


def _store(tmp_path):
    store = Store(path=str(tmp_path / "db"), embed_model="fake")
    chunks = [Chunk("wind.pdf", 2, 0, "monopiles are steel tubes driven into the seabed"),
              Chunk("wind.pdf", 2, 1, "monopiles are the most common foundation"),
              Chunk("wind.pdf", 7, 0, "floating turbines are moored with chains"),
              Chunk("cake.pdf", 1, 0, "mix flour sugar and eggs")]
    store.add(chunks, fake_embed([c.text for c in chunks]))
    return store


def test_answer_cites_file_and_page(tmp_path):
    client = ChatOllama()
    reply, cited, hits = answer("what are monopiles?", _store(tmp_path), client, k=2)
    assert "[wind.pdf p.2]" in reply
    assert cited == [("wind.pdf", 2)]  # two chunks from the same page show up once
    prompt = client.sent[1]["content"]
    assert "[wind.pdf p.2]" in prompt and "Question: what are monopiles?" in prompt


def test_empty_index(tmp_path):
    store = Store(path=str(tmp_path / "db"), embed_model="fake")
    reply, cited, hits = answer("anything?", store, ChatOllama())
    assert "ingest.py" in reply and cited == [] and hits == []


def test_system_prompt_says_not_to_guess():
    msgs = build_messages("q", [(Chunk("a.pdf", 1, 0, "text"), 0.1)])
    assert msgs[0]["role"] == "system" and "don't guess" in msgs[0]["content"]


def test_sources_keep_order():
    hits = [(Chunk("b.pdf", 3, 0, ""), 0.1), (Chunk("a.pdf", 1, 0, ""), 0.2), (Chunk("b.pdf", 3, 1, ""), 0.3)]
    assert sources(hits) == [("b.pdf", 3), ("a.pdf", 1)]
