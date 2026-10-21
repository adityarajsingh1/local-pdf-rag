"""
End to end: real PDFs on disk, the real ingest.py and ask.py scripts, the real
Ollama client and chromadb. Only Ollama itself is faked, by a tiny HTTP server
that speaks the same /api/embed and /api/chat as the real one.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import fake_embed, make_pdf

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"fake-embed", "fake-chat"}


class FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload))
        if payload.get("model") not in MODELS:
            return self._send(404, {"error": f"model '{payload.get('model')}' not found"})

        if self.path == "/api/embed":
            return self._send(200, {"embeddings": fake_embed(payload["input"])})

        if self.path == "/api/chat":
            # answer with the first excerpt it was given, cited the way the prompt asks
            prompt = payload["messages"][-1]["content"]
            found = re.search(r"\[(.+? p\.\d+)\]\n(.+)", prompt)
            reply = f"{found.group(2).strip()} [{found.group(1)}]" if found else "I couldn't find that."
            if not payload.get("stream"):
                return self._send(200, {"message": {"role": "assistant", "content": reply}, "done": True})
            lines = [json.dumps({"message": {"content": w + " "}, "done": False}) for w in reply.split()]
            lines.append(json.dumps({"message": {"content": ""}, "done": True}))
            return self._send(200, "\n".join(lines) + "\n")

        self._send(404, {"error": "unknown endpoint"})


@pytest.fixture
def ollama_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def project(tmp_path, ollama_server):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "bikes.pdf", ["Chains need oil every month.", "Tyres should be pumped to four bar."])
    make_pdf(docs / "plants.pdf", ["Basil likes warm sunny windows."])
    env = {**os.environ,
           "OLLAMA_URL": f"http://127.0.0.1:{ollama_server.server_port}",
           "EMBED_MODEL": "fake-embed", "CHAT_MODEL": "fake-chat",
           "DB_PATH": str(tmp_path / "db")}

    def run(script, *args, **env_overrides):
        return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=tmp_path,
                              env={**env, **env_overrides}, capture_output=True, text=True, timeout=120)

    run.docs = docs
    return run


def test_ingest_then_ask_cites_the_right_page(project, ollama_server):
    out = project("ingest.py", "docs")
    assert out.returncode == 0, out.stderr
    assert "2 new" in out.stdout and "3 chunks from 2 files" in out.stdout

    out = project("ask.py", "what should the tyres be pumped to?", "-k", "1")
    assert out.returncode == 0, out.stderr
    assert "pumped to four bar. [bikes.pdf p.2]" in out.stdout
    assert "sources: bikes.pdf p.2" in out.stdout

    # the model was asked with the retrieved excerpt in the prompt, streamed
    path, chat = ollama_server.requests[-1]
    assert path == "/api/chat" and chat["stream"] is True
    assert "[bikes.pdf p.2]" in chat["messages"][-1]["content"]


def test_reindexing_picks_up_changes_and_deletions(project):
    project("ingest.py", "docs")
    assert "2 unchanged" in project("ingest.py", "docs").stdout

    make_pdf(project.docs / "plants.pdf", ["Basil likes warm sunny windows.", "Mint grows anywhere."])
    (project.docs / "bikes.pdf").unlink()
    out = project("ingest.py", "docs")
    assert "1 updated" in out.stdout and "1 removed" in out.stdout
    assert "2 chunks from 1 files" in out.stdout

    out = project("ask.py", "which plant grows anywhere?", "-k", "1")
    assert "[plants.pdf p.2]" in out.stdout


def test_ask_before_ingest_says_what_to_do(project):
    out = project("ask.py", "anything?")
    assert out.returncode != 0
    assert "run `python ingest.py` first" in out.stderr


def test_ollama_not_running(project):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # free port, nothing listening once closed
    out = project("ingest.py", "docs", OLLAMA_URL=f"http://127.0.0.1:{port}")
    assert out.returncode != 0
    assert "Couldn't connect to Ollama" in out.stderr and "ollama serve" in out.stderr


def test_missing_model_tells_you_to_pull_it(project):
    out = project("ingest.py", "docs", EMBED_MODEL="nomic-embed-text")
    assert out.returncode != 0
    assert "ollama pull nomic-embed-text" in out.stderr
