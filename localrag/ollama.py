"""
A small client for the two Ollama endpoints this project needs:
/api/embed for embeddings and /api/chat for answers.
Plain requests, no extra SDK.
"""

import json

import requests

from localrag import config


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url=None, embed_model=None, chat_model=None, timeout=120):
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.embed_model = embed_model or config.EMBED_MODEL
        self.chat_model = chat_model or config.CHAT_MODEL
        self.timeout = timeout

    def _post(self, path, payload, stream=False):
        try:
            r = requests.post(f"{self.url}{path}", json=payload, timeout=self.timeout, stream=stream)
        except requests.ConnectionError:
            raise OllamaError(
                f"Couldn't connect to Ollama at {self.url}. Is it running? Start it with `ollama serve` "
                f"(or set OLLAMA_URL if it's somewhere else)."
            ) from None
        except requests.Timeout:
            raise OllamaError(f"Ollama at {self.url} took longer than {self.timeout}s to answer.") from None
        if r.status_code == 404:
            # Ollama answers 404 when the model hasn't been downloaded
            model = payload.get("model")
            raise OllamaError(f"Ollama doesn't have the model '{model}'. Get it with `ollama pull {model}`.")
        if not r.ok:
            raise OllamaError(f"Ollama returned {r.status_code}: {r.text[:200]}")
        return r

    def embed(self, texts, batch_size=32):
        """One vector per text. Sent in batches so a big PDF doesn't become one giant request."""
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = self._post("/api/embed", {"model": self.embed_model, "input": batch}).json()
            vectors.extend(data["embeddings"])
        return vectors

    def chat(self, messages):
        data = self._post("/api/chat", {"model": self.chat_model, "messages": messages, "stream": False}).json()
        return data["message"]["content"]

    def chat_stream(self, messages):
        """Yields the answer piece by piece as Ollama writes it."""
        r = self._post("/api/chat", {"model": self.chat_model, "messages": messages, "stream": True}, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise OllamaError(data["error"])
            yield data.get("message", {}).get("content", "")
            if data.get("done"):
                break
