"""
Settings, all from environment variables (or a .env file) so nothing is hard coded.

    OLLAMA_URL    where Ollama is running      (default http://localhost:11434)
    EMBED_MODEL   model for embeddings         (default nomic-embed-text)
    CHAT_MODEL    model that writes answers    (default llama3.2)
    DB_PATH       where chromadb keeps its files (default chroma_db/)
    CHUNK_SIZE / CHUNK_OVERLAP   in characters (default 1000 / 200)
"""

import os
from pathlib import Path


def load_dotenv(path=".env"):
    # tiny .env reader, KEY=value per line. real env vars win over the file
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
DB_PATH = os.getenv("DB_PATH", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
