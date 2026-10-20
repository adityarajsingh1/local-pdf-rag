"""
Web version of ask.py.

    streamlit run app.py

Upload PDFs in the sidebar, hit "index", then ask away. Same index as the
command line tools, so anything indexed with ingest.py shows up here too.
"""

from pathlib import Path

import streamlit as st

from ingest import index_folder
from localrag import config
from localrag.ollama import Ollama, OllamaError
from localrag.qa import answer_stream
from localrag.store import Store

DOCS = Path(config.DOCS_DIR)

st.set_page_config(page_title="local-pdf-rag", page_icon="📄")
st.title("📄 Chat with your PDFs, locally")


@st.cache_resource
def get_store():
    return Store()


@st.cache_resource
def get_client():
    return Ollama()


store, client = get_store(), get_client()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("add PDFs", type="pdf", accept_multiple_files=True)
    if st.button("index", type="primary", use_container_width=True):
        DOCS.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            (DOCS / f.name).write_bytes(f.getvalue())
        with st.spinner("reading and embedding, this can take a while for big PDFs..."):
            try:
                stats = index_folder(DOCS, store, client)
                st.success(f"{stats['added']} new, {stats['updated']} updated, "
                           f"{stats['unchanged']} unchanged, {stats['removed']} removed")
            except OllamaError as e:
                st.error(str(e))

    files = store.files()
    if files:
        st.caption(f"{store.count()} chunks from {len(files)} PDFs")
        for name in files:
            st.markdown(f"- {name}")
    else:
        st.info("nothing indexed yet")

    k = st.slider("chunks per answer", 1, 10, 5)
    st.caption(f"embeddings: {client.embed_model} · answers: {client.chat_model}")
    if st.button("clear chat", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("sources: " + msg["sources"])

if question := st.chat_input("ask something about your PDFs", disabled=not files):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            pieces, cited, hits = answer_stream(question, store, client, k=k)
            reply = st.write_stream(pieces)
        except OllamaError as e:
            st.error(str(e))
            st.stop()
        sources = ", ".join(f"{f} p.{p}" for f, p in cited)
        if sources:
            st.caption("sources: " + sources)
        with st.expander("the chunks this answer is based on"):
            for chunk, dist in hits:
                st.markdown(f"**{chunk.file} p.{chunk.page}** · distance {dist:.3f}")
                st.text(chunk.text)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
