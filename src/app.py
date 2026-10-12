"""Streamlit UI for the AI Research Assistant.

Run from the project root:
    streamlit run src/app.py

Pipeline: upload -> extract text -> chunk -> keyword search -> LLM answer with citations
"""

import streamlit as st

import config
import llm
from chunking import chunk_pages
from loaders import load_document
from retrieval import BM25Index

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research Assistant")

# Streamlit re-runs this whole script on every click, so anything we want to
# keep between runs goes in st.session_state.
if "documents" not in st.session_state:
    st.session_state.documents = {}  # file name -> list[Page]

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (words)", 50, 500, 200, step=25)
    overlap = st.slider("Chunk overlap (words)", 0, chunk_size - 25, min(40, chunk_size - 25), step=5)
    top_k = st.slider("Chunks to retrieve (top k)", 1, 10, 4)
    min_score = st.slider("Minimum relevance score", 0.0, 5.0, 0.0, step=0.1,
                          help="Chunks scoring at or below this are ignored.")
    st.divider()
    if llm.is_configured():
        st.success(f"LLM: OpenAI `{config.OPENAI_MODEL}`")
    else:
        st.warning("No OPENAI_API_KEY found. Running in **retrieval-only mode** "
                   "(no API calls, no cost). See README to add a key.")

# ---------- 1. Upload ----------
st.subheader("1. Upload documents")
uploads = st.file_uploader("TXT or PDF files", type=["txt", "pdf"], accept_multiple_files=True)
for f in uploads or []:
    if f.name not in st.session_state.documents:
        try:
            st.session_state.documents[f.name] = load_document(f.name, f.getvalue())
        except Exception as e:  # bad/corrupt file: report it, keep the app running
            st.error(f"Could not read {f.name}: {e}")

# Drop documents the user removed from the uploader
uploaded_names = {f.name for f in uploads or []}
for name in list(st.session_state.documents):
    if name not in uploaded_names:
        del st.session_state.documents[name]

documents = st.session_state.documents
if not documents:
    st.info("Upload at least one document to get started.")
    st.stop()

# ---------- 2. Show extracted text ----------
st.subheader("2. Extracted text")
for name, pages in documents.items():
    full_text = "\n\n".join(p.text for p in pages)
    with st.expander(f"{name} — {len(pages)} page(s), {len(full_text.split())} words"):
        if not full_text.strip():
            st.warning("No text could be extracted (the PDF may be scanned images).")
        st.text(full_text[:20000] + ("\n\n… (truncated for display)" if len(full_text) > 20000 else ""))

# ---------- 3. Chunk + index ----------
all_pages = [p for pages in documents.values() for p in pages]
chunks = chunk_pages(all_pages, chunk_size=chunk_size, overlap=overlap)
index = BM25Index(chunks)
st.caption(f"Split into **{len(chunks)} chunks** of up to {chunk_size} words ({overlap}-word overlap).")

# ---------- 4. Ask ----------
st.subheader("3. Ask a question")
question = st.text_input("Your question", placeholder="e.g. What method does the paper propose?")
if not question:
    st.stop()

results = index.search(question, top_k=top_k, min_score=min_score)

st.subheader("Answer")
if not results:
    # Retrieval found nothing relevant: say so without spending an API call.
    st.error("**Not enough information.** None of your documents contain words related "
             "to this question, so I can't answer it from them.")
    st.stop()

if llm.is_configured():
    with st.spinner("Asking the LLM…"):
        try:
            answer = llm.generate_answer(question, results)
        except RuntimeError as e:
            st.error(str(e))
            answer = None
    if answer and answer.enough_info:
        st.markdown(answer.text)
    elif answer:
        st.error(f"**Not enough information in your documents.** {answer.text}")
else:
    st.info("Retrieval-only mode: here are the most relevant passages. "
            "Add an OpenAI API key to get a written answer.")

# ---------- 5. Sources ----------
st.subheader("Sources")
for n, r in enumerate(results, start=1):
    c = r.chunk
    with st.expander(f"[{n}] {c.doc_name}, page {c.page_number} — score {r.score:.2f}"):
        st.write(c.text)
