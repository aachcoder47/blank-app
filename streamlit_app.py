import streamlit as st
import io
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

# ---- Document loaders ----
def load_txt(file):
    return [{"source": file.name, "text": file.read().decode("utf-8")}]

def load_pdf(file):
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return [{"source": file.name, "text": text}]

def load_docx(file):
    doc = Document(io.BytesIO(file.read()))
    text = " ".join(p.text for p in doc.paragraphs)
    return [{"source": file.name, "text": text}]

def load_csv(file):
    df = pd.read_csv(file)
    text = " ".join(df.astype(str).values.flatten())
    return [{"source": file.name, "text": text}]

def load_files(files):
    docs = []
    for f in files:
        if f.name.endswith(".txt"):
            docs.extend(load_txt(f))
        elif f.name.endswith(".pdf"):
            docs.extend(load_pdf(f))
        elif f.name.endswith(".docx"):
            docs.extend(load_docx(f))
        elif f.name.endswith(".csv"):
            docs.extend(load_csv(f))
    return docs

# ---- Web content loader ----
def fetch_web_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        return [{"source": url, "text": soup.get_text(separator=" ")}]
    except Exception as e:
        st.error(f"Error fetching URL {url}: {e}")
        return []

# ---- Chunking ----
def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for idx, chunk in enumerate(splits):
            chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_chunk{idx}",
                "content": chunk
            })
    return chunks

# ---- Embeddings loader (preload models) ----
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline(
        "text-generation",
        model="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
        tokenizer="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
        max_length=1024,
        do_sample=True,
        temperature=0.2,
        trust_remote_code=True,
    )
    return embedder, llm

# ---- Vector index build ----
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ---- Embeddings extraction ----
def embed_chunks(chunks, embedder):
    texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

# ---- Retrieval ----
def retrieve_top_k(query, embedder, index, chunks, k=3):
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    dists, inds = index.search(q_emb, k)
    return [{"chunk": chunks[i], "score": float(dists[0][idx])} for idx, i in enumerate(inds[0])]

# ---- Answer generation with citations ----
def generate_answer(llm, context_chunks, query):
    context_text = "\n".join([f"[{c['chunk_id']}] {c['content']}" for c in context_chunks])
    prompt = (
        f"Answer the question using ONLY the provided context and cite the chunk ids.\n"
        f"Question: {query}\n"
        f"Context:\n{context_text}\n"
        "Answer with citations:"
    )
    result = llm(prompt, max_length=512, num_return_sequences=1)
    return result[0]["generated_text"]

# ---- Streamlit App ----
def main():
    st.set_page_config(page_title="RAG Demo - Preloaded Models", layout="wide")
    st.title("💡 Retrieval-Augmented Generation (RAG) with Preloaded Models")

    # Preload embedder and LLM on app launch
    embedder, llm = load_models()
    st.success("Models loaded and cached ✅")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV), multiple:", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True
    )
    url_text = st.text_area("Or enter web URLs (one per line):")

    web_docs = []
    if url_text.strip():
        urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        with st.spinner("Fetching web content..."):
            for url in urls:
                web_docs.extend(fetch_web_text(url))

    file_docs = load_files(uploaded_files) if uploaded_files else []
    all_docs = file_docs + web_docs

    if all_docs:
        st.success(f"Loaded {len(all_docs)} documents (files + URLs)")
        with st.spinner("Chunking documents and generating embeddings..."):
            chunks = chunk_documents(all_docs)
            embeddings = embed_chunks(chunks, embedder)
            index = build_faiss_index(embeddings)
        st.write(f"Indexed {len(chunks)} chunks.")

        query = st.text_input("Ask a question about the loaded content:")
        if query:
            results = retrieve_top_k(query, embedder, index, chunks, k=3)
            answer = generate_answer(llm, [r["chunk"] for r in results], query)

            st.markdown("### Answer")
            st.markdown(answer)

            st.markdown("### Sources")
            source_refs = "\n".join([f"[{r['chunk']['chunk_id']} from {r['chunk']['source']}]" for r in results])
            st.code(source_refs)

            for r in results:
                with st.expander(f"Chunk: {r['chunk']['chunk_id']} (Source: {r['chunk']['source']})"):
                    st.write(r["chunk"]["content"])
    else:
        st.info("Upload documents or enter URLs to start.")

    st.caption("RAG Demo with FAISS, Sentence-Transformers & Mistral-7B — Streamlit")

if __name__ == "__main__":
    main()
