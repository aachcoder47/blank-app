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

# -------- Document loaders --------

def load_txt(file):
    text = file.read().decode('utf-8')
    return [{"source": file.name, "text": text}]

def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return [{"source": file.name, "text": text}]

def load_docx(file):
    doc = Document(io.BytesIO(file.read()))
    text = " ".join([para.text for para in doc.paragraphs])
    return [{"source": file.name, "text": text}]

def load_csv(file):
    df = pd.read_csv(file)
    text = df.astype(str).values.flatten()
    return [{"source": file.name, "text": " ".join(text)}]

def load_documents(files):
    docs = []
    for file in files:
        if file.name.endswith('.pdf'):
            docs += load_pdf(file)
        elif file.name.endswith('.docx'):
            docs += load_docx(file)
        elif file.name.endswith('.csv'):
            docs += load_csv(file)
        elif file.name.endswith('.txt'):
            docs += load_txt(file)
    return docs

# -------- Webpage loader with User-Agent to avoid 403 --------

def fetch_web_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        return [{"source": url, "text": text}]
    except Exception as e:
        st.error(f"Failed to fetch {url}: {str(e)}")
        return []

# -------- Text chunking --------

def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc['text'])
        for idx, chunk in enumerate(splits):
            chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_chunk{idx}",
                "content": chunk
            })
    return chunks

# -------- Embeddings --------

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks, embedder):
    texts = [chunk['content'] for chunk in chunks]
    emb = embedder.encode(texts, show_progress_bar=True)
    return np.array(emb).astype('float32')

# -------- Vector store (FAISS) --------

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -------- Retriever --------

def retrieve(query, embedder, index, chunks, top_k=3):
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype('float32')
    distances, indices = index.search(q_emb, top_k)
    retrieved = [{"chunk": chunks[i], "score": float(distances[0][idx])} for idx, i in enumerate(indices[0])]
    return retrieved

# -------- LLM Integration with open ungated model --------

@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
        tokenizer="MehdiHosseiniMoghadam/AVA-Mistral-7B-V2",
        max_length=1024,
        do_sample=True,
        temperature=0.2,
        trust_remote_code=True,
    )

def answer_with_llm(context_chunks, query, llm):
    context_text = "\n".join([f"[{c['chunk_id']}] {c['content']}" for c in context_chunks])
    prompt = (
        f"Answer the following question using ONLY the provided context and cite the chunk ids used.\n"
        f"Question: {query}\n"
        "Context:\n"
        f"{context_text}\n"
        "Answer with citations:"
    )
    generation = llm(prompt, max_length=512, num_return_sequences=1)
    return generation[0]['generated_text']

# -------- Chat memory --------

def update_memory(session, query, answer, sources):
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"query": query, "answer": answer, "sources": sources})

# -------- Streamlit UI --------

st.set_page_config(page_title="RAG Q&A with Open Model and Web URLs", layout="wide")
st.title("💡 Retrieval-Augmented Generation (RAG) Demo with Open Model & Web URLs")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT, CSV)", 
    type=["pdf", "txt", "docx", "csv"], 
    accept_multiple_files=True
)

url_input = st.text_area(
    "Or enter web URLs (one per line) to fetch content",
    value="",
    help="Paste one or more URLs. Their text contents will be fetched and included."
)

web_docs = []
if url_input.strip():
    urls = [url.strip() for url in url_input.splitlines() if url.strip()]
    with st.spinner("Fetching web content..."):
        for url in urls:
            web_docs += fetch_web_text(url)

file_docs = load_documents(uploaded_files) if uploaded_files else []
all_docs = file_docs + web_docs

if all_docs:
    st.success(f"{len(all_docs)} document(s) loaded from files and URLs.")

    with st.spinner("Chunking documents and generating embeddings..."):
        chunks = chunk_documents(all_docs)
        embedder = load_embedder()
        embeddings = get_embeddings(chunks, embedder)
        index = build_faiss_index(embeddings)

    st.write(f"{len(chunks)} chunks created and indexed.")

    query = st.text_input("Ask a question about the loaded documents:")

    if query:
        llm = load_llm()
        relevant_chunks = retrieve(query, embedder, index, chunks, top_k=3)
        answer_text = answer_with_llm([rc["chunk"] for rc in relevant_chunks], query, llm)
        source_refs = "\n".join([f"[{rc['chunk']['chunk_id']} from {rc['chunk']['source']}]" for rc in relevant_chunks])

        st.markdown("### ► Answer")
        st.markdown(answer_text)
        st.markdown("### ► Sources")
        st.code(source_refs)

        for rc in relevant_chunks:
            with st.expander(f"Source: {rc['chunk']['chunk_id']} ({rc['chunk']['source']})"):
                st.write(rc['chunk']['content'])

        update_memory(st.session_state, query, answer_text, source_refs)

    if 'chat_history' in st.session_state and st.session_state['chat_history']:
        st.markdown("---")
        st.markdown("### Session Chat History")
        for chat in reversed(st.session_state['chat_history']):
            st.write(f"*Q:* {chat['query']}")
            st.write(f"*A:* {chat['answer']}")
            st.write(f"*Sources:* {chat['sources']}")

else:
    st.info("Upload documents or enter URLs to get started.")

st.caption("RAG Demo using Mistral-7B-Instruct (ungated open model), FAISS, and Web URLs – Streamlit All-in-One")
