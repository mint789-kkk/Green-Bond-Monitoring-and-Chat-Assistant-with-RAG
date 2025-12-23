import os
import streamlit as st
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Optional but strongly recommended for real PDF text extraction
# pip install pymupdf
import fitz  # PyMuPDF


# ----------------------------
# Helpers
# ----------------------------
def load_embeddings():
    # Requires NVIDIA_API_KEY in env (or set from UI below)
    return NVIDIAEmbeddings(model="nvidia/nv-embed-v1")


def build_vector_store(docs):
    embeddings = load_embeddings()
    return FAISS.from_documents(docs, embeddings)


def format_sources(retrieved_docs, max_chars=1200):
    sources = []
    for i, d in enumerate(retrieved_docs, start=1):
        sources.append(
            {
                "id": i,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", None),
                "chunk": d.metadata.get("chunk", None),
                # show the actual retrieved chunk text (truncated for UI)
                "text": d.page_content[:max_chars],
            }
        )
    return sources



def answer_question(llm, question, retrieved_docs):
    # Give the model ONLY the original text chunks (no SOURCE/PAGE labels),
    # so citations [1],[2] refer to these chunks without echoing file names.
    context_blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"Source [{i}]:\n{d.page_content}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Green Bond RAG assistant. Use ONLY the provided sources. "
                "When you answer, include citations like [1], [2] that refer to the sources. "
                "Do NOT repeat file names, page numbers, or the words SOURCE/PAGE/CHUNK in your answer. "
                "If the answer is not in the sources, say you can't find it.",
            ),
            ("human", "Question:\n{question}\n\nSources:\n{context}"),
        ]
    )

    msg = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msg)
    return resp.content


def extract_pdf_to_chunked_docs(
    pdf_bytes: bytes,
    source_name: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
):
    """
    Extract real text from PDF (per page) and split into chunks to avoid embedding token limits.
    Returns a list[Document] with metadata: source, page, chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for pno in range(len(pdf)):
        text = pdf[pno].get_text("text") or ""
        text = text.strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for ci, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source_name, "page": pno + 1, "chunk": ci},
                )
            )

    return docs


# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("GreenBond-RAG: Sustainability Document Assistant")
    st.caption("Upload docs → index → chat with citations (audit-friendly).")

    # Session state init
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    nvidia_key = st.text_input("NVIDIA API Key", type="password")
    uploaded_files = st.file_uploader(
        "Upload sustainability PDFs",
        accept_multiple_files=True,
        type=["pdf"],
    )

    with st.expander("Chunking settings (for large documents)", expanded=False):
        chunk_size = st.slider("Chunk size (characters)", 500, 2500, 1200, step=100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 150, step=25)
        top_k = st.slider("Retriever top-k", 2, 10, 4)

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("Build / Rebuild Index", type="primary")
    with col2:
        clear_btn = st.button("Clear Session")

    if clear_btn:
        st.session_state.vs = None
        st.session_state.docs = []
        st.session_state.messages = []
        st.rerun()

    # Build index
    if build_btn:
        if not nvidia_key:
            st.error("Please enter your NVIDIA API key.")
            st.stop()
        if not uploaded_files:
            st.error("Please upload at least 1 PDF.")
            st.stop()

        os.environ["NVIDIA_API_KEY"] = nvidia_key

        all_docs: list[Document] = []
        progress = st.progress(0, text="Extracting + chunking PDFs...")

        for i, file in enumerate(uploaded_files, start=1):
            pdf_bytes = file.read()

            # REAL extraction + chunking (safe for large PDFs)
            chunked_docs = extract_pdf_to_chunked_docs(
                pdf_bytes=pdf_bytes,
                source_name=file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_docs.extend(chunked_docs)

            progress.progress(i / len(uploaded_files), text=f"Processed: {file.name}")

        progress.empty()

        if not all_docs:
            st.error("No extractable text found in uploaded PDFs.")
            st.stop()

        st.session_state.docs = all_docs
        st.session_state.vs = build_vector_store(all_docs)

        st.success(
            f"Indexed {len(all_docs)} chunks from {len(uploaded_files)} PDF(s). You can now chat below."
        )

    # Show chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    st.json(m["sources"])

    # Chat input (only if index exists)
    if st.session_state.vs is None:
        st.info("Upload PDFs and click **Build / Rebuild Index** to start chatting.")
        return

    if not nvidia_key:
        st.warning("Enter NVIDIA API key to chat.")
        return

    os.environ["NVIDIA_API_KEY"] = nvidia_key

    llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

    user_q = st.chat_input("Ask about allocations, impact KPIs, eligibility, reporting gaps...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        retriever = st.session_state.vs.as_retriever(search_kwargs={"k": top_k})
        retrieved = retriever.get_relevant_documents(user_q)

        answer = answer_question(llm, user_q, retrieved)
        sources = format_sources(retrieved)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()


if __name__ == "__main__":
    main()
