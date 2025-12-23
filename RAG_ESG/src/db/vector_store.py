"""FAISS-backed vector store with CLIP embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.vectorstores import FAISS
import faiss

from src.models.embedding import EmbeddingModel


class VectorStore:
    def __init__(self, index_path: Path, embedding_model: EmbeddingModel | None = None):
        self.index_path = index_path
        self.embedding_model = embedding_model or EmbeddingModel()
        self._store: FAISS | None = None
        self._bm25: BM25Retriever | None = None
        self._documents: List[Document] = []

    def load_or_create(self, documents: List[Document]) -> FAISS:
        if self.index_path.exists():
            self._store = FAISS.load_local(str(self.index_path), self.embedding_model)
            if documents:
                self.add_documents(documents)
        elif documents:
            self._store = FAISS.from_documents(documents, self.embedding_model)
            self._documents.extend(documents)
            self.save()
        if self._documents and not self._bm25:
            self._bm25 = BM25Retriever.from_documents(self._documents)
        return self._store

    def save(self) -> None:
        if self._store:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_path))

    def add_documents(self, documents: List[Document]) -> None:
        if not self._store:
            self._store = FAISS.from_documents(documents, self.embedding_model)
        else:
            self._store.add_documents(documents)
        self._documents.extend(documents)
        if self._documents:
            self._bm25 = BM25Retriever.from_documents(self._documents)
        self.save()

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self._store:
            raise RuntimeError("Vector store not initialized")
        docs_and_scores = self._store.similarity_search_with_score(query, k=k)
        return docs_and_scores

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.6) -> List[Tuple[Document, float]]:
        """
        Hybrid retrieval inspired by ChatPDF:
        alpha * embedding_score + (1-alpha) * bm25_score (normalized rank-based).
        """
        if not self._store:
            raise RuntimeError("Vector store not initialized")
        vector_hits = self._store.similarity_search_with_score(query, k=k * 2)
        bm25_hits: List[Document] = self._bm25.get_relevant_documents(query) if self._bm25 else []

        combined: Dict[str, Tuple[Document, float]] = {}

        def doc_key(doc: Document) -> str:
            return f"{doc.metadata.get('source','')}-{doc.metadata.get('page','')}-{hash(doc.page_content)}"

        # Normalize vector scores (lower is better in FAISS distances) by inverse rank.
        for rank, (doc, score) in enumerate(vector_hits):
            key = doc_key(doc)
            inv_rank_score = 1 / (1 + rank)
            combined[key] = (doc, alpha * inv_rank_score)

        # BM25 gives relevance in order; assign descending rank-based score.
        for rank, doc in enumerate(bm25_hits):
            key = doc_key(doc)
            bm25_score = 1 / (1 + rank)
            if key in combined:
                combined[key] = (combined[key][0], combined[key][1] + (1 - alpha) * bm25_score)
            else:
                combined[key] = (doc, (1 - alpha) * bm25_score)

        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]



