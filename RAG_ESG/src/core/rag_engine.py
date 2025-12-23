"""RAG engine coordinating processing, embedding, storage, and querying."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.schema import Document

from src.core.config import AppConfig
from src.db.vector_store import VectorStore
from src.models.embedding import EmbeddingModel
from src.models.llm import LLMModel
from src.processors.document import DocumentProcessor
from src.schemas import StandardizedBondInformationCard, AuditTrail
from src.greenwashing_verifier import GreenwashingVerifier
from src.hybrid_parser import HybridParser


class RAGEngine:
    def __init__(self, config: AppConfig, api_key: str):
        self.config = config
        self.processor = DocumentProcessor()
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(config.cache_dir / "faiss.index", self.embedding_model)
        self.llm = LLMModel(api_key=api_key)
        self.verifier = GreenwashingVerifier(openai_api_key=api_key)
        self.parser = HybridParser()

    def ingest(self, paths: List[Path]) -> None:
        docs: List[Document] = []
        for path in paths:
            docs.extend(self.processor.process_file(path))
        if self.vector_store._store:
            self.vector_store.add_documents(docs)
        else:
            self.vector_store.load_or_create(docs)

    def query(self, question: str) -> StandardizedBondInformationCard:
        if not self.vector_store._store:
            self.vector_store.load_or_create([])
        if not self.vector_store._store:
            raise RuntimeError("No documents ingested; please add files before querying.")

        results = self.vector_store.search_hybrid(question, k=5)
        docs = [doc for doc, _ in results]

        # Greenwashing check on retrieved docs
        detections = self.verifier.detect_claims(docs)
        verified = self.verifier.verify_claims(detections)
        green_ratio = self.verifier.green_implement_ratio(verified)

        # Basic card with audit trail from retrieved docs
        audit = [
            AuditTrail(
                source_document=doc.metadata.get("source", ""),
                page_number=doc.metadata.get("page"),
                snippet=doc.page_content[:500],
            )
            for doc in docs
        ]

        return StandardizedBondInformationCard(
            issuer=None,
            objective=None,
            taxonomy_category=None,
            kpis=[],
            greenwashing_score=green_ratio,
            audit_trail=audit,
        )



