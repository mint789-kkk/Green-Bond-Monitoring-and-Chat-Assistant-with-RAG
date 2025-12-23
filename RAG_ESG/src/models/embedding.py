"""CLIP-based embedding interface."""

from __future__ import annotations

from typing import List

from langchain.embeddings import HuggingFaceInstructEmbeddings


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        # Placeholder for 8-bit quantized CLIP; using HF instruct embeddings wrapper.
        self.model_name = model_name
        self._model = HuggingFaceInstructEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)



