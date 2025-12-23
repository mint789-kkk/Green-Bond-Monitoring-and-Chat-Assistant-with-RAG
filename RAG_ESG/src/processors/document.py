"""Document processor: load PDFs/DOCX/TXT/MD/images and chunk for embeddings."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        if suffix in {".txt"}:
            return self._load_text(path)
        if suffix in {".md", ".markdown"}:
            return self._load_markdown(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._load_image(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

    def _load_pdf(self, path: Path) -> List[Document]:
        loader = PyPDFLoader(str(path))
        return self._split(loader.load())

    def _load_text(self, path: Path) -> List[Document]:
        loader = TextLoader(str(path), autodetect_encoding=True)
        return self._split(loader.load())

    def _load_markdown(self, path: Path) -> List[Document]:
        loader = UnstructuredMarkdownLoader(str(path))
        return self._split(loader.load())

    def _load_docx(self, path: Path) -> List[Document]:
        loader = Docx2txtLoader(str(path))
        return self._split(loader.load())

    def _load_image(self, path: Path) -> List[Document]:
        # Placeholder: OCR or CLIP image embedding goes here.
        with open(path, "rb") as f:
            image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = f"[image:{path.name} size={image.size}]"
        return self._split([Document(page_content=text, metadata={"source": str(path)})])



