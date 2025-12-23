"""Tkinter application entrypoint for DeskRAG."""

import tkinter as tk
from pathlib import Path

from src.core.config import load_config
from src.core.rag_engine import RAGEngine
from src.ui.components import FileSelector, ChatBox


class DeskRAGApp(tk.Tk):
    def __init__(self, api_key: str):
        super().__init__()
        self.title("DeskRAG - GreenBond RAG")
        self.geometry("720x640")

        self.config_obj = load_config()
        self.engine = RAGEngine(self.config_obj, api_key=api_key)

        self.file_selector = FileSelector(self, on_files_selected=self.on_files_selected)
        self.file_selector.pack(fill="x")

        self.chat = ChatBox(self, on_send=self.on_send)
        self.chat.pack(fill="both", expand=True)

    def on_files_selected(self, paths: list[Path]):
        self.chat.append(f"Ingesting {len(paths)} file(s)...")
        self.engine.ingest(paths)
        self.chat.append("Ingestion complete.")

    def on_send(self, question: str):
        self.chat.append(f"Q: {question}")
        card = self.engine.query(question)
        self.chat.append(f"A (greenwashing_score={card.greenwashing_score}):")
        for audit in card.audit_trail:
            self.chat.append(f"- {audit.source_document} p{audit.page_number}: {audit.snippet[:160]}")


def main():
    # Expect API key via environment or user prompt; placeholder here.
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before launching the app.")

    app = DeskRAGApp(api_key=api_key)
    app.mainloop()


if __name__ == "__main__":
    main()



