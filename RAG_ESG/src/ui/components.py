"""Tkinter UI components for DeskRAG."""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Callable, List


class FileSelector(ttk.Frame):
    def __init__(self, master, on_files_selected: Callable[[List[Path]], None]):
        super().__init__(master)
        self.on_files_selected = on_files_selected
        self.button = ttk.Button(self, text="Add files", command=self.open_dialog)
        self.button.pack(fill="x", padx=4, pady=4)

    def open_dialog(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Documents", "*.pdf *.txt *.md *.markdown *.png *.jpg *.jpeg")]
        )
        if file_paths:
            self.on_files_selected([Path(p) for p in file_paths])


class ChatBox(ttk.Frame):
    def __init__(self, master, on_send: Callable[[str], None]):
        super().__init__(master)
        self.on_send = on_send
        self.text = tk.Text(self, height=15, wrap="word")
        self.entry = ttk.Entry(self)
        self.send = ttk.Button(self, text="Ask", command=self._send)
        self.text.pack(fill="both", expand=True, padx=4, pady=4)
        self.entry.pack(fill="x", padx=4)
        self.send.pack(padx=4, pady=4)

    def _send(self):
        question = self.entry.get()
        if question:
            self.on_send(question)
            self.entry.delete(0, tk.END)

    def append(self, message: str):
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)



