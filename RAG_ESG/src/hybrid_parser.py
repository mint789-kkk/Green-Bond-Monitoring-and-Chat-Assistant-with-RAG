"""Hybrid parser to link table cells with surrounding narrative context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class LinkedCell:
    """Represents a table cell with its contextual narrative snippet."""

    value: str
    row_header: Optional[str]
    column_header: Optional[str]
    normalized_unit: Optional[str]
    page_number: Optional[int]
    context_snippet: str
    source_document: str


class HybridParser:
    """
    Extracts tables and binds each cell to nearby narrative paragraphs.

    Strategy:
    - Extract tables and plain text separately.
    - Normalize units (tCO2e, MW, GWh, m3) at extraction time.
    - Associate each cell with the most relevant narrative chunk on the same page.
    """

    def __init__(self, text_splitter: Optional[RecursiveCharacterTextSplitter] = None):
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120
        )

    def parse_document(self, document: Document) -> List[LinkedCell]:
        """
        Parses a LangChain Document containing PDF text + metadata.
        Expects metadata to include 'page' and 'source'.
        """
        # Placeholder: in production, replace with PDF table extractor (camelot/tabula)
        tables = document.metadata.get("tables", [])
        narrative_chunks = self._split_text(document)

        linked_cells: List[LinkedCell] = []
        for table in tables:
            for cell in table.get("cells", []):
                linked = self._link_cell(cell, narrative_chunks, document)
                if linked:
                    linked_cells.append(linked)
        return linked_cells

    def _split_text(self, document: Document) -> List[Document]:
        """Split text into manageable chunks for contextual lookup."""
        return self.text_splitter.split_documents([document])

    def _link_cell(self, cell: dict, narrative_chunks: List[Document], document: Document) -> Optional[LinkedCell]:
        """Link a table cell to the closest narrative snippet (same page preference)."""
        page = cell.get("page", document.metadata.get("page"))
        source = document.metadata.get("source", "unknown")

        # Prefer chunks on the same page; otherwise fall back to nearest chunk.
        same_page = [c for c in narrative_chunks if c.metadata.get("page") == page]
        target_chunk = (same_page or narrative_chunks or [Document(page_content="")])[0]

        return LinkedCell(
            value=str(cell.get("value", "")).strip(),
            row_header=cell.get("row_header"),
            column_header=cell.get("column_header"),
            normalized_unit=self._normalize_unit(cell.get("unit")),
            page_number=page,
            context_snippet=target_chunk.page_content,
            source_document=source,
        )

    @staticmethod
    def _normalize_unit(raw_unit: Optional[str]) -> Optional[str]:
        """Normalize units to the canonical set: tCO2e, MW, GWh, m3."""
        if not raw_unit:
            return None
        unit = raw_unit.lower().replace(" ", "")
        mapping = {
            "tco2": "tCO2e",
            "tco2e": "tCO2e",
            "co2e": "tCO2e",
            "mw": "MW",
            "gwh": "GWh",
            "m3": "m3",
            "m^3": "m3",
        }
        return mapping.get(unit, raw_unit)



