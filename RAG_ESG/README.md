# DeskRAG (GreenBond-RAG flavor)

Local-first RAG tailored for sustainability/green bond analytics. It ingests hybrid PDFs (tables + narrative), stores CLIP embeddings in FAISS, and orchestrates LLM responses using a standardized bond information card with page-level audit trails.

## Structure
```
deskrag/
├── src/
│   ├── core/
│   ├── processors/
│   ├── models/
│   ├── db/
│   └── ui/
├── scripts/
├── tests/
└── requirements.txt
```

## Quick start
- Install: `pip install -r requirements.txt`
- Run UI: `python -m src.ui.app`
- Download models: `python scripts/download_models.py`



