from pathlib import Path

from src.core.config import load_config
from src.core.rag_engine import RAGEngine


def test_engine_init(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = load_config()
    engine = RAGEngine(config, api_key="test-key")
    assert engine is not None



