"""LLM interface with optional quantization placeholder."""

from __future__ import annotations

from typing import Any, Dict

from langchain.llms import OpenAI


class LLMModel:
    """
    Wraps the reasoning engine. For now uses GPT-4 API; swap with local 4-bit model as needed.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self._llm = OpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)

    def generate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        return self._llm(prompt, **kwargs)



