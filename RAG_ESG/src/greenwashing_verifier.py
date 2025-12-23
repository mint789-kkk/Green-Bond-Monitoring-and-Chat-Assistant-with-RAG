"""Dual-layer greenwashing detection (Green Cognition + Implementation check)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import Document


class GreenwashingVerifier:
    """
    Layer A: Detection of green claims via lightweight keyword/context heuristics ("Green Cognition").
    Layer B: Verification of whether claims are implemented vs empty promises (LLM reasoning).
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
        # Placeholder: use 4-bit quantized local LLM if available. For now, rely on GPT-4 API.
        self.llm = OpenAI(openai_api_key=openai_api_key, model_name=model, temperature=temperature)
        self._verification_chain = self._build_verification_chain()

    def detect_claims(self, documents: List[Document]) -> List[Tuple[Document, str]]:
        """
        Layer A detection: flag snippets containing sustainability claims.
        Lightweight heuristic using green keywords; can be replaced with CLIP embeddings.
        """
        green_terms = [
            "green",
            "renewable",
            "solar",
            "wind",
            "energy efficiency",
            "emissions",
            "carbon",
            "sustainable",
            "impact",
            "sdg",
            "taxonomy",
        ]
        hits = []
        for doc in documents:
            text_lower = doc.page_content.lower()
            if any(term in text_lower for term in green_terms):
                hits.append((doc, "claim_detected"))
        return hits

    def verify_claims(self, claims: List[Tuple[Document, str]]) -> List[dict]:
        """
        Layer B verification: judge whether claims show evidence of implementation.
        Returns a list of dicts with verdict and greenImplement score.
        """
        results = []
        for doc, _ in claims:
            verdict = self._verification_chain.run(text=doc.page_content)
            results.append(
                {
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "verdict": verdict,
                }
            )
        return results

    def green_implement_ratio(self, verified: List[dict]) -> float:
        """Compute GreenImplement ratio: implemented / total."""
        if not verified:
            return 0.0
        implemented = sum(1 for v in verified if "implemented" in v["verdict"].lower())
        return implemented / len(verified)

    def _build_verification_chain(self) -> LLMChain:
        template = """
You are the GreenwashingVerifier. Decide if the claim shows evidence of IMPLEMENTATION.
Answer with one of: "implemented", "empty", or "unclear".

Text:
{text}
"""
        prompt = PromptTemplate(template=template, input_variables=["text"])
        return LLMChain(llm=self.llm, prompt=prompt)



