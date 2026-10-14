"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Package: llm
=============================================================================
Local Ollama LLM integration for patient-friendly explanation generation.
All inference runs locally — no cloud APIs, no API keys required.
"""

from llm.ollama_client import OllamaClient, OllamaStatus
from llm.evidence_builder import build_explanation_evidence
from llm.explanation_renderer import generate_explanation, render_fallback_explanation

__all__ = [
    "OllamaClient",
    "OllamaStatus",
    "build_explanation_evidence",
    "generate_explanation",
    "render_fallback_explanation",
]
