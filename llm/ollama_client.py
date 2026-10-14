"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: llm.ollama_client
=============================================================================
Thin wrapper around the Ollama local REST API (http://localhost:11434).

Design principles:
- No API keys, no cloud endpoints, no paid services.
- Configurable model name; default is qwen2.5:3b.
- Health-check + model-presence check before generation.
- Hard timeout on every request (default 120 s).
- Bounded output length via options.num_predict.
- Clear status enum so callers can show meaningful error messages.
- Never logs prompts that contain health/patient data.
"""

from __future__ import annotations

import json
import enum
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL_NAME: str = "qwen2.5:3b"
REQUEST_TIMEOUT_SECONDS: int = 120  # per-call wall-clock timeout
MAX_TOKENS_OUT: int = 1024          # bounded output length
HEALTH_TIMEOUT_SECONDS: int = 5     # fast check


class OllamaStatus(str, enum.Enum):
    """Status codes returned by every public method."""
    OK = "ok"
    OLLAMA_NOT_RUNNING = "ollama_not_running"
    MODEL_NOT_FOUND = "model_not_found"
    GENERATION_FAILED = "generation_failed"
    TIMEOUT = "timeout"
    RESPONSE_EMPTY = "response_empty"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Minimal Ollama API client for local LLM text generation.

    Usage::

        client = OllamaClient(model_name="qwen2.5:3b")
        status, text = client.generate(prompt, system_prompt)
        if status == OllamaStatus.OK:
            # use text
        else:
            # show error / fallback
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_tokens: int = MAX_TOKENS_OUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check_health(self) -> OllamaStatus:
        """
        Returns OllamaStatus.OK if Ollama is reachable, else OLLAMA_NOT_RUNNING.
        Fast check (5 s timeout).
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
            if resp.status_code == 200:
                return OllamaStatus.OK
            return OllamaStatus.OLLAMA_NOT_RUNNING
        except (requests.ConnectionError, requests.Timeout):
            return OllamaStatus.OLLAMA_NOT_RUNNING

    def check_model(self) -> OllamaStatus:
        """
        Returns OllamaStatus.OK if the configured model is installed locally.
        Assumes Ollama is already running (call check_health first).
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                return OllamaStatus.OLLAMA_NOT_RUNNING

            data = resp.json()
            installed = [m.get("name", "") for m in data.get("models", [])]
            # Match with or without the tag suffix (:latest etc.)
            model_base = self.model_name.split(":")[0].lower()
            for name in installed:
                if name.lower().startswith(model_base):
                    return OllamaStatus.OK
            return OllamaStatus.MODEL_NOT_FOUND
        except (requests.ConnectionError, requests.Timeout):
            return OllamaStatus.OLLAMA_NOT_RUNNING
        except Exception:
            return OllamaStatus.OLLAMA_NOT_RUNNING

    def get_installed_models(self) -> list[str]:
        """Returns list of installed model names, empty list on any error."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
            if resp.status_code == 200:
                return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[OllamaStatus, str]:
        """
        Sends a generation request to Ollama and returns (status, text).

        The system_prompt scopes the model's role. Never put raw image
        pixel data or patient-identifying information in these strings.

        Returns:
            (OllamaStatus.OK, generated_text) on success.
            (OllamaStatus.<error_code>, "") on any failure.
        """
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.3,   # lower = more deterministic / less hallucination
                "top_p": 0.9,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout:
            return OllamaStatus.TIMEOUT, ""
        except (requests.ConnectionError, requests.RequestException):
            return OllamaStatus.OLLAMA_NOT_RUNNING, ""

        if resp.status_code != 200:
            return OllamaStatus.GENERATION_FAILED, ""

        try:
            data = resp.json()
            text = data.get("response", "").strip()
            if not text:
                return OllamaStatus.RESPONSE_EMPTY, ""
            return OllamaStatus.OK, text
        except (json.JSONDecodeError, KeyError):
            return OllamaStatus.GENERATION_FAILED, ""

    # ------------------------------------------------------------------
    # Status human-readable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def status_message(status: OllamaStatus) -> str:
        """Returns a user-friendly message for each status code."""
        messages = {
            OllamaStatus.OK: "Ollama is running and the model is available.",
            OllamaStatus.OLLAMA_NOT_RUNNING: (
                "Ollama is not running. Start it with `ollama serve` in a terminal."
            ),
            OllamaStatus.MODEL_NOT_FOUND: (
                "The configured model is not installed. "
                "Run `ollama pull <model_name>` to install it."
            ),
            OllamaStatus.GENERATION_FAILED: (
                "Ollama returned an error during generation."
            ),
            OllamaStatus.TIMEOUT: (
                "Ollama did not respond within the time limit. "
                "The model may be too large for your hardware, or the system is under load."
            ),
            OllamaStatus.RESPONSE_EMPTY: (
                "Ollama returned an empty response."
            ),
        }
        return messages.get(status, "Unknown Ollama status.")
