"""OpenAI, optional. Needs OPENAI_API_KEY."""

from __future__ import annotations

import os

from mispick.models.base import BackendError
from mispick.models.ollama import OpenAICompatibleBackend

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIBackend(OpenAICompatibleBackend):
    def __init__(self, model: str = DEFAULT_MODEL, *, timeout: float = 120.0) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BackendError(
                "OPENAI_API_KEY is not set. Either export it, or use the default local "
                "backend: --model ollama/qwen3.5:4b"
            )
        super().__init__(model, base_url=None, api_key=key, name=f"openai/{model}", timeout=timeout)
