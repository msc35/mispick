"""Any OpenAI-compatible endpoint, including Gemini.

A lot of providers speak the OpenAI chat-completions dialect, so one backend covers Gemini,
vLLM, OpenRouter, Together, LM Studio and anything else behind a base URL. The selection call
is the same shape everywhere: real tool definitions on the request, a `tool_calls` reply.

What differs is which sampling controls are honoured, and reports must not claim one that was
not sent - see `supports_seed` and `supports_temperature` on `Backend`.
"""

from __future__ import annotations

import os

from mispick.models.base import BackendError
from mispick.models.ollama import OpenAICompatibleBackend

#: Gemini's OpenAI-compatible endpoint.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

#: The cheapest Gemini model Google documents as supporting function calling.
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiBackend(OpenAICompatibleBackend):
    """Gemini, through its OpenAI-compatible endpoint.

    `seed` is declared unsupported. The compatibility layer accepts the field but Google does
    not document it as honoured, and silently ignoring a seed while a report says
    "reproducible" would be worse than admitting it. Temperature is supported.
    """

    supports_seed = False
    supports_temperature = True

    def __init__(self, model: str = GEMINI_DEFAULT_MODEL, *, timeout: float = 180.0) -> None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise BackendError(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. Export one, or use the "
                "default local backend: --model ollama/qwen3.5:4b"
            )
        super().__init__(
            model,
            base_url=GEMINI_BASE_URL,
            api_key=key,
            name=f"gemini/{model}",
            timeout=timeout,
            # Gemini rejects unknown request fields, and its own thinking control is not
            # `reasoning_effort`, so send nothing extra.
            extra_body=None,
        )


class CustomBackend(OpenAICompatibleBackend):
    """An arbitrary OpenAI-compatible endpoint, configured by environment.

    MISPICK_BASE_URL is required; MISPICK_API_KEY is sent if set. For a local server such as
    vLLM or LM Studio no key is usually needed.
    """

    supports_seed = True
    supports_temperature = True

    def __init__(self, model: str, *, timeout: float = 180.0) -> None:
        base_url = os.environ.get("MISPICK_BASE_URL")
        if not base_url:
            raise BackendError(
                "MISPICK_BASE_URL is not set. Point it at an OpenAI-compatible endpoint, "
                "e.g. MISPICK_BASE_URL=http://localhost:8000/v1 "
                "--model compatible/my-model"
            )
        super().__init__(
            model,
            base_url=base_url,
            api_key=os.environ.get("MISPICK_API_KEY", "unused"),
            name=f"compatible/{model}",
            timeout=timeout,
        )
