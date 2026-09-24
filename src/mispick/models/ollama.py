"""The default backend: a local model through Ollama's OpenAI-compatible endpoint.

Free, offline, no account. Ollama supports `seed` and `temperature` on
`/v1/chat/completions`, so SPEC section 12's deterministic mode is achievable here.
"""

from __future__ import annotations

import json
from typing import Any

from mispick.models.base import Backend, BackendError, Selection, build_tool_payload
from mispick.types import Tool

#: Ollama's OpenAI-compatible base URL.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: A small, current, tool-capable model. See docs/research.md section 6 - the current
#: generation of tool-tagged models skews large, so this is a deliberate pick.
DEFAULT_MODEL = "qwen3.5:4b"

SYSTEM_PROMPT = (
    "You are the tool-routing layer of an application. Given the user's request, call "
    "exactly one of the available tools. Choose the single best tool. If genuinely no tool "
    "fits the request, reply with the word NONE and call nothing. Never explain your choice."
)


class OpenAICompatibleBackend(Backend):
    """Shared implementation for any OpenAI-compatible chat endpoint."""

    supports_seed = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str = "unused",
        name: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise BackendError("The `openai` package is required. Run: uv sync") from exc
        self.model = model
        self.name = name or model
        self.base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "tools": build_tool_payload(tools),
            "tool_choice": "auto",
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            return Selection(error=self._explain(exc))

        if not response.choices:
            return Selection(error="the model returned no choices")
        message = response.choices[0].message
        text = (message.content or "").strip()
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return Selection(chosen=None, raw=text or "(no tool call)")

        call = calls[0]
        raw_args = getattr(call.function, "arguments", "") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            # Invalid JSON is itself a finding: the choice stands, the arguments do not.
            arguments = {}
            text = f"{text} [unparseable arguments: {raw_args[:120]}]".strip()
        if not isinstance(arguments, dict):
            arguments = {}
        return Selection(chosen=call.function.name, arguments=arguments, raw=text)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise BackendError(self._explain(exc)) from exc
        if not response.choices:
            raise BackendError("the model returned no choices")
        return (response.choices[0].message.content or "").strip()

    def _explain(self, exc: Exception) -> str:
        """Turn a client error into something that says what to do next."""
        text = str(exc)
        if self.base_url and ("Connection" in type(exc).__name__ or "connect" in text.lower()):
            return (
                f"could not reach the model backend at {self.base_url}. "
                "Is Ollama running? Start it with `ollama serve`, then "
                f"`ollama pull {self.model}`."
            )
        if "not found" in text.lower() and self.base_url:
            return f"model {self.model!r} is not installed. Run: ollama pull {self.model}"
        return f"{type(exc).__name__}: {text}"

    async def aclose(self) -> None:
        await self._client.close()


class OllamaBackend(OpenAICompatibleBackend):
    """Ollama, via its OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key="ollama",
            name=f"ollama/{model}",
            timeout=timeout,
        )
