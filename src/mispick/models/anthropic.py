"""Anthropic, optional. Needs ANTHROPIC_API_KEY and the `anthropic` extra.

The Messages API is not a chat-completions dialect, so this is a separate implementation
rather than a base-URL swap. It also does not take a seed, and says so - reports must not
promise determinism we cannot deliver.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mispick.models.base import (
    DEFAULT_MAX_TOKENS,
    Backend,
    BackendError,
    Selection,
    build_tool_payload,
)
from mispick.models.ollama import SYSTEM_PROMPT
from mispick.types import Tool

if TYPE_CHECKING:
    from anthropic.types import ToolParam

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicBackend(Backend):
    #: The Messages API takes neither `seed` nor `temperature` (nor `top_p`). Passing either
    #: is a TypeError, not a silently ignored argument. Runs against this backend lean on
    #: repeated trials (K) and the confidence intervals instead of on sampling controls.
    supports_seed = False
    supports_temperature = False

    def __init__(self, model: str = DEFAULT_MODEL, *, timeout: float = 120.0) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise BackendError(
                "ANTHROPIC_API_KEY is not set. Either export it, or use the default local "
                "backend: --model ollama/qwen3.5:4b"
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise BackendError(
                "The `anthropic` package is not installed. Run: uv sync --extra anthropic"
            ) from exc
        self.model = model
        self.name = f"anthropic/{model}"
        self._client = AsyncAnthropic(api_key=key, timeout=timeout)

    @staticmethod
    def _tools(tools: list[Tool]) -> list[ToolParam]:
        """Messages API shape: name / description / input_schema.

        Built as plain dict literals rather than by calling `ToolParam(...)`: it is a
        TypedDict, so the name only exists for the type checker, and the SDK is imported
        lazily so that a missing `anthropic` extra produces a readable error instead of an
        ImportError at module load.
        """
        payload: list[ToolParam] = []
        for entry in build_tool_payload(tools):
            payload.append(
                {
                    "name": entry["function"]["name"],
                    "description": entry["function"]["description"],
                    "input_schema": entry["function"]["parameters"],
                }
            )
        return payload

    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
                tools=self._tools(tools),
            )
        except Exception as exc:
            return Selection(error=f"{type(exc).__name__}: {exc}")

        text_parts: list[str] = []
        for block in response.content:
            # `block.type` is a discriminator, so matching on it narrows the union -
            # getattr does not, and a response can carry a dozen other block kinds.
            if block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                return Selection(
                    chosen=str(block.name),
                    arguments=dict(arguments),
                    raw=" ".join(text_parts),
                )
            if block.type == "text":
                text_parts.append(block.text)
        return Selection(chosen=None, raw=" ".join(text_parts).strip() or "(no tool call)")

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        chunks = [b.text for b in response.content if b.type == "text"]
        return "".join(chunks).strip()

    async def aclose(self) -> None:
        await self._client.close()
