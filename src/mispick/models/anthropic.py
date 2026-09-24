"""Anthropic, optional. Needs ANTHROPIC_API_KEY and the `anthropic` extra.

The Messages API is not a chat-completions dialect, so this is a separate implementation
rather than a base-URL swap. It also does not take a seed, and says so - reports must not
promise determinism we cannot deliver.
"""

from __future__ import annotations

import os
from typing import Any

from mispick.models.base import Backend, BackendError, Selection, build_tool_payload
from mispick.models.ollama import SYSTEM_PROMPT
from mispick.types import Tool

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicBackend(Backend):
    supports_seed = False

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
    def _tools(tools: list[Tool]) -> list[dict[str, Any]]:
        """Messages API shape: name / description / input_schema."""
        return [
            {
                "name": entry["function"]["name"],
                "description": entry["function"]["description"],
                "input_schema": entry["function"]["parameters"],
            }
            for entry in build_tool_payload(tools)
        ]

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
                temperature=temperature,
            )
        except Exception as exc:
            return Selection(error=f"{type(exc).__name__}: {exc}")

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                arguments = raw_input if isinstance(raw_input, dict) else {}
                return Selection(
                    chosen=str(block.name),
                    arguments=dict(arguments),
                    raw=" ".join(text_parts),
                )
            if getattr(block, "type", None) == "text":
                text_parts.append(str(getattr(block, "text", "")))
        return Selection(chosen=None, raw=" ".join(text_parts).strip() or "(no tool call)")

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> str:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        chunks = [
            str(getattr(b, "text", ""))
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        return "".join(chunks).strip()

    async def aclose(self) -> None:
        await self._client.close()
