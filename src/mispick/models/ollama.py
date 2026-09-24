"""The default backend: a local model through Ollama's OpenAI-compatible endpoint.

Free, offline, no account. Ollama supports `seed` and `temperature` on
`/v1/chat/completions`, so SPEC section 12's deterministic mode is achievable here.
"""

from __future__ import annotations

import json
from typing import Any

from mispick.models.base import (
    DEFAULT_MAX_TOKENS,
    Backend,
    BackendError,
    Selection,
    build_tool_payload,
)
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
        timeout: float = 600.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise BackendError("The `openai` package is required. Run: uv sync") from exc
        self.model = model
        self.name = name or model
        self.base_url = base_url
        #: Extra request fields. Dropped automatically if the endpoint rejects them.
        self.extra_body = dict(extra_body or {})
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """Send a request, retrying once without our extra fields if they are refused.

        `reasoning_effort` is the one that matters: Ollama honours it and it is worth a 40x
        speedup on a small reasoning model, but not every OpenAI-compatible endpoint accepts
        it, and a hard failure there would be a bad trade.
        """
        if self.extra_body:
            try:
                return await self._client.chat.completions.create(
                    **kwargs, extra_body=self.extra_body
                )
            except Exception as exc:
                if not _looks_like_rejected_field(exc):
                    raise
                self.extra_body = {}
        return await self._client.chat.completions.create(**kwargs)

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
            response = await self._create(kwargs)
        except Exception as exc:
            return Selection(error=self._explain(exc))

        if not response.choices:
            return Selection(error="the model returned no choices")

        truncation = _truncation_error(response, tools)
        if truncation:
            return Selection(error=truncation)

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
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
            response = await self._create(kwargs)
        except Exception as exc:
            raise BackendError(self._explain(exc)) from exc
        if not response.choices:
            raise BackendError("the model returned no choices")
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise BackendError(self._explain_empty(choice, max_tokens))
        return text

    @staticmethod
    def _explain_empty(choice: Any, max_tokens: int) -> str:
        """Say why a reply was blank, which is nearly always the reasoning budget."""
        reasoning = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )
        if choice.finish_reason == "length":
            extra = (
                " It spent the whole budget on hidden reasoning tokens before writing "
                "anything visible."
                if reasoning
                else ""
            )
            return (
                f"the model hit the {max_tokens}-token limit without producing any visible "
                f"output.{extra} Raise it with --max-tokens, or pick a model that does less "
                "thinking."
            )
        return (
            "the model returned an empty reply. If it is a reasoning model, it may need a "
            "larger --max-tokens budget."
        )

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


#: Flag truncation when the endpoint reports processing less than this fraction of what we
#: estimated sending. Our estimate is characters/4, which normally *under*-counts real tokens,
#: so a reported figure well below it means content was dropped rather than tokenised
#: differently.
TRUNCATION_RATIO = 0.75


def _truncation_error(response: Any, tools: list[Tool]) -> str | None:
    """Detect a tool list the model never actually saw.

    Ollama silently truncates a prompt that exceeds the context window: send 24,000 tokens of
    tool definitions into a 4,096-token context and it reports `prompt_tokens: 2050`, answers
    "I don't have access to tools", and every trial records "chose nothing". The confusion
    matrix then blames the server's descriptions for a limit of our own configuration - which
    is worse than failing, because the number looks real.

    Rather than ask each provider for its context length, compare what we sent against the
    `prompt_tokens` the API itself reports. That works for any provider and needs no table of
    model limits to go stale.
    """
    usage = getattr(response, "usage", None)
    reported = getattr(usage, "prompt_tokens", None)
    if not reported:
        return None

    from mispick.metrics import estimate_tool_tokens

    estimate = estimate_tool_tokens(tools)
    if estimate <= 0 or reported >= estimate * TRUNCATION_RATIO:
        return None
    return (
        f"the model processed only {reported} prompt tokens, but this tool list is roughly "
        f"{estimate}. The tool list was truncated, so the model never saw most of the tools "
        "and any result would be meaningless. Raise the model's context window - for Ollama, "
        "`OLLAMA_CONTEXT_LENGTH` or a Modelfile with a larger `num_ctx`, and note that "
        "`OLLAMA_NUM_PARALLEL` divides the context between concurrent requests - or measure a "
        "server with fewer tools."
    )


def _looks_like_rejected_field(exc: Exception) -> bool:
    """Whether an error reads like "I do not know that request field"."""
    text = str(exc).lower()
    if "400" not in text and "unsupported" not in text and "unknown" not in text:
        return False
    return any(
        word in text for word in ("reasoning", "unknown field", "unsupported", "unrecognized")
    )


class OllamaBackend(OpenAICompatibleBackend):
    """Ollama, via its OpenAI-compatible endpoint.

    Thinking is off by default. Most tool-capable models that fit on a laptop are reasoning
    models, and on this workload their hidden reasoning is pure cost: measured on
    qwen3.5:4b, one generation call took 89s and 1078 completion tokens with thinking on
    versus 2.2s and 20 tokens with it off, for the same answer. Worse, thinking sometimes
    spirals and consumes the entire output budget, producing an empty reply. Pass
    `think=True` to turn it back on.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
        think: bool = False,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key="ollama",
            name=f"ollama/{model}",
            timeout=timeout,
            extra_body=None if think else {"reasoning_effort": "none"},
        )
