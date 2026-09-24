"""Turn a `--model` string into a backend.

Accepted forms:
  qwen3.5:4b              -> Ollama (the default backend)
  ollama/qwen3.5:4b       -> Ollama, explicit
  openai/gpt-4.1-mini     -> OpenAI, needs OPENAI_API_KEY
  anthropic/claude-...    -> Anthropic, needs ANTHROPIC_API_KEY and the extra
  mock                    -> the offline deterministic backend
"""

from __future__ import annotations

from mispick.models.base import Backend, BackendError

#: The default when the user says nothing at all.
DEFAULT_MODEL = "ollama/qwen3.5:4b"


def available_backends() -> list[str]:
    return ["ollama", "openai", "anthropic", "mock"]


def get_backend(spec: str | None = None, *, jitter: bool = True) -> Backend:
    """Build a backend from a `provider/model` string."""
    spec = (spec or DEFAULT_MODEL).strip()
    if not spec:
        raise BackendError("Empty --model.")

    provider, _, model = spec.partition("/")
    if not model:
        # Bare model name: Ollama, unless it names the mock.
        if provider == "mock":
            from mispick.models.mock import MockBackend

            return MockBackend(jitter=jitter)
        from mispick.models.ollama import OllamaBackend

        return OllamaBackend(provider)

    provider = provider.lower()
    if provider == "ollama":
        from mispick.models.ollama import OllamaBackend

        return OllamaBackend(model)
    if provider == "openai":
        from mispick.models.openai import OpenAIBackend

        return OpenAIBackend(model)
    if provider == "anthropic":
        from mispick.models.anthropic import AnthropicBackend

        return AnthropicBackend(model)
    if provider == "mock":
        from mispick.models.mock import MockBackend

        return MockBackend(jitter=jitter)
    raise BackendError(
        f"Unknown model provider {provider!r}. Use one of: {', '.join(available_backends())}.\n"
        "Examples: qwen3.5:4b (local, the default) | openai/gpt-4.1-mini | "
        "anthropic/claude-haiku-4-5-20251001"
    )
