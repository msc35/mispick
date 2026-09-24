"""Backend contracts, including the optional ones the offline suite never calls.

The Anthropic backend was fully broken for a while and nothing caught it: it passed
`temperature` to the Messages API, which accepts neither `temperature` nor `top_p`, so every
call would have raised TypeError. `uv sync` without `--all-extras` leaves `anthropic`
uninstalled and mypy is configured to ignore its imports when missing, so the local loop was
blind to it. These tests exercise the parts that need no API key.
"""

from __future__ import annotations

import inspect

import pytest

from mispick.models.base import Backend
from mispick.models.mock import MockBackend, ScriptedBackend
from mispick.models.ollama import OllamaBackend, OpenAICompatibleBackend
from mispick.models.registry import DEFAULT_MODEL, available_backends, get_backend
from mispick.types import ToolSet


def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class TestCapabilityFlags:
    """A report must not claim a sampling control the provider never received."""

    def test_base_defaults_to_supporting_temperature(self) -> None:
        assert Backend.supports_temperature is True

    def test_ollama_supports_both_knobs(self) -> None:
        assert OpenAICompatibleBackend.supports_seed is True
        assert OpenAICompatibleBackend.supports_temperature is True

    @pytest.mark.skipif(not _anthropic_available(), reason="anthropic extra not installed")
    def test_anthropic_declares_both_unsupported(self) -> None:
        from mispick.models.anthropic import AnthropicBackend

        assert AnthropicBackend.supports_seed is False
        assert AnthropicBackend.supports_temperature is False


@pytest.mark.skipif(not _anthropic_available(), reason="anthropic extra not installed")
class TestAnthropicWithoutAKey:
    """Everything here runs without ANTHROPIC_API_KEY and without a network call."""

    def test_never_passes_temperature_or_seed_to_the_messages_api(self) -> None:
        """The API rejects both. Passing either is a TypeError on every call."""
        from anthropic.resources.messages import AsyncMessages

        accepted = set(inspect.signature(AsyncMessages.create).parameters)
        assert "temperature" not in accepted, "SDK changed; revisit supports_temperature"
        assert "seed" not in accepted

        from mispick.models import anthropic as backend_module

        source = inspect.getsource(backend_module.AnthropicBackend)
        # The parameters exist on our own method signatures to satisfy the Backend
        # interface, but must never reach messages.create.
        for call in ("temperature=temperature", "seed=seed"):
            assert call not in source, f"{call} must not be forwarded to the Messages API"

    def test_tool_payload_builds_at_runtime(self, tool_set: ToolSet) -> None:
        """ToolParam is a TypedDict imported only for typing; calling it would NameError."""
        from mispick.models.anthropic import AnthropicBackend

        payload = AnthropicBackend._tools(tool_set.sorted_tools())
        assert len(payload) == len(tool_set.tools)
        for entry in payload:
            assert set(entry) == {"name", "description", "input_schema"}
            assert isinstance(entry["input_schema"], dict)

    def test_missing_key_explains_the_alternative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mispick.models.anthropic import AnthropicBackend
        from mispick.models.base import BackendError

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
            AnthropicBackend()


class TestOpenAIWithoutAKey:
    def test_missing_key_explains_the_alternative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mispick.models.base import BackendError
        from mispick.models.openai import OpenAIBackend

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(BackendError, match="OPENAI_API_KEY"):
            OpenAIBackend()


class TestRegistry:
    def test_default_is_a_local_ollama_model(self) -> None:
        assert DEFAULT_MODEL.startswith("ollama/")

    def test_bare_name_means_ollama(self) -> None:
        backend = get_backend("qwen3.5:4b")
        assert isinstance(backend, OllamaBackend)
        assert backend.name == "ollama/qwen3.5:4b"

    def test_mock_is_reachable_by_name(self) -> None:
        assert isinstance(get_backend("mock"), MockBackend)

    def test_unknown_provider_lists_the_real_ones(self) -> None:
        from mispick.models.base import BackendError

        with pytest.raises(BackendError, match="Unknown model provider"):
            get_backend("hotdog/wat")

    def test_every_advertised_backend_name_resolves_or_explains(self) -> None:
        """A name in the help text must not produce a confusing error."""
        from mispick.models.base import BackendError

        for name in available_backends():
            try:
                get_backend(f"{name}/some-model")
            except BackendError as exc:
                # Only ever because a key or extra is missing, never "unknown provider".
                assert "Unknown model provider" not in str(exc), name

    def test_thinking_is_off_by_default_for_ollama(self) -> None:
        """Measured 89s -> 2.2s per call with it off, for the same answers."""
        assert get_backend("ollama/qwen3.5:4b").extra_body == {"reasoning_effort": "none"}
        assert get_backend("ollama/qwen3.5:4b", think=True).extra_body == {}


class TestInterfaceCompleteness:
    @pytest.mark.parametrize("cls", [MockBackend, ScriptedBackend, OpenAICompatibleBackend])
    def test_implements_the_interface(self, cls: type) -> None:
        for method in ("choose", "generate", "aclose"):
            assert callable(getattr(cls, method, None)), f"{cls.__name__} lacks {method}"

    def test_anthropic_implements_the_interface(self) -> None:
        if not _anthropic_available():
            pytest.skip("anthropic extra not installed")
        from mispick.models.anthropic import AnthropicBackend

        for method in ("choose", "generate", "aclose"):
            assert callable(getattr(AnthropicBackend, method, None))
