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


class TestGemini:
    """Gemini via its OpenAI-compatible endpoint. No key needed for any of this."""

    def test_resolves_and_points_at_the_compat_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        backend = get_backend("gemini/gemini-3.5-flash-lite")
        assert backend.name == "gemini/gemini-3.5-flash-lite"
        assert "generativelanguage.googleapis.com" in (backend.base_url or "")

    def test_accepts_either_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        assert get_backend("gemini/x").name == "gemini/x"

    def test_declares_seed_unsupported_but_temperature_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compat layer accepts `seed` but Google does not document honouring it."""
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        backend = get_backend("gemini/x")
        assert backend.supports_seed is False
        assert backend.supports_temperature is True

    def test_sends_no_extra_request_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gemini rejects unknown fields, and reasoning_effort is not its thinking control."""
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        assert get_backend("gemini/x").extra_body == {}

    def test_missing_key_names_both_variables_and_the_free_alternative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mispick.models.base import BackendError

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(BackendError) as excinfo:
            get_backend("gemini/x")
        message = str(excinfo.value)
        assert "GEMINI_API_KEY" in message and "GOOGLE_API_KEY" in message
        assert "ollama" in message


class TestCustomCompatibleEndpoint:
    def test_uses_the_configured_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISPICK_BASE_URL", "http://localhost:8000/v1")
        backend = get_backend("compatible/my-model")
        assert backend.base_url == "http://localhost:8000/v1"
        assert backend.name == "compatible/my-model"

    def test_openai_compatible_is_an_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISPICK_BASE_URL", "http://localhost:8000/v1")
        assert get_backend("openai-compatible/m").name == "compatible/m"

    def test_no_base_url_says_what_to_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mispick.models.base import BackendError

        monkeypatch.delenv("MISPICK_BASE_URL", raising=False)
        with pytest.raises(BackendError, match="MISPICK_BASE_URL"):
            get_backend("compatible/m")


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


class TestTruncationDetection:
    """A truncated tool list produces a real-looking score that blames the wrong thing.

    Ollama silently clips a prompt that exceeds the context window: send 24,000 tokens of tool
    definitions into a 4,096-token context and it reports 2,050 prompt tokens, replies "I do
    not have access to tools", and every trial records "chose nothing". The confusion matrix
    then blames the server's descriptions for a limit of our own configuration.
    """

    @staticmethod
    def _response(prompt_tokens: int):
        class Usage:
            def __init__(self, n: int) -> None:
                self.prompt_tokens = n

        class Response:
            def __init__(self, n: int) -> None:
                self.usage = Usage(n)

        return Response(prompt_tokens)

    def _tools(self, count: int, description_length: int = 400):
        from mispick.types import Tool

        return [
            Tool(
                name=f"tool_{i}",
                description="x" * description_length,
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
            for i in range(count)
        ]

    def test_fires_when_far_fewer_tokens_were_processed(self) -> None:
        from mispick.metrics import estimate_tool_tokens
        from mispick.models.ollama import _truncation_error

        tools = self._tools(64)
        estimate = estimate_tool_tokens(tools)
        error = _truncation_error(self._response(2050), tools)
        assert error is not None
        assert "2050" in error
        assert str(estimate) in error
        assert "truncated" in error
        # It must say what to do about it.
        assert "OLLAMA_CONTEXT_LENGTH" in error
        assert "OLLAMA_NUM_PARALLEL" in error

    def test_silent_when_the_whole_prompt_was_processed(self) -> None:
        from mispick.metrics import estimate_tool_tokens
        from mispick.models.ollama import _truncation_error

        tools = self._tools(9)
        estimate = estimate_tool_tokens(tools)
        assert _truncation_error(self._response(estimate + 200), tools) is None

    def test_tolerates_a_tokenizer_counting_slightly_fewer(self) -> None:
        """chars/4 is an estimate; a real tokenizer may land a little under it."""
        from mispick.metrics import estimate_tool_tokens
        from mispick.models.ollama import _truncation_error

        tools = self._tools(9)
        estimate = estimate_tool_tokens(tools)
        assert _truncation_error(self._response(int(estimate * 0.9)), tools) is None

    def test_silent_when_usage_is_not_reported(self) -> None:
        from mispick.models.ollama import _truncation_error

        class Bare:
            usage = None

        assert _truncation_error(Bare(), self._tools(9)) is None

    def test_silent_for_an_empty_tool_list(self) -> None:
        from mispick.models.ollama import _truncation_error

        assert _truncation_error(self._response(10), []) is None

    def test_the_estimator_takes_tools_directly(self) -> None:
        """So it can be called before a run exists, e.g. mid-request."""
        from mispick.metrics import estimate_tool_tokens

        assert estimate_tool_tokens(self._tools(4)) > 0
        assert estimate_tool_tokens([]) == 0
