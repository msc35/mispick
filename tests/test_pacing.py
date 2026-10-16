"""Free-tier runs: pacing, quota stops, and resuming without repeating paid calls."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mispick.generate import GenerationPlan, generate_for_tool
from mispick.models import ollama
from mispick.models.base import Backend, QuotaExhaustedError, Selection
from mispick.models.mock import MockBackend
from mispick.models.pacing import PacedBackend, RateLimiter
from mispick.select import RunConfig, run_selection
from mispick.types import Query, Tool, ToolSet

ROOT = Path(__file__).resolve().parent.parent


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestRateLimiter:
    def test_requests_per_minute(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(rpm=2, clock=clock, sleep=clock.sleep)

        async def go() -> None:
            for _ in range(3):
                await limiter.acquire()

        asyncio.run(go())
        assert clock.now == pytest.approx(60.0)

    def test_tokens_per_minute(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(tpm=100, clock=clock, sleep=clock.sleep)

        async def go() -> None:
            await limiter.acquire(60)
            clock.now = 10.0
            await limiter.acquire(30)
            await limiter.acquire(50)  # 140 > 100: waits for the first to expire at t=60

        asyncio.run(go())
        assert clock.now == pytest.approx(60.0)

    def test_a_request_bigger_than_the_budget_still_goes_once_the_window_is_empty(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(tpm=100, clock=clock, sleep=clock.sleep)

        async def go() -> None:
            await limiter.acquire(10)
            await limiter.acquire(500)

        asyncio.run(go())
        assert clock.now == pytest.approx(60.0)

    def test_no_limits_never_waits(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(clock=clock, sleep=clock.sleep)

        async def go() -> None:
            for _ in range(100):
                await limiter.acquire(10_000)

        asyncio.run(go())
        assert clock.slept == []


class TestCallBudget:
    def test_stops_with_a_quota_error_not_a_miss(self) -> None:
        backend = PacedBackend(MockBackend(), max_calls=2)
        tools = [Tool(name="search", description="search things")]

        async def go() -> None:
            await backend.choose(tools, "search")
            await backend.choose(tools, "search")
            await backend.choose(tools, "search")

        with pytest.raises(QuotaExhaustedError, match="budget of 2"):
            asyncio.run(go())
        assert backend.calls == 2

    def test_reports_the_wrapped_backend(self) -> None:
        backend = PacedBackend(MockBackend())
        assert backend.name == "mock"
        assert backend.supports_seed is True


def _error(status: int, message: str, retry_after: str | None = None) -> Exception:
    exc = Exception(message)
    exc.status_code = status  # type: ignore[attr-defined]
    headers = {"retry-after": retry_after} if retry_after else {}
    exc.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
    return exc


GEMINI_DAILY = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota', "
    "'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '41s'}]}}]"
)
GEMINI_MINUTE = (
    "Error code: 429 - [{'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, {'retryDelay': '17s'}]}}]"
)


class TestRateLimitClassification:
    def test_daily_quota(self) -> None:
        assert ollama.rate_limit_kind(_error(429, GEMINI_DAILY)) == "daily"

    def test_per_minute(self) -> None:
        assert ollama.rate_limit_kind(_error(429, GEMINI_MINUTE)) == "minute"

    def test_openai_out_of_credit_is_not_worth_retrying(self) -> None:
        assert ollama.rate_limit_kind(_error(429, "insufficient_quota")) == "daily"

    def test_other_errors_are_not_rate_limits(self) -> None:
        assert ollama.rate_limit_kind(_error(500, "boom")) is None
        assert ollama.rate_limit_kind(ValueError("nope")) is None

    def test_delay_from_the_body(self) -> None:
        assert ollama.retry_delay(_error(429, GEMINI_MINUTE), 0) == pytest.approx(18.0)

    def test_delay_from_the_header_wins(self) -> None:
        assert ollama.retry_delay(_error(429, GEMINI_MINUTE, "3"), 0) == pytest.approx(4.0)

    def test_fallback_backs_off(self) -> None:
        assert ollama.retry_delay(_error(429, "slow down"), 0) == 5
        assert ollama.retry_delay(_error(429, "slow down"), 9) == 60


class _FlakyCompletions:
    def __init__(self, failures: list[Exception]) -> None:
        self.failures = list(failures)
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        call = SimpleNamespace(function=SimpleNamespace(name="search", arguments="{}"))
        message = SimpleNamespace(content="", tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _backend_with(
    failures: list[Exception], monkeypatch: pytest.MonkeyPatch
) -> tuple[ollama.OpenAICompatibleBackend, _FlakyCompletions]:
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(ollama, "_sleep", no_sleep)
    backend = ollama.OpenAICompatibleBackend("m", base_url="http://unused", name="t")
    completions = _FlakyCompletions(failures)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return backend, completions


class TestBackendRateLimits:
    def test_waits_out_a_per_minute_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, completions = _backend_with(
            [_error(429, GEMINI_MINUTE), _error(429, GEMINI_MINUTE)], monkeypatch
        )
        selection = asyncio.run(backend.choose([Tool(name="search")], "find it"))
        assert selection.chosen == "search"
        assert completions.calls == 3

    def test_a_daily_quota_propagates_from_choose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, completions = _backend_with([_error(429, GEMINI_DAILY)], monkeypatch)
        with pytest.raises(QuotaExhaustedError, match="daily quota"):
            asyncio.run(backend.choose([Tool(name="search")], "find it"))
        assert completions.calls == 1

    def test_a_daily_quota_propagates_from_generate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, _ = _backend_with([_error(429, GEMINI_DAILY)], monkeypatch)
        with pytest.raises(QuotaExhaustedError):
            asyncio.run(backend.generate("write queries"))

    def test_gives_up_after_the_retries_as_an_error_not_a_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failures = [_error(429, GEMINI_MINUTE)] * (ollama.RATE_LIMIT_RETRIES + 1)
        backend, _ = _backend_with(failures, monkeypatch)
        selection = asyncio.run(backend.choose([Tool(name="search")], "find it"))
        assert selection.error
        assert selection.chosen is None


class _QuotaAfter(Backend):
    """Answers `allowed` times, then reports a used-up daily quota."""

    name = "quota"

    def __init__(self, allowed: int) -> None:
        self.allowed = allowed
        self.calls = 0

    async def choose(self, tools: list[Tool], query: str, **_: Any) -> Selection:
        self.calls += 1
        if self.calls > self.allowed:
            raise QuotaExhaustedError("daily quota")
        return Selection(chosen=tools[0].name)

    async def generate(self, prompt: str, **_: Any) -> str:
        raise QuotaExhaustedError("daily quota")


def _small_run() -> tuple[ToolSet, list[Query]]:
    tool_set = ToolSet(tools=[Tool(name="a", description="a"), Tool(name="b", description="b")])
    queries = [Query(id=f"q{i}", text=f"query {i}", expected="a") for i in range(4)]
    return tool_set, queries


class TestResume:
    def test_completed_trials_are_not_asked_again(self) -> None:
        tool_set, queries = _small_run()
        config = RunConfig(k=2, concurrency=1)
        first = asyncio.run(run_selection(MockBackend(), tool_set, queries, config=config))
        assert len(first.choices) == 8

        backend = MockBackend()
        again = asyncio.run(
            run_selection(
                backend, tool_set, queries, config=RunConfig(k=2), completed=first.choices[:5]
            )
        )
        assert backend.calls == 3
        assert sorted((c.query_id, c.run) for c in again.choices) == sorted(
            (c.query_id, c.run) for c in first.choices
        )

    def test_errored_trials_are_retried(self) -> None:
        tool_set, queries = _small_run()
        first = asyncio.run(
            run_selection(MockBackend(), tool_set, queries, config=RunConfig(k=1))
        )
        broken = [c.model_copy(update={"error": "timeout"}) for c in first.choices]
        backend = MockBackend()
        asyncio.run(
            run_selection(backend, tool_set, queries, config=RunConfig(k=1), completed=broken)
        )
        assert backend.calls == 4

    def test_a_quota_stop_raises_instead_of_recording_misses(self) -> None:
        tool_set, queries = _small_run()
        seen: list[Any] = []
        with pytest.raises(QuotaExhaustedError):
            asyncio.run(
                run_selection(
                    _QuotaAfter(3),
                    tool_set,
                    queries,
                    config=RunConfig(k=2, concurrency=1),
                    on_progress=seen.append,
                )
            )
        assert len(seen) == 3
        assert all(c.chosen == "a" and not c.error for c in seen)

    def test_query_generation_does_not_swallow_a_quota_stop(self) -> None:
        tool = Tool(name="a", description="a")
        with pytest.raises(QuotaExhaustedError):
            asyncio.run(
                generate_for_tool(_QuotaAfter(0), tool, [tool], plan=GenerationPlan.for_n(3))
            )


def _measure_module() -> Any:
    sys.path.insert(0, str(ROOT / "benchmark"))
    try:
        return importlib.import_module("measure")
    finally:
        sys.path.pop(0)


class TestMeasureResumes:
    def _args(self, snapshots: Path, out: Path, budget: int) -> Any:
        return SimpleNamespace(
            snapshots=snapshots,
            out=out,
            model="mock",
            queries=4,
            runs=3,
            temperature=0.0,
            seed=1,
            concurrency=2,
            max_tools=120,
            only=None,
            rerun=False,
            rpm=0,
            tpm=0,
            budget_calls=budget,
        )

    def test_a_budget_stop_then_a_second_run_finishes_without_repeating_calls(
        self, tmp_path: Path, snapshot_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        measure = _measure_module()
        snapshots = tmp_path / "snapshots"
        snapshots.mkdir()
        shutil.copy(snapshot_path, snapshots / "demo.json")
        out = tmp_path / "results"

        assert asyncio.run(measure.main_async(self._args(snapshots, out, budget=20))) == 0
        first = capsys.readouterr().out
        assert "Stopped early" in first
        assert not (out / "demo.json").exists()
        index = json.loads((out / "index.json").read_text())
        assert index["complete"] is False
        assert index["pending"] == ["demo"]
        saved = (out / ".trials" / "demo.jsonl").read_text().splitlines()
        assert saved

        assert asyncio.run(measure.main_async(self._args(snapshots, out, budget=0))) == 0
        second = capsys.readouterr().out
        assert "All servers measured." in second
        result = json.loads((out / "demo.json").read_text())
        trials = len(result["trials"])
        index = json.loads((out / "index.json").read_text())
        assert index["complete"] is True
        calls = int(re.search(r"model calls this run: (\d+)", second).group(1))  # type: ignore[union-attr]
        assert trials == len(result["queries"]) * 3
        # Queries were cached by the first run, so only the missing trials were paid for.
        assert calls == trials - len(saved)

        # A third run spends nothing.
        assert asyncio.run(measure.main_async(self._args(snapshots, out, budget=0))) == 0
        assert "model calls this run: 0" in capsys.readouterr().out

    def test_saved_trials_are_dropped_when_the_queries_change(
        self, tmp_path: Path, snapshot_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ids survive regeneration but the text does not; old answers must not be reused."""
        measure = _measure_module()
        snapshots = tmp_path / "snapshots"
        snapshots.mkdir()
        shutil.copy(snapshot_path, snapshots / "demo.json")
        out = tmp_path / "results"
        assert asyncio.run(measure.main_async(self._args(snapshots, out, budget=20))) == 0
        assert (out / ".trials" / "demo.jsonl").read_text()
        capsys.readouterr()

        cache = next((out / ".queries" / "demo").glob("*.yaml"))
        cache.write_text(cache.read_text().replace("text: ", "text: Please ", 1))
        assert asyncio.run(measure.main_async(self._args(snapshots, out, budget=0))) == 0
        result = json.loads((out / "demo.json").read_text())
        calls = int(re.search(r"model calls this run: (\d+)", capsys.readouterr().out).group(1))  # type: ignore[union-attr]
        assert calls >= len(result["trials"])


class TestTransientFailures:
    def test_overloaded_and_dropped_connections_are_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dropped = type("APIConnectionError", (Exception,), {})("Connection error.")
        backend, completions = _backend_with(
            [_error(503, "This model is currently experiencing high demand"), dropped],
            monkeypatch,
        )
        selection = asyncio.run(backend.choose([Tool(name="search")], "find it"))
        assert selection.chosen == "search"
        assert completions.calls == 3

    def test_a_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, completions = _backend_with([_error(400, "Unknown name seed")], monkeypatch)
        selection = asyncio.run(backend.choose([Tool(name="search")], "find it"))
        assert selection.error
        assert completions.calls == 1

    def test_a_remote_connection_failure_does_not_mention_ollama(self) -> None:
        backend = ollama.OpenAICompatibleBackend(
            "m", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        message = backend._explain(type("APIConnectionError", (Exception,), {})("Connection"))
        assert "Ollama" not in message
