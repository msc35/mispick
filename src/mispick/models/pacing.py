"""Stay under a provider's rate limits, and stop at a call budget.

Free tiers cap requests per minute, tokens per minute and requests per day. Hitting the
per-minute caps costs retries; hitting the daily cap mid-server wastes the calls already made
unless the caller can stop cleanly. `PacedBackend` paces ahead of the limits instead of
reacting to 429s, and raises `QuotaExhaustedError` once its own call budget is spent.
"""

from __future__ import annotations

import asyncio
import collections
import time
from collections.abc import Awaitable, Callable

from mispick.metrics import estimate_tool_tokens
from mispick.models.base import (
    DEFAULT_MAX_TOKENS,
    Backend,
    QuotaExhaustedError,
    Selection,
)
from mispick.types import Tool

WINDOW_SECONDS = 60.0


class RateLimiter:
    """A sliding one-minute window over requests and tokens."""

    def __init__(
        self,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.rpm = rpm or None
        self.tpm = tpm or None
        self._clock = clock
        self._sleep = sleep
        self._events: collections.deque[tuple[float, int]] = collections.deque()
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _wait_needed(self, now: float, tokens: int) -> float:
        """Seconds until a request of this size fits in the window, or 0."""
        self._expire(now)
        waits = [0.0]
        if self.rpm and len(self._events) >= self.rpm:
            waits.append(self._events[0][0] + WINDOW_SECONDS - now)
        if self.tpm and self._events:
            used = sum(t for _, t in self._events)
            # A single request larger than the whole budget can only wait for an empty window.
            for stamp, spent in self._events:
                if used + tokens <= self.tpm:
                    break
                used -= spent
                waits.append(stamp + WINDOW_SECONDS - now)
        return max(waits)

    async def acquire(self, tokens: int = 0) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                wait = self._wait_needed(now, tokens)
                if wait <= 0:
                    self._events.append((now, tokens))
                    return
                await self._sleep(wait)


class PacedBackend(Backend):
    """Wraps any backend with a rate limiter and an optional total call budget."""

    def __init__(
        self,
        inner: Backend,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        max_calls: int | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.supports_seed = inner.supports_seed
        self.supports_temperature = inner.supports_temperature
        self.limiter = limiter or RateLimiter(rpm=rpm, tpm=tpm)
        self.max_calls = max_calls or None
        self.calls = 0

    async def _admit(self, tokens: int) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise QuotaExhaustedError(
                f"Call budget of {self.max_calls} reached. Progress is saved; run again to "
                "continue."
            )
        self.calls += 1
        await self.limiter.acquire(tokens)

    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        await self._admit(estimate_tool_tokens(tools) + len(query) // 4 + 100)
        return await self.inner.choose(tools, query, temperature=temperature, seed=seed)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        await self._admit(len(prompt) // 4)
        return await self.inner.generate(
            prompt, temperature=temperature, seed=seed, max_tokens=max_tokens
        )

    async def aclose(self) -> None:
        await self.inner.aclose()
