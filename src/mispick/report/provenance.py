"""The header every report must carry.

SPEC section 7: results depend on the model, so every report states the model, N, K and the
date, and says so out loud.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from mispick.select import RunResult


@dataclass
class Provenance:
    model: str
    n: int
    k: int
    date: str
    temperature: float
    seed: int | None
    servers: str
    tool_count: int
    deterministic: bool
    token_estimate: int = 0
    supports_temperature: bool = True
    supports_seed: bool = True

    @property
    def one_line(self) -> str:
        bits = [
            f"model {self.model}",
            f"N={self.n}",
            f"K={self.k}",
        ]
        # Only report the knobs the backend actually honours. Printing
        # "temperature=0" for a provider that accepts no temperature would be a
        # statement about this run that is not true.
        if self.supports_temperature:
            bits.append(f"temperature={self.temperature:g}")
        else:
            bits.append("temperature=n/a")
        if self.seed is not None:
            # Say "ignored" rather than nothing: the user asked for a seed, and silence
            # would leave them believing it took effect.
            bits.append(f"seed={self.seed}" if self.supports_seed else f"seed={self.seed}-ignored")
        bits.append(self.date)
        return " · ".join(bits)

    #: The honesty line. Not optional, and not softened.
    caveat = (
        "Results depend on the model. A different model, or the same model at a different "
        "temperature, will produce different numbers. These are measurements of one model's "
        "behaviour on this tool list, not a property of the server."
    )


def provenance_of(result: RunResult, token_estimate: int = 0) -> Provenance:
    config = result.config
    servers = ", ".join(
        f"{info.name or info.label}{' ' + info.version if info.version else ''}"
        for info in result.tool_set.servers
    )
    return Provenance(
        model=config.model,
        n=config.n,
        k=config.k,
        date=dt.date.today().isoformat(),
        temperature=config.temperature,
        seed=config.seed,
        servers=servers or "unknown",
        tool_count=len(result.tools),
        deterministic=config.deterministic,
        token_estimate=token_estimate,
        supports_temperature=config.supports_temperature,
        supports_seed=config.supports_seed,
    )
