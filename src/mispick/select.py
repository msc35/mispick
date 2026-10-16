"""Run the selection step: for each query, ask the model to pick a tool.

The model sees the full tool list, as a real client would. We run each query K times with
temperature above zero so stability is measurable - a tool that wins three times out of
three is a different finding from one that wins two.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from mispick.models.base import Backend, from_wire_name
from mispick.types import Choice, Query, Tool, ToolSet

DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.7
#: How many selection calls to have in flight at once.
DEFAULT_CONCURRENCY = 4


@dataclass
class RunConfig:
    """Everything that affects a number in the report, so we can print it."""

    model: str = "unknown"
    n: int = 8
    k: int = DEFAULT_K
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    supports_seed: bool = False
    supports_temperature: bool = True

    @property
    def deterministic(self) -> bool:
        return self.temperature == 0.0 and (self.seed is None or self.supports_seed)


@dataclass
class RunResult:
    """Raw choices plus the inputs that produced them."""

    tool_set: ToolSet
    queries: list[Query]
    choices: list[Choice] = field(default_factory=list)
    config: RunConfig = field(default_factory=RunConfig)
    errors: list[str] = field(default_factory=list)

    @property
    def tools(self) -> list[Tool]:
        return self.tool_set.sorted_tools()


def validate_arguments(tool: Tool, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check the model's arguments against the tool's inputSchema.

    Uses Draft 2020-12, which is the MCP default, with a registry-free resolver so
    `$ref`/`$defs` inside the schema work. A schema we cannot compile is not the model's
    fault, so that counts as valid rather than penalising the model for a server's bug.
    """
    schema = tool.input_schema or {"type": "object"}
    try:
        validator = Draft202012Validator(schema)
        # An unknown "type" keyword raises here rather than above, so both are guarded.
        errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.path))
    except jsonschema.exceptions.SchemaError:
        return True, []
    except jsonschema.exceptions.UnknownType:
        return True, []
    if not errors:
        return True, []
    messages = []
    for err in errors[:5]:
        where = ".".join(str(p) for p in err.path) or "(root)"
        messages.append(f"{where}: {err.message}")
    return False, messages


async def _one(
    backend: Backend,
    tools: list[Tool],
    query: Query,
    run: int,
    config: RunConfig,
    by_name: dict[str, Tool],
    semaphore: asyncio.Semaphore,
) -> Choice:
    async with semaphore:
        seed = None if config.seed is None else config.seed + run
        selection = await backend.choose(
            tools, query.text, temperature=config.temperature, seed=seed
        )

    if selection.error:
        return Choice(query_id=query.id, chosen=None, run=run, error=selection.error)

    if selection.chosen is None:
        return Choice(query_id=query.id, chosen=None, run=run)

    resolved = from_wire_name(selection.chosen, tools)
    if resolved is None:
        # The model named a tool that does not exist.
        return Choice(
            query_id=query.id,
            chosen=None,
            phantom=selection.chosen,
            run=run,
            arguments=selection.arguments,
        )

    tool = by_name[resolved]
    ok, problems = validate_arguments(tool, selection.arguments)
    return Choice(
        query_id=query.id,
        chosen=resolved,
        arguments=selection.arguments,
        args_valid=ok,
        args_errors=problems,
        run=run,
    )


async def run_selection(
    backend: Backend,
    tool_set: ToolSet,
    queries: list[Query],
    *,
    config: RunConfig | None = None,
    on_progress: Any = None,
    completed: Iterable[Choice] = (),
) -> RunResult:
    """Ask the model to choose, for every query, K times.

    `completed` holds trials from an earlier, interrupted run of the same configuration;
    they are kept and not asked again.
    """
    config = config or RunConfig()
    config.model = backend.name
    config.supports_seed = backend.supports_seed
    config.supports_temperature = backend.supports_temperature
    tools = tool_set.sorted_tools()
    by_name = tool_set.by_qualified_name()
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    # Trials already recorded by an interrupted run are reused, not repeated. Errored ones
    # are retried: an error is not a measurement.
    wanted = {(query.id, run) for query in queries for run in range(config.k)}
    done = {
        (c.query_id, c.run): c
        for c in completed
        if not c.error and (c.query_id, c.run) in wanted
    }
    choices: list[Choice] = list(done.values())
    pending = [
        asyncio.ensure_future(_one(backend, tools, query, run, config, by_name, semaphore))
        for query in queries
        for run in range(config.k)
        if (query.id, run) not in done
    ]

    try:
        for future in asyncio.as_completed(pending):
            choice = await future
            choices.append(choice)
            if on_progress:
                on_progress(choice)
    except BaseException:
        # A used-up quota stops every trial, not just the one that saw it.
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise

    errors = sorted({c.error for c in choices if c.error})
    return RunResult(
        tool_set=tool_set, queries=queries, choices=choices, config=config, errors=errors
    )
