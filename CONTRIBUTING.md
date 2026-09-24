# Contributing

```bash
uv sync --all-extras          # --all-extras matters, see below
uv run pytest
uv run ruff check . && uv run mypy src
```

**Always `--all-extras`.** Plain `uv sync` leaves the optional `anthropic` dependency out, and
mypy is configured to ignore its imports when absent — so `models/anthropic.py` goes unchecked
and nothing tells you. That is exactly how a backend that passed `temperature` to an API which
accepts no `temperature`, and would have raised `TypeError` on every call, survived a green
local run. CI uses `--all-extras` and caught it on the first push.

Workflows are linted too; `ruff` does not read them:

```bash
actionlint                    # brew install actionlint
```

A workflow with two `env` keys that differ only in case is rejected by GitHub wholesale, and the
failure arrives as a zero-second run with no job and no log.

## Things the tests enforce, so a reviewer does not have to

**mispick never calls a tool.** `tests/test_never_calls_tools.py` fails the build if any module
under `src/mispick/` so much as references `call_tool` or the string `tools/call`, if anything
returned by `sources/` exposes a client, or if a tool load succeeds with `Client.call_tool`
monkeypatched to raise. This is the promise that makes mispick safe to point at a destructive
server. If you need to change it, you need a different tool.

**The suite runs offline.** No network, no Ollama, no API keys. Use `--model mock`. CI re-runs the
suite with the key environment variables blanked to keep this true.

**Every report states the model, N, K and the date**, and carries the caveat that results depend on
the model. There are tests for this in every format, including one that renders the terminal report
at 52 columns to make sure the date is not truncated.

**Unmeasured rates are `null`, not `0`.** A K=1 run has no stability; reporting 0% would be a lie.

**Benchmark wording stays neutral.** "Most confused pair", never "worst server". Tested.

## Style

Python 3.11+, type hints everywhere, pydantic for data, small modules. Error messages say what to
do next, not just what went wrong. `ruff` and `mypy --strict` clean.

## Adding a model backend

Implement `mispick.models.base.Backend` (`choose` and `generate`), register it in
`models/registry.py`, and set `supports_seed` honestly — reports use it to decide whether to claim
a run is reproducible.
