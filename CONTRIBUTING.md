# Contributing

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy src
```

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
