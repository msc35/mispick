# Committed tool-list snapshots

One `<slug>.json` per server, plus `index.json` recording what happened to every candidate —
including the ones that were skipped and why.

**This directory is empty on purpose.** Populating it runs `capture.py`, which launches other
people's MCP servers locally through `npx` / `uvx`. That downloads and executes third-party code,
so it is a decision for whoever runs it rather than something committed blind. Run it yourself:

```bash
uv run python benchmark/pick.py --limit 30
uv run python benchmark/capture.py
```

or let the `benchmark` workflow do it on a runner with `refresh-snapshots: true`.

Capture only ever sends `initialize` and `tools/list` — it cannot call a tool. That is enforced
in `mispick.sources.server` and tested in `tests/test_never_calls_tools.py` and
`tests/test_benchmark.py`.
