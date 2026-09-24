# mispick

**Find out which of your MCP tools the model mixes up — measured, not guessed.**

Point it at any MCP server. It generates realistic user requests, asks a model to pick a tool for
each, and shows a **confusion matrix**: which tools get picked for which intent. Then it suggests
description rewrites for confused pairs and re-runs to prove the fix improved accuracy.

It never calls a tool. Only `initialize` and `tools/list`, so it is safe to point at any server.

## Status

**Pre-alpha, under active construction.** M0–M3 are done: the pipeline works and reports in
every format.

- [`SPEC.md`](SPEC.md) — the source of truth for what this is
- [`docs/research.md`](docs/research.md) — Phase 0 research: competitors, naming, MCP spec
  `2026-07-28`, the Python `mcp` 2.2.0 client API, the registry API, Ollama models, and the
  description-quality literature

Working today:

```bash
uv sync

# list what a server exposes, and any cross-server name collisions
uv run mispick tools --snapshot tests/fixtures/confusing_tools.json
uv run mispick tools --cmd "python -m tests.fixtures.fixture_server"
uv run mispick tools --config tests/fixtures/two_servers.json

# capture a tools/list so later runs work offline
uv run mispick snapshot --cmd "python -m my_server" -o tools.json

# measure which tools the model mixes up
uv run mispick run --cmd "python -m my_server"
uv run mispick run --snapshot tools.json --model mock   # offline, no model needed

# reports: terminal (default), json, md for a PR comment, self-contained html
uv run mispick run --snapshot tools.json --format html --out report.html --badge badge.svg
uv run mispick run --snapshot tools.json --fail-under 80   # exit 1 for CI
```

`run` generates test queries per tool, caches them in `.mispick/queries.yaml` (edit them — they
are your test set), asks the model to pick a tool for each K times, and prints a confusion
matrix, the worst pairs, per-tool precision and recall with Wilson 95% intervals, and a 0–100
score. Every report states the model, N, K and the date, because the results depend on all four.

The default backend is a local Ollama model, so it costs nothing and needs no account.
`--model mock` is a deterministic offline stand-in used by the test suite.

Measured on the fixture server with `ollama/qwen3.5:4b`: the two tools that share the
description "Search for information." get confused 42% of the time, and `search_docs` ends up
with 35% precision — it attracts work belonging to four other tools. That is the finding a
static linter cannot give you.

**On local reasoning models.** Most tool-capable models that fit on a laptop think before
answering, and on this workload the thinking is pure cost: on `qwen3.5:4b` one generation call
took 89s with thinking on versus 2.2s with it off, for the same answer — and left to itself it
sometimes spends the entire output budget thinking and returns nothing. mispick therefore turns
thinking off by default. `--think` restores it.

Install instructions, the demo GIF, and the benchmark link land with M8 and M7 per SPEC
section 11.

### Why this vs the static linters

`mcp-lint`, `mcpolish`, `oh-my-mcp`, `mcp-conform` and `mcp-cve-lint` check tool definitions
against rules, and they are good at it. A description can pass every rule and still be confused
with the tool next to it. Two newer projects do measure selection with a real model — `whichtool`
(local models, no rewrites, one server per run) and `toolfit` (rewrites, but cloud-only). Neither
looks across servers. See [docs/research.md](docs/research.md) for the full comparison.
