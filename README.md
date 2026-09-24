# mispick

**Find out which of your MCP tools the model mixes up — measured, not guessed.**

Point it at any MCP server. It generates realistic user requests, asks a model to pick a tool for
each, and shows a **confusion matrix**: which tools get picked for which intent. Then it suggests
description rewrites for confused pairs and re-runs to prove the fix improved accuracy.

It never calls a tool. Only `initialize` and `tools/list`, so it is safe to point at any server.

## Status

**Pre-alpha, under active construction.** M0 (research) and M1 (loading tools) are done.

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
```

Measurement (`mispick run`) lands at M2. Install instructions, the demo GIF, and the benchmark
link land at M2 and M7 per SPEC section 11.

### Why this vs the static linters

`mcp-lint`, `mcpolish`, `oh-my-mcp`, `mcp-conform` and `mcp-cve-lint` check tool definitions
against rules, and they are good at it. A description can pass every rule and still be confused
with the tool next to it. Two newer projects do measure selection with a real model — `whichtool`
(local models, no rewrites, one server per run) and `toolfit` (rewrites, but cloud-only). Neither
looks across servers. See [docs/research.md](docs/research.md) for the full comparison.
