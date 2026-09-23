# tool-confusion (working name — not yet decided)

**Find out which of your MCP tools the model mixes up — measured, not guessed.**

Point it at any MCP server. It generates realistic user requests, asks a model to pick a tool for
each, and shows a **confusion matrix**: which tools get picked for which intent. Then it suggests
description rewrites for confused pairs and re-runs to prove the fix improved accuracy.

It never calls a tool. Only `initialize` and `tools/list`, so it is safe to point at any server.

## Status

**Pre-alpha. No product code yet.** Milestone M0 (research) is complete; M1 has not started.

- [`SPEC.md`](SPEC.md) — the source of truth for what this is
- [`docs/research.md`](docs/research.md) — Phase 0 research: competitors, naming, MCP spec
  `2026-07-28`, the Python `mcp` 2.2.0 client API, the registry API, Ollama models, and the
  description-quality literature

### Open decisions before M1

- **The name.** The working name `toolconf` is taken on PyPI; `mispick` is the recommended
  replacement. See [research §2](docs/research.md#2-name-availability).
- **Positioning.** Two projects now measure model tool-selection with a real model
  (`whichtool`, `toolfit`). Neither does cross-server collision detection, and neither combines
  all four differentiators, but SPEC section 2 needs updating. See
  [research §1](docs/research.md#1-competitors) and
  [§8](docs/research.md#8-verdict-and-recommended-spec-changes).

Install instructions, the demo GIF, and the benchmark link land at M2 and M7 per SPEC section 11.
