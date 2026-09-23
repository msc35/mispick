# tool-confusion — Project Spec

> Working name. Final name decided in Phase 0 (must be free on PyPI, npm and GitHub).
> CLI command (working): `toolconf`

## 1. One-line pitch

**Find out which of your MCP tools the model mixes up — measured, not guessed.**

Point it at any MCP server. It generates realistic user requests, asks a model to pick a tool for each, and shows a **confusion matrix**: which tools get picked for which intent. Then it suggests description rewrites for confused pairs and **re-runs to prove the fix improved accuracy**.

## 2. Why this exists (positioning)

Static MCP linters already exist and are good: `mcp-lint` (PyPI, 23 rules), `oh-my-mcp` (npm, includes token budget), `mcpolish` (description quality rules), `mcp-conform`, `mcp-cve-lint`. **Do not rebuild a static linter.**

What they don't do: run an actual model and measure selection behavior. A description can pass every lint rule and still be confused with a neighboring tool. Rule-based linters predict problems. This tool measures them.

Differentiators (keep all four, they are the product):
1. **Confusion matrix** between tools, not just a single accuracy score.
2. **Proven fixes**: every suggested rewrite is re-tested; only rewrites that improve the score are shown.
3. **Multi-server collisions**: load a whole `claude_desktop_config.json` / `.mcp.json` and find tools that collide *across* servers (e.g. three different `search` tools).
4. **Free by default**: runs on a local model via Ollama. Cloud models are optional.

Plus the launch asset:
5. **Public benchmark**: nightly-ish run over popular servers from the official MCP registry, published as a static leaderboard site.

## 3. Hard safety rule

**Never call a tool.** The model only *chooses* a tool and arguments. We record the choice and validate arguments against the JSON schema. We never send `tools/call` to any server. This makes the tool safe to run on any server, including destructive ones, and costs nothing beyond model inference. Enforce this in code (the client wrapper must not expose `call_tool`) and in a test.

## 4. Phase 0 — Research (do this first, write findings to `docs/research.md`)

Before writing product code, Claude Code must check and record:

1. **Competitors**: search GitHub, PyPI, npm for MCP tool-selection evaluation tools (known: `mcp-evals` by mclenhard, `scorecard-ai/mcp-eval`, `mcp-toolbench` (archived), `mcpolish`). For each: what it measures, whether it needs API keys, whether it produces a confusion matrix, whether it validates fixes. Update section 2 if any competitor already does the four differentiators. If one does, stop and report to the user before continuing.
2. **Name availability**: propose 5 names, check PyPI, npm, and GitHub for each. Prefer short, memorable, and not containing "lint".
3. **Current MCP spec**: latest protocol version, `tools/list` response shape, tool annotations (`readOnlyHint`, `destructiveHint`, etc.), pagination of `tools/list`.
4. **Python MCP SDK**: current package name and version, how to spawn a stdio server and call `initialize` + `tools/list` as a client, and how to connect over Streamable HTTP. Write a 20-line working example into `docs/research.md`.
5. **Official MCP registry API**: endpoint for listing servers, what metadata is available (package name, runtime, install command), rate limits.
6. **Ollama**: tool-calling support, which small models (≤8B) handle tool calling reliably, and the OpenAI-compatible endpoint.
7. **Research paper**: find the February 2026 paper by Wang et al. on tool descriptions across ~10,800 MCP servers (cited by mcpolish). Summarize its method and taxonomy of description problems. Use its taxonomy to label confusion causes if it fits.

Deliverable: `docs/research.md` with sources as links. No product code in Phase 0.

## 5. Architecture

Python 3.11+, managed with `uv`. Installable via `uvx <name>` and `pip`.

```
src/<pkg>/
  cli.py            # typer CLI
  sources/
    server.py       # connect to stdio / HTTP server, run initialize + tools/list (NO call_tool)
    config.py       # load claude_desktop_config.json / .mcp.json, fan out to servers
    snapshot.py     # load a tools/list JSON file directly
  generate.py       # create test queries per tool (LLM), cache to YAML
  select.py         # ask the model to choose a tool for each query
  models/
    base.py         # interface: choose(tools, query) -> Choice
    ollama.py       # default, OpenAI-compatible endpoint
    anthropic.py    # optional, needs ANTHROPIC_API_KEY
    openai.py       # optional
  metrics.py        # accuracy, per-tool precision/recall, confusion matrix, arg validity, stability
  fix.py            # propose rewrites for confused pairs, re-test, keep only improvements
  report/
    terminal.py     # rich
    json.py
    markdown.py     # for PR comments
    html.py         # single self-contained file with heatmap
    badge.py        # shields-style SVG
benchmark/          # scripts for the public leaderboard (Phase 5)
tests/
  fixtures/         # fake MCP servers with deliberately confusing tools
```

Dependencies: `mcp` (official SDK), `typer`, `rich`, `pydantic`, `jsonschema`, `pyyaml`, `httpx`, `openai` (used against Ollama's compatible endpoint), `anthropic` (optional extra). Keep the dependency list short. No heavy frameworks (no LangChain, no litellm).

## 6. Core pipeline

### 6.1 Load tools
Inputs (one of):
- `toolconf run --cmd "python -m my_server"` (stdio)
- `toolconf run --url https://.../mcp` (Streamable HTTP)
- `toolconf run --config ~/.../claude_desktop_config.json` (all servers, cross-server mode)
- `toolconf run --snapshot tools.json` (offline)

Handle `tools/list` pagination. Timeouts: 20s per server start. Record server name + version.

### 6.2 Generate test queries
For each tool, ask the model for N (default 8) realistic user requests that *should* trigger it:
- 4 straightforward
- 2 paraphrased / indirect
- 2 **hard negatives**: requests for the nearest neighboring tool (nearest by embedding or by the model's judgment) written to be close to this tool

Also generate a few "no tool fits" queries so we measure over-triggering.

Cache to `.toolconf/queries.yaml`. The user can edit this file and it becomes a stable test set. Cache key = hash of tool name + description + schema, so queries regenerate only when a tool changes.

Important: the generator model should be a different call from the selector call, and generation should only see the target tool plus its neighbors' names, to avoid leaking the answer through wording. Document this choice.

### 6.3 Selection
For each query: send the full tool list (as the model would see it in a real client) plus the query. Record: chosen tool (or none), arguments. Run K (default 3) times with temperature > 0 to measure stability.

### 6.4 Metrics
- Overall selection accuracy
- Per-tool precision and recall
- **Confusion matrix** (expected tool × chosen tool, plus a "none" row/column)
- Top confused pairs, ranked
- Argument validity rate (args validate against `inputSchema`)
- Stability (agreement across K runs)
- Token cost of the tool list (tokenizer estimate, label it an estimate)

### 6.5 Fix mode (`toolconf fix`)
For the top 3 confused pairs:
1. Ask the model to rewrite both descriptions to disambiguate, keeping meaning.
2. Re-run selection on the same cached queries with the rewritten descriptions (in memory only).
3. Show before/after accuracy per pair. Discard rewrites that don't improve.
4. Output a patch-style suggestion. Never modify the user's source code.

### 6.6 Scoring and exit codes
Score 0–100 from accuracy, stability, and arg validity (document the formula). `--fail-under N` for CI. Exit codes: 0 pass, 1 below threshold, 2 error.

## 7. Outputs

- Terminal: summary score, top confused pairs, heatmap using rich.
- `--format json|md|html`.
- HTML: single self-contained file, heatmap, per-tool table, fix suggestions. Must open offline.
- Badge SVG: `toolconf badge` → `toolconf-badge.svg` showing score and model used.
- Every report states the model name, N, K, and the date. Results depend on the model. Say so in the report.

## 8. GitHub Action

`action.yml` in repo root. Runs on PR, posts the markdown report as a PR comment, and compares with the base branch score ("accuracy 81% → 74%, `search_docs` now confused with `search_issues`"). For CI, default to a cloud model if an API key secret is set, else skip with a clear message (Ollama in CI is slow; document it).

## 9. Public benchmark (launch asset)

`benchmark/` folder:
1. Pull the top N (start with 30) servers from the official registry that start **without API keys**.
2. Snapshot their `tools/list` into `benchmark/snapshots/` (committed, so results are reproducible).
3. Run the pipeline with one fixed local model and fixed seeds.
4. Build a static site (plain HTML, no framework) with a leaderboard and a per-server page with its confusion matrix. Deploy to GitHub Pages.

Rules: only `initialize` + `tools/list`. Be respectful and neutral in wording ("most confused pair"), not "worst server". Link each server's repo. Add a "request a re-run" issue template.

## 10. Milestones (each ends with passing tests and a short README update)

- **M0** Phase 0 research doc. Stop and let the user review.
- **M1** Load tools from stdio + snapshot. Fixture server with 6 tools, 2 deliberately confusable. Test that `call_tool` is never reachable.
- **M2** Query generation with cache + selection with Ollama. Metrics + terminal report. **This is the MVP. The demo GIF is recorded here.**
- **M3** JSON / markdown / HTML reports, badge, `--fail-under`.
- **M4** Fix mode with before/after re-testing.
- **M5** Multi-server config mode with cross-server collision report.
- **M6** GitHub Action.
- **M7** Benchmark + GitHub Pages leaderboard.
- **M8** Launch polish: README, GIF, `uvx` install, PyPI release, MCP registry / awesome-lists submissions.

## 11. README requirements

First screen must contain: the one-line pitch, a GIF of the confusion matrix, and a one-line install (`uvx <name> run --cmd "..."`). Then: why this vs static linters (one short honest paragraph naming them), how it works (5 lines), benchmark link, limitations (results depend on the model; small local models are noisier).

## 12. Quality bar

- Tests with pytest, fixture servers, and a mocked model backend so tests run without Ollama.
- `ruff` + `mypy` clean. CI on GitHub Actions.
- Deterministic mode: `--seed`, temperature 0 option.
- No telemetry. No network calls except to the chosen model backend and the servers being tested.
- Honest docs: never claim results are model-independent.

## 13. Out of scope (v1)

Static lint rules (link to existing linters instead), security scanning, calling tools, a web app, auth flows for remote servers beyond a bearer token env var.
