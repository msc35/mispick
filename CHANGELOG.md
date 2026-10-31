# Changelog

## 0.2.0 — 2026-10-31

**New**
- Cloud backends beyond OpenAI and Anthropic: `--model gemini/<model>` (needs `GEMINI_API_KEY`
  or `GOOGLE_API_KEY`), and `--model compatible/<model>` for any OpenAI-compatible endpoint, such
  as vLLM, OpenRouter or LM Studio, via `MISPICK_BASE_URL`.
- Rate limits: OpenAI-compatible backends (Ollama, OpenAI, Gemini, custom) wait out a per-minute
  429 using the provider's own retry delay. Remote endpoints also retry dropped connections,
  timeouts and 5xx "overloaded" replies. A used-up daily quota raises `QuotaExhaustedError` and
  is never recorded as a model decision.
- `PacedBackend` paces calls under a requests-per-minute and tokens-per-minute cap and stops
  cleanly at a call budget.
- `run_selection(completed=...)` reuses trials from an interrupted run instead of repeating them.
- Benchmark: `measure.py` is resumable (`--only`, `--rerun`, `--rpm`, `--tpm`, `--budget-calls`),
  and the `benchmark` workflow runs it daily on the Gemini free tier.

**Fixed**
- A tool list larger than the model's context was silently truncated, so every trial read as
  "chose nothing" and produced a convincing 0%. mispick now compares the prompt size the API
  reports against its own estimate and refuses to score.
- Gemini rejected every request that carried `seed`. Seed is now sent only to backends that
  accept it.
- Generated requests leaned on missing context ("insert these nodes"), which pushed models to
  look before acting and was scored against the server. The generator now has to spell out
  concrete details.
- A server that fails to start now reports why, instead of "1 sub-exception".
- A remote connection failure no longer suggests starting Ollama.

## 0.1.0 — 2026-09-24

First release.

- Load tools from a stdio server, a Streamable HTTP server, a captured snapshot, or every server
  in a `claude_desktop_config.json` / `.mcp.json` at once. Only `initialize` and `tools/list` are
  ever sent; `tools/call` is unreachable by construction and by test.
- Generate a test set per tool (straightforward, paraphrased, hard negatives against the nearest
  neighbour, plus "no tool fits" cases), cached to an editable `.mispick/queries.yaml` keyed on a
  per-tool fingerprint.
- Measure selection accuracy, per-tool precision and recall, a confusion matrix with `(none)` and
  `(phantom)` rows, argument validity against `inputSchema`, stability across K runs, over-
  triggering, and a tool-list token estimate. Every rate carries a Wilson 95% interval.
- `mispick fix`: rewrite the worst confused pairs, re-run the same queries, and accept a rewrite
  only when an exact McNemar test agrees. Rejected rewrites are shown.
- Cross-server analysis: name collisions plus measured leakage between servers.
- Reports: terminal, JSON (versioned schema), Markdown for PR comments, a self-contained HTML page,
  and an SVG badge. `mispick report` re-renders a saved run; `mispick compare` diffs two runs.
- Backends: Ollama (default, local, free), OpenAI, Anthropic, and a deterministic mock.
- GitHub Action with a sticky PR comment, base-branch comparison and `--fail-under`.
- Benchmark scripts and a static leaderboard generator.
