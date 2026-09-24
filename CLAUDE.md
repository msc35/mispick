# CLAUDE.md

Read `SPEC.md` fully before doing anything. It is the source of truth.

## How to work on this repo
- Work milestone by milestone (SPEC section 10). Start with **M0 research**. Do not write product code until `docs/research.md` exists and the user has reviewed it.
- After each milestone: run tests, update README, commit, and stop to summarize what changed and what's next.
- If research shows a competitor already does the four differentiators in SPEC section 2, stop and report. Don't build a duplicate.
- Ask before adding any dependency not listed in SPEC section 5.

## Non-negotiables
- **Never call `tools/call` on any MCP server.** Only `initialize` and `tools/list`. There's a test for this. Keep it passing.
- Default model backend is local Ollama. Tests use a mocked backend and must pass offline.
- Never modify the user's MCP server source. Fix mode only suggests.
- Reports always state model, N, K, and date.

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run mypy src` — lint and types
- `uv run mispick run --snapshot tests/fixtures/confusing_tools.json` — quick manual check

## Style
- Python 3.11+, type hints everywhere, pydantic models for data.
- Small modules, small functions. No frameworks beyond SPEC section 5.
- Clear error messages that tell the user what to do next.
