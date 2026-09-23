# Phase 0 — Research

Date: 2026-09-23. Researcher: Claude Code. Deliverable for SPEC section 4.
No product code was written for this milestone.

**Headline: do not stop, but do rewrite SPEC section 2.** No single competitor ships all four
differentiators. Two projects that did not exist when the SPEC was written now cover two each,
and the SPEC's central claim — that nobody runs a real model to measure selection behavior —
is false as of today. Details in [§1](#1-competitors) and [§8](#8-verdict-and-recommended-spec-changes).

---

## 1. Competitors

### 1.1 The two that matter

**[`whichtool`](https://github.com/mattagame/whichtool)** (TypeScript, 0 stars, created 2026-08-22,
last push 2026-09-01). This is the closest architectural twin to our SPEC, including the safety rule.

Its README states the same hard rule we do: *"whichtool never executes a tool. It reads `tools/list`,
records what the model would have called, and stops."* It ships:

- a confusion matrix, with Wilson 95% confidence intervals (**our differentiator 1**)
- providers `anthropic`, `ollama`, `openai`, `openai-chat`, `openrouter`, `together`, `vllm`, any
  OpenAI-compatible endpoint, plus a deterministic `mock` (**our differentiator 4**)
- snapshot / stdio / Streamable HTTP targets; `legacy-sse` refused
- formats `terminal`, `json`, `markdown`, `html`, `junit`, `badge` (**our SPEC section 7, all of it**)
- a GitHub Action with base-branch comparison (**our SPEC section 8**)
- `--seed`, `--temperature`, permuted tool order, a trial cache, `--fail-under`-equivalent
  thresholds (`--min-accuracy`, `--max-over-trigger`), exit codes 0/1/2 (**our SPEC sections 6.6, 12**)
- `tasks generate` to draft a task set from tool descriptions, which you then edit and commit
  (**our SPEC section 6.2**)
- a `diff` command that refuses to subtract runs from different models and uses a paired sign test

What it does **not** do:

- **No fix mode.** This is a deliberate, documented refusal, and it is an argument we have to answer.
  From its [SPEC.md](https://github.com/mattagame/whichtool/blob/main/SPEC.md): *"Diagnosis stops where
  the domain begins. The tool says two descriptions are indistinguishable and shows the evidence.
  Choosing the words that distinguish them belongs to whoever owns the domain, and a generated rewrite
  would read as authoritative while knowing nothing about it."*
- **No cross-server mode.** Zero hits for cross-server / multi-server / `claude_desktop_config` in its
  README or SPEC. One target per run.
- **No public benchmark / leaderboard.**

Availability: it published `0.1.0` to npm on 2026-08-22 and **unpublished it the same day**
(`registry.npmjs.org/whichtool` shows `time.unpublished.versions: ["0.1.0"]`, and
`/whichtool/latest` is a 404). Its README carries a warning: *"Publication is temporarily paused."*
So it is source-only today, 0 stars, and not installable from npm.

**[`toolfit`](https://github.com/sreshtalluri/toolfit)** (Python, 0 stars, created 2026-08-27, last push
2026-09-15) — **published and installable**: [`toolfit` 0.2.1 on PyPI](https://pypi.org/project/toolfit/),
`uvx toolfit`. Its one-line description is nearly our pitch: *"finds the specific places your MCP server
confuses models, rewrites the tool descriptions and schemas to fix them, and proves the fix with a before
and after eval."*

It ships:

- a confusion matrix, per-tool pass rates with 95% CIs (**our differentiator 1**)
- **validated rewrites, reported with a p-value, showing accepted *and* rejected fixes**
  (**our differentiator 2** — its README: *"The rejected ones are what make the accepted ones
  believable."*) This is the differentiator we were most confident was ours.
- a free static `scan` (no model calls, no key), plus a model-graded `eval --fix --badge`
- a GitHub Action

What it does **not** do:

- **No local model.** Providers are inferred from the model name: `claude*` → Anthropic, `gpt*`/`o*` →
  OpenAI, `vendor/model` → OpenRouter, keyed by `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
  `OPENROUTER_API_KEY`. No Ollama. And: *"Task generation and fix proposals always use Anthropic"* — so
  an Anthropic key is mandatory for the whole eval+fix path. **Our differentiator 4 is untouched.**
- **No cross-server mode.** Stated flatly: *"One server per run."*
- **No leaderboard**: *"toolfit is not a model leaderboard."*

### 1.2 Others checked

| Project | What it measures | Keys needed | Confusion matrix | Validates fixes | Notes |
|---|---|---|---|---|---|
| [`mcpopt`](https://github.com/MichaelTheMay/mcpopt) (Python, 1 star) | Synthesizes a verifier-gated eval set, then auto-rewrites descriptions, accepting a rewrite only if it measurably improves | Yes, API spend; warns if gate + target share a provider | Mines confusion *pairs*, but does not publish a matrix as an output | **Yes** — gated on measured gains | **Not on PyPI.** Also **calls tools**: grounds arguments in "Tier A: harvested from read-only calls". That breaks our section 3 rule. Author notes its own E1–E6 results are not yet run ("require real API spend"). |
| [`mcp-evals`](https://github.com/mclenhard/mcp-evals) (133 stars, last push 2025-06-23) | LLM-as-judge scores per tool: accuracy, completeness, relevance, clarity, reasoning | Yes | No | No | The popular one, but a different question: it grades tool *output quality*, and it executes tools. Stale ~15 months. |
| [`scorecard-ai/mcp-eval`](https://github.com/scorecard-ai/mcp-eval) (1 star) | Hosted MCP eval product | Yes | No | No | Thin repo; product lives at mcpevals.ai. |
| [`mcp-scope`](https://github.com/Bharath-code/mcp-scope) (TypeScript, 0 stars) | "Which of your tools does Claude actually use?" — selection eval + token "context tax" | Yes, Anthropic Haiku | Not stated | No | A hosted SaaS (Cloudflare Workers + D1 + Polar payments, $149 tune-up), not a local CLI. |
| `mcp-toolbench` | — | — | — | — | **Could not locate.** No matching repo surfaced in GitHub search; treat the SPEC's "archived" note as unverified. |
| Static linters: [`mcp-lint`](https://pypi.org/project/mcp-lint/) (PyPI 0.1.0; also an unrelated [npm `mcp-lint`](https://www.npmjs.com/package/mcp-lint) 0.5.3), [`mcpolish`](https://pypi.org/project/mcpolish/) 0.1.1, [`mcp-conform`](https://pypi.org/project/mcp-conform/) 0.1.0, [`mcp-cve-lint`](https://pypi.org/project/mcp-cve-lint/) 0.2.0, [`oh-my-mcp`](https://www.npmjs.com/package/oh-my-mcp) 0.1.0 | Static rules only | No | No | No | All confirmed live. SPEC section 2's description of these is still accurate. |

Also relevant, as prior art rather than competition:
[GitHub's own offline evaluation of the GitHub MCP Server](https://github.blog/ai-and-ml/generative-ai/measuring-what-matters-how-offline-evaluation-of-github-mcp-server-works/)
uses exactly our method and says why: *"The confusion matrix allows us to see the reason behind low
precision and recall for certain tools and tweak their descriptions to minimize confusion."* Useful to
cite in the README — it is independent validation of the approach from a large MCP vendor.

### 1.3 Scorecard against SPEC section 2

| Differentiator | whichtool | toolfit | mcpopt | Us (planned) |
|---|---|---|---|---|
| 1. Confusion matrix | ✅ | ✅ | ~ (pairs only) | ✅ |
| 2. Proven / re-tested fixes | ❌ (refused on principle) | ✅ | ✅ | ✅ |
| 3. Cross-server collisions | ❌ | ❌ ("one server per run") | ❌ | ✅ |
| 4. Free by default (local Ollama) | ✅ | ❌ (Anthropic key required) | ❌ | ✅ |
| 5. Public benchmark | ❌ | ❌ | ❌ | ✅ |
| Never calls tools | ✅ | (static scan is free; eval does not call) | ❌ calls read-only tools | ✅ |

**Nobody holds 1+2+3+4.** The unoccupied ground is (a) the *combination*, (b) cross-server collision
detection, which no one does and which the MCP spec itself flags as a real problem (see §3), and
(c) the public benchmark.

---

## 2. Name availability

`toolconf` is **taken on PyPI** — [`toolconf` 0.1.2](https://pypi.org/project/toolconf/), "toolconfig
provides tools for loading and managing config files". The working name in the SPEC cannot be used as-is
for a PyPI package. It is free on npm.

Five candidates, all checked against PyPI (`/pypi/<name>/json`), npm (`registry.npmjs.org/<name>`), and
GitHub repo-name search:

| Name | PyPI | npm | GitHub name matches | Notes |
|---|---|---|---|---|
| **`mispick`** | free | free | 2, both unrelated and inactive | Cleanest of the set. Says what the tool finds. |
| **`misroute`** | free | free | 10; top is an unrelated OpenWrt networking repo (21 stars) | Good metaphor (routing), mild collision with networking vocabulary. |
| **`toolmix`** | free | free | 3, all unrelated SEO/utility sites | "mix up" reads as the confusion itself. |
| **`wrongtool`** | free | free | 1, unrelated | Bluntest; possibly too negative for a report you hand a vendor. |
| **`toolsight`** | free | free | 4; one is "ToolSight — AI tool recommender" (0 stars) | Weakest tie to the actual measurement. |

Rejected for being taken on both registries: `nearmiss`, `decoy`, `mixup`, `tangle`, `picky`.
Taken on npm only: `pickrate`, `toolpick`, `routecheck`.

Recommendation: **`mispick`** (PyPI + npm + GitHub all clear, no "lint" in the name, 7 letters), with
`misroute` as the fallback. Deliberately avoiding anything close to `whichtool` or `toolfit`.
A final check of the trademark-ish surface and the CLI binary name should happen at the moment of
reservation, since these can be claimed at any time.

---

## 3. Current MCP spec

**Latest protocol version: `2026-07-28`**, confirmed in the schema source:
`export const LATEST_PROTOCOL_VERSION = "2026-07-28";`
([schema.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2026-07-28/schema.ts),
[spec index](https://modelcontextprotocol.io/specification/latest)).

### `tools/list` response shape

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "resultType": "complete",
    "tools": [
      {
        "name": "get_weather",
        "title": "Weather Information Provider",
        "description": "Get current weather information for a location",
        "inputSchema": {
          "type": "object",
          "properties": { "location": { "type": "string", "description": "City name or zip code" } },
          "required": ["location"]
        },
        "icons": [{ "src": "https://example.com/weather-icon.png", "mimeType": "image/png", "sizes": ["48x48"] }]
      }
    ],
    "nextCursor": "next-page-cursor",
    "ttlMs": 300000,
    "cacheScope": "public"
  }
}
```

Things here that the SPEC does not yet account for:

- **`resultType: "complete"`** wraps results now, and `ttlMs` / `cacheScope` carry caching hints.
- **`_meta` is mandatory on every request**: `io.modelcontextprotocol/protocolVersion`,
  `io.modelcontextprotocol/clientInfo`, `io.modelcontextprotocol/clientCapabilities`. The SDK handles
  this, but a hand-rolled JSON-RPC client would not.
- **`Tool` fields** ([schema.ts](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2026-07-28/schema.ts)):
  `name`, `title`, `description?`, `inputSchema` (required, `type: "object"`, JSON Schema 2020-12 by
  default, any 2020-12 keyword allowed including `$ref`/`$defs`/`oneOf`/`if`), `outputSchema?`,
  `annotations?`, `icons?`, `_meta?`. Display-name precedence is `title` → `annotations.title` → `name`.
  **Our arg-validation step must handle `$ref`/`$defs`**, which rules out a naive validator.
- **`x-mcp-header`**: a property-level annotation mirroring an argument into an `Mcp-Param-*` HTTP
  header. Clients using Streamable HTTP **MUST reject** tools whose `x-mcp-header` values are invalid
  (empty, non-token, duplicated case-insensitively, on a `number`-typed or non-statically-reachable
  property) and **MUST exclude them from `tools/list`**. `whichtool`'s `inspect` already checks this.

### Tool annotations, with defaults

From `interface ToolAnnotations`:

| Field | Default | Meaning |
|---|---|---|
| `title?: string` | — | Human-readable title |
| `readOnlyHint?: boolean` | **`false`** | True = tool does not modify its environment |
| `destructiveHint?: boolean` | **`true`** | True = may perform destructive updates. Meaningful only when `readOnlyHint == false` |
| `idempotentHint?: boolean` | **`false`** | True = repeat calls with same args have no additional effect. Meaningful only when `readOnlyHint == false` |
| `openWorldHint?: boolean` | **`true`** | True = interacts with an open world of external entities |

Note the defaults are the *pessimistic* ones (`destructiveHint` defaults to true). And the spec is
explicit that these are **untrusted**: *"clients MUST consider tool annotations to be untrusted unless
they come from trusted servers."* We should surface contradictory annotation combinations but never rely
on them for safety — which is moot for us, since we never call anything.

### Pagination

Opaque cursor, **server-chosen page size** ([pagination spec](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)):
send `params.cursor`, read `result.nextCursor`, repeat until `nextCursor` is absent. Three rules worth
encoding as tests:

1. Clients **MUST NOT** assume a fixed page size.
2. Cursors are opaque — **an empty string is a valid cursor and MUST NOT be treated as end-of-results**.
   Only absence of a non-null `nextCursor` means done. This is an easy off-by-one bug; test it.
3. Invalid cursor → JSON-RPC error `-32602`.

Also: servers **SHOULD** return tools in deterministic order, and the tool set **MUST NOT** vary
per-connection. Both are assumptions our cache key in section 6.2 quietly depends on; neither is a MUST
for ordering, so we should sort before hashing.

### Cross-server collisions are a spec-acknowledged problem

Directly supporting differentiator 3, from the tools spec:

> *"Tool name uniqueness is scoped to a single server. Clients or proxies that aggregate tools from
> multiple servers MAY encounter naming collisions (for example, two servers each exposing a `search`
> tool) and SHOULD implement a disambiguation strategy such as prefixing tool names with a server
> identifier. The server `name` (from `serverInfo`) is not guaranteed to be unique across servers and
> SHOULD NOT be relied upon for disambiguation."*

The spec names our exact example (`search`), says clients *should* disambiguate, and warns that
`serverInfo.name` is not a reliable key. Good quote for the README, and a constraint: our cross-server
report must key servers by the user's config label, not by `serverInfo.name`.

### Transports

`stdio` and Streamable HTTP are current. HTTP+SSE has been deprecated since `2025-03-26`; `whichtool`
refuses it outright, and we should too rather than carry a compatibility path. Note
`send_roots_list_changed` is now deprecated as of 2026-07-28 (SEP-2577).

---

## 4. Python MCP SDK

Package: **[`mcp` 2.2.0](https://pypi.org/project/mcp/)**, uploaded 2026-09-07, `requires_python >=3.10`.
Repo: [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk).

**This is the 2.x line and the client API is different from the 1.x `stdio_client` + `ClientSession`
dance most tutorials show.** There is now a high-level `Client`.

Dependencies it already brings, which changes our SPEC section 5 list: `pydantic>=2.12`,
**`jsonschema>=4.20`**, `httpx2>=2.5`, `anyio`, `mcp-types==2.2.0`, `starlette`, `uvicorn`,
`sse-starlette`, `pyjwt[crypto]`, `opentelemetry-api`, `python-multipart`. `typer` and `rich` are
available as the `cli` / `rich` extras.

So **`jsonschema` and `httpx` are already transitive** — we should still declare `jsonschema` directly
(we import it), but we do not need to add `httpx` for its own sake, and note the SDK uses `httpx2`.

Verified exports, from
[`src/mcp/__init__.py`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/__init__.py):
`Client`, `ClientSession`, `ClientSessionGroup`, `StdioServerParameters`, `stdio_client`.
(The docs site shows `from mcp.server.stdio import StdioServerParameters`, which is **wrong** — the real
path is `mcp.client.stdio`, and it is re-exported at top level. Use the top-level import.)

`Client` signature highlights ([client/client.py](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/client.py)):

- `Client(server, *, raise_exceptions=False, read_timeout_seconds=None, ...)`
- `server` accepts a **URL string** (Streamable HTTP), a **`StdioServerParameters`** (launched as a
  subprocess), any `Transport`, or — *in tests* — a `Server` / `MCPServer` instance connected
  **in-process**.
- `await client.list_tools(*, cursor=None, meta=None, cache_mode="use") -> ListToolsResult`
  (`.tools`, `.next_cursor`)
- `client.server_info`, `client.protocol_version`, `client.server_capabilities`, `client.instructions`
- and `await client.call_tool(...)`, which **is exposed on this class** — see the wrapper note below.

### Working example (verified against the 2.2.0 source)

```python
import asyncio
from mcp import Client, StdioServerParameters


async def list_all_tools(server: str | StdioServerParameters) -> list:
    """initialize + tools/list only, following pagination. Never calls a tool."""
    tools, cursor = [], None
    async with Client(server, read_timeout_seconds=20) as client:
        print(f"connected: {client.server_info} protocol={client.protocol_version}")
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:          # empty string is a VALID cursor, only None ends it
                break
    return tools


async def main() -> None:
    stdio = StdioServerParameters(command="python", args=["-m", "my_server"])
    for tool in await list_all_tools(stdio):                 # stdio
        print(tool.name, "-", tool.description)
    for tool in await list_all_tools("https://example.com/mcp"):   # Streamable HTTP
        print(tool.name)


asyncio.run(main())
```

Three consequences for our architecture:

1. **The SPEC section 3 rule needs a wrapper, exactly as written.** `mcp.Client` exposes `call_tool`,
   so `sources/server.py` must never hand a raw `Client` to the rest of the codebase. It should return
   plain `Tool` models (or our own pydantic type) and keep the `Client` private to that module. The
   enforcement test should assert that no `call_tool` attribute is reachable from whatever
   `sources/server.py` returns, and ideally grep the tree for `call_tool` / `tools/call`.
2. **In-process `Server` connections are a gift for M1 tests.** Our fixture servers with deliberately
   confusable tools can be plain `FastMCP`/`Server` objects passed straight to `Client(...)` — no
   subprocess, no ports, fast and offline.
3. **`ClientSessionGroup` exists** and is worth evaluating for M5's multi-server fan-out before we hand-roll it.

---

## 5. Official MCP registry API

Base: `https://registry.modelcontextprotocol.io`. OpenAPI at
[`/openapi.yaml`](https://registry.modelcontextprotocol.io/openapi.yaml) (1866 lines, HTTP 200).
Main endpoint `GET /v0/servers`, verified live.

Query parameters (from the OpenAPI spec, confirmed by probing):

| Param | Notes |
|---|---|
| `limit` | default **30**, **max 100** — empirically `limit=100` → 200, `limit=101` → **422** |
| `cursor` | opaque pagination cursor; response has `metadata.nextCursor` and `metadata.count` |
| `version` | `latest` for the latest version, or an exact version like `1.2.3` |
| `search` | substring match **on name only** |
| `updated_since` | RFC3339 datetime |

**`version=latest` is essential.** Without it the endpoint returns every published version as a separate
entry — a plain `?limit=2` returned `ac.inference.sh/mcp` twice (1.0.0 and 1.0.1), both with
`_meta."io.modelcontextprotocol.registry/official".isLatest: false`. The benchmark scripts must pass
`version=latest` or they will double-count servers.

Server entry shape (keys observed across a 40-entry page): `$schema`, `name`, `title`, `description`,
`version`, `repository`, `websiteUrl`, `icons`, `packages`, `remotes`, `_meta`. Names are reverse-DNS
(`io.github.owner/repo`, `com.pulsemcp/remote-filesystem`). Current schema:
`https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json`.

Install metadata is exactly what the benchmark needs:

```json
"packages": [{ "registryType": "npm",  "identifier": "pretrip-mcp", "version": "1.0.1",
               "transport": { "type": "stdio" } }]
"packages": [{ "registryType": "pypi", "identifier": "adeu", "version": "1.5.2",
               "transport": { "type": "stdio" } }]
"remotes":  [{ "type": "streamable-http", "url": "https://api.inference.sh/mcp" }]
```

`_meta."io.modelcontextprotocol.registry/official"` carries `status` (`active`), `isLatest`,
`publishedAt`, `updatedAt`.

**Rate limits: not documented** in the OpenAPI spec and I found no published figure. For section 9,
assume none is guaranteed: serialize requests, cache aggressively, and commit snapshots (which the SPEC
already requires). Note the registry does **not** expose an "API keys required" flag — picking the
key-free top 30 means reading each package's own docs or attempting a start and recording the failure.
Budget real time for that in M7; it is the fiddliest part of the benchmark.

---

## 6. Ollama

**Tool calling is supported.** Request shape uses OpenAI-style function definitions; responses carry
`tool_calls` ([tool-calling docs](https://docs.ollama.com/capabilities/tool-calling)):

```json
{ "role": "assistant",
  "tool_calls": [ { "type": "function",
                    "function": { "index": 0, "name": "function_name",
                                  "arguments": { "param_name": "value" } } } ] }
```

**OpenAI-compatible endpoint: `http://localhost:11434/v1/`** (cloud: `https://ollama.com/v1`),
supporting `/chat/completions`, `/completions`, `/models`, `/embeddings`, and `/responses`
([compatibility docs](https://docs.ollama.com/api/openai-compatibility)). Tools are supported through
it, and the supported sampling parameters are `frequency_penalty, presence_penalty, response_format,
**seed**, stop, stream, **temperature**, top_p, max_tokens`.

**`seed` and `temperature` both work through the compatible endpoint, so SPEC section 12's deterministic
mode is achievable on the default backend.** Using the `openai` client against `/v1` — as SPEC section 5
plans — is confirmed viable.

### Small tool-capable models — this needs a decision

Local Ollama is installed here (**client 0.33.3**), but no server was running, so these are library
findings, not measured results. Filtering the library by tool support (`ollama.com/search?c=tools`)
returns: `qwen3.6`, `qwen3.8`, `qwen3.8-flash-next`, `granite4.1`, `glm-5.1/5.2/5.3`, `deepseek-v4-*`,
`kimi-k2.6`, `minimax-m3`, `nemotron3`, `mistral-medium-3.5`, and others.

**The problem: the current generation is large.** Base sizes actually published:

- `qwen3.6`: **27b, 35b** only
- `qwen3.8`: **27b** only
- `nemotron3`: 33b only
- `granite4.1`: **3b, 8b**, 30b

So among *current* tool-tagged families, only `granite4.1:3b` / `granite4.1:8b` meet the SPEC's ≤8B bar.
Previous generations are still pullable and still tool-capable:

- `qwen3`: 0.6b, 1.7b, **4b**, **8b**, 14b, 30b, 32b, 235b
- `qwen3.5`: 0.8b, 2b, **4b**, **9b**, 27b, 35b, 122b, 397b
- `granite4`: 1b, 3b, **7b**, 32b
- `llama3.1:8b`, `mistral-nemo:12b`, `qwen2.5:7b`

Recommendation for the default: **`qwen3.5:4b`**, with `granite4.1:3b` as the smaller fallback and
`qwen3.5:9b` for anyone with headroom. Rationale: Qwen3 is the family most consistently reported as
reliable at emitting valid tool calls locally, `qwen3.5:4b` is current-generation and small enough to be
genuinely free to run, and `whichtool` independently settled on a small qwen3 (`qwen3:4b`) for the same
job. **This must be confirmed by measurement in M2 before the README claims it** — pick the default by
running our own fixture set across `qwen3.5:4b`, `qwen3.5:9b`, `granite4.1:3b` and `qwen3:8b`.

Two operational warnings, both from `whichtool`'s README and worth inheriting:

- Reasoning models spend heavily on hidden thinking tokens — *"on reasoning models such as qwen3, a
  single trial can take tens of seconds of thinking tokens whichtool never reads."* With N=8 and K=3 per
  tool, a 6-tool server is ~144 selection calls. Time that before promising a fast demo GIF.
- Prompt-token estimates are a **lower bound**, never a price estimate. SPEC section 6.4 already says to
  label the token figure an estimate; keep that wording.

---

## 7. The research paper

**The SPEC's citation is wrong on two counts.** It asks for "the February 2026 paper by Wang et al. on
tool descriptions across ~10,800 MCP servers". The paper that matches everything else — February 2026,
tool-description quality, cited by `mcpolish` — is:

> **Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency
> with Augmented MCP Tool Descriptions**
> Mohammed Mehedi Hasan, Hao Li, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan
> arXiv [2602.14878](https://arxiv.org/abs/2602.14878), submitted 2026-02-16.
> HTML: <https://arxiv.org/html/2602.14878v1>

Corrections: the authors are **Hasan et al.**, not Wang et al., and the dataset is **856 tools across
103 MCP servers** (23 official, 80 community), **not ~10,800 servers**. I could not find any Feb-2026
description-quality paper covering 10,800 servers; the nearest large-N works are different studies
(e.g. MCPAgentBench, ~9,714 servers). Treat the 10,800 figure as an error and do not repeat it in our
README.

### Method

Six-component rubric drawn from Anthropic docs, community guidelines and prior work. A three-model
**"LLM-as-Jury"** ensemble (`gpt-4.1-mini`, `claude-haiku-3.5`, `qwen3-30b`) scored each component 1–5;
inter-rater agreement ICC > 0.75 for most components; scores averaged, and a mean **< 3** raises the
corresponding smell. Agent effects were measured on **MCP-Universe** (231 tasks, six domains) against
GPT-4.1, Qwen3-Coder-480B, GLM-4.5 and Qwen3-Next-80B, reporting Success Rate, Average Evaluator score
and Average steps.

### Taxonomy — adopt this for labeling confusion causes

| Smell | Prevalence | Definition |
|---|---|---|
| **Unclear Purpose** | 56% | The tool's functional core and identity are not clearly articulated |
| **Missing Usage Guidelines** | 89.3% | No decision criteria for when/how to use the tool; no activation criteria |
| **Unstated Limitations** | 89.8% | Omits constraints, caveats, corner cases where the tool fails |
| **Opaque Parameters** | 84.3% | Input parameters unexplained beyond their data types |
| **Underspecified / Incomplete** | 79.1% | Too brief relative to the tool's complexity |
| **Exemplar Issues** | 77.9% | Usage examples sparse, uninformative or missing |

**97.1% of descriptions carry at least one smell; only 2.9% are clean.** No significant difference
between official and community servers.

It fits our needs well, with one caveat: the taxonomy describes *description defects*, while our
confusion matrix finds *pairwise* failures. "Unclear Purpose" is the one that maps most directly onto a
confused pair. I'd recommend using these six as the label vocabulary for `fix.py`'s explanation of *why*
a pair confuses, and citing prevalence figures in the README, while keeping our own pairwise framing.

### The finding that most matters to us

Augmenting descriptions raised task success by a **median 5.85 percentage points** — but increased
execution steps by a **median 67.46%**, and **regressed in 16.67% of cases**. Ablation found *no single
component combination that consistently improved performance across all domains and models*, and
removing the Examples component did not significantly degrade performance.

**This is the strongest available argument for differentiator 2.** A published, peer-reviewed-style study
found that LLM-rewritten tool descriptions make things *worse* one time in six. A tool that proposes
rewrites without re-testing them is shipping a 1-in-6 chance of harm. Our SPEC section 6.5 — re-run and
discard rewrites that don't improve — is precisely the control this result calls for. It is also the
direct rebuttal to `whichtool`'s "a generated rewrite would read as authoritative while knowing nothing
about it": correct, which is why the rewrite must be *measured* rather than trusted. Lead with this in
the README.

---

## 8. Verdict and recommended SPEC changes

### Do not stop

Per CLAUDE.md, I stop only if one competitor already does all four differentiators. **None does.**
`whichtool` has 1+4 and refuses 2 on principle. `toolfit` has 1+2 and needs an Anthropic key. Nobody
has 3 or 5. The combination, cross-server collisions, and the public benchmark are all unclaimed, and
both close competitors are 0-star and weeks old — one of them unpublished from npm.

But the field moved under the SPEC, and two things in it are now false. **Recommended edits, for your
approval — I have not touched SPEC.md:**

1. **Section 2, replace the "What they don't do" paragraph.** It currently says static linters are the
   competition and *"what they don't do: run an actual model and measure selection behavior."* That is
   no longer true. Proposed replacement:

   > Static MCP linters already exist and are good: `mcp-lint`, `oh-my-mcp`, `mcpolish`, `mcp-conform`,
   > `mcp-cve-lint`. Do not rebuild a static linter. Two projects now do measure selection behavior with
   > a real model: `whichtool` (TypeScript, local-model support, confusion matrix — but deliberately no
   > rewrites, one server per run) and `toolfit` (Python, confusion matrix *and* re-tested rewrites — but
   > cloud-only, an Anthropic key is mandatory, one server per run). Neither does cross-server collision
   > detection; neither publishes a benchmark. Our claim is the combination: measured confusion **and**
   > proven fixes **and** across your whole config **and** free on a local model.

2. **Section 4.7 and any README copy: fix the citation.** Hasan et al., arXiv 2602.14878, 856 tools
   across 103 servers. Drop "Wang et al." and "~10,800 servers".

3. **Working name.** `toolconf` is taken on PyPI. Recommend `mispick`. Needs your decision before M1,
   since it sets the package path `src/<pkg>/` and the CLI name.

4. **Section 5 dependency list.** `mcp` 2.2.0 already pulls `jsonschema`, `pydantic`, and `httpx2`.
   Declare `jsonschema` (we import it) but drop `httpx` as a direct dependency unless we need it
   independently. Also: the SDK requires Python ≥3.10, so our 3.11+ floor is fine.

5. **Two additions to section 6.1 worth writing down now**, both cheap and both easy to get wrong later:
   an empty-string `nextCursor` is valid and must not terminate pagination; and tool order is only a
   SHOULD, so sort tools before computing the section 6.2 cache key.

6. **Consider adding to section 6.4**: `whichtool` reports Wilson 95% confidence intervals and `toolfit`
   reports p-values on before/after. With K=3 our per-cell numbers are noisy, and shipping bare
   percentages next to two competitors that both quantify uncertainty would look careless. Worth a
   decision now, since it affects `metrics.py`'s shape.

### Open questions for you

- Name: `mispick`, or one of the alternatives, or your own?
- Given `toolfit` occupies differentiator 2 and `whichtool` occupies 1+4, do you want to **re-order the
  milestones** to reach cross-server mode (M5, currently unclaimed by anyone) earlier than M4? As
  written, the MVP at M2 is the part of the product that is now least distinctive.
- Do we adopt confidence intervals in M2's metrics, or defer?

### Sources

- Competitors: [whichtool](https://github.com/mattagame/whichtool) ([SPEC](https://github.com/mattagame/whichtool/blob/main/SPEC.md)) · [toolfit](https://github.com/sreshtalluri/toolfit) ([PyPI](https://pypi.org/project/toolfit/)) · [mcpopt](https://github.com/MichaelTheMay/mcpopt) · [mcp-evals](https://github.com/mclenhard/mcp-evals) · [scorecard-ai/mcp-eval](https://github.com/scorecard-ai/mcp-eval) · [mcp-scope](https://github.com/Bharath-code/mcp-scope) · [GitHub's offline MCP evaluation](https://github.blog/ai-and-ml/generative-ai/measuring-what-matters-how-offline-evaluation-of-github-mcp-server-works/)
- Static linters: [mcp-lint](https://pypi.org/project/mcp-lint/) · [mcpolish](https://pypi.org/project/mcpolish/) · [mcp-conform](https://pypi.org/project/mcp-conform/) · [mcp-cve-lint](https://pypi.org/project/mcp-cve-lint/) · [oh-my-mcp](https://www.npmjs.com/package/oh-my-mcp)
- MCP spec: [latest](https://modelcontextprotocol.io/specification/latest) · [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) · [pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination) · [schema.ts 2026-07-28](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2026-07-28/schema.ts)
- Python SDK: [PyPI `mcp`](https://pypi.org/project/mcp/) · [repo](https://github.com/modelcontextprotocol/python-sdk) · [client docs](https://py.sdk.modelcontextprotocol.io/client/) · [`__init__.py`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/__init__.py) · [`client/client.py`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/client.py)
- Registry: [`/v0/servers`](https://registry.modelcontextprotocol.io/v0/servers?limit=2&version=latest) · [OpenAPI](https://registry.modelcontextprotocol.io/openapi.yaml)
- Ollama: [tool calling](https://docs.ollama.com/capabilities/tool-calling) · [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) · [tool-support blog](https://ollama.com/blog/tool-support) · [library, tools filter](https://ollama.com/search?c=tools)
- Paper: [arXiv 2602.14878](https://arxiv.org/abs/2602.14878) · [HTML v1](https://arxiv.org/html/2602.14878v1)
