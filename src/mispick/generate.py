"""Generate the test queries, and cache them so they become a stable test set.

**Why generation is a separate model call from selection.** If we asked one call to both
invent a request and pick a tool, the wording of the request would be conditioned on the
answer we are about to score - the model would write "search the documentation for X" and
then triumphantly pick `search_docs`. So generation happens in its own call, and it is shown
**only the target tool plus the bare names of its neighbours**, never their descriptions.
That way a generated request cannot quote description wording that only the right tool has.

Per tool (N=8 by default):
  4 straightforward, 2 paraphrased or indirect, 2 hard negatives.

A **hard negative** for tool T is a request that genuinely belongs to T's nearest neighbour
but is written to sit as close to T as possible. Its `expected` answer is the neighbour, so a
model that grabs T instead is caught over-triggering. Plus a handful of "no tool fits"
queries, which measure over-triggering against the whole tool list.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mispick.models.base import DEFAULT_MAX_TOKENS, Backend, BackendError
from mispick.types import Query, Tool, ToolSet

#: Where the cache lives, relative to the working directory.
CACHE_DIR = Path(".mispick")
CACHE_FILE = "queries.yaml"
CACHE_VERSION = 1

DEFAULT_N = 8
#: How many "no tool fits" queries to generate for the whole tool list.
DEFAULT_NO_TOOL = 3

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def nearest_neighbour(tool: Tool, tools: list[Tool]) -> Tool | None:
    """The other tool most easily mistaken for this one.

    Token overlap over name and description, rather than embeddings: it needs no extra
    dependency and no model call, and for the failure we are hunting - two tools that read
    alike - it is the right signal. Ties break on name, so the choice is deterministic.
    """
    others = [t for t in tools if t.qualified_name != tool.qualified_name]
    if not others:
        return None
    mine = _words(f"{tool.name} {tool.description or ''}")
    if not mine:
        return sorted(others, key=lambda t: t.qualified_name)[0]

    def score(other: Tool) -> tuple[float, str]:
        theirs = _words(f"{other.name} {other.description or ''}")
        if not theirs:
            return (0.0, other.qualified_name)
        jaccard = len(mine & theirs) / len(mine | theirs)
        # An identical description is the strongest possible signal.
        if (other.description or "") == (tool.description or "") and tool.description:
            jaccard += 1.0
        return (-jaccard, other.qualified_name)

    return sorted(others, key=score)[0]


@dataclass
class GenerationPlan:
    """How many of each kind of query to ask for."""

    straightforward: int = 4
    paraphrased: int = 2
    hard_negative: int = 2

    @classmethod
    def for_n(cls, n: int) -> GenerationPlan:
        """Scale the 4/2/2 split to an arbitrary N, keeping the shape."""
        if n <= 0:
            raise ValueError("N must be at least 1")
        hard = max(1, round(n * 0.25)) if n >= 4 else 0
        para = max(1, round(n * 0.25)) if n >= 3 else 0
        straight = max(1, n - hard - para)
        return cls(straightforward=straight, paraphrased=para, hard_negative=hard)

    @property
    def total(self) -> int:
        return self.straightforward + self.paraphrased + self.hard_negative


def _schema_summary(tool: Tool) -> str:
    props = (tool.input_schema or {}).get("properties") or {}
    required = set((tool.input_schema or {}).get("required") or [])
    if not props:
        return "(no parameters)"
    bits = []
    for key, spec in props.items():
        kind = spec.get("type", "any") if isinstance(spec, dict) else "any"
        mark = "*" if key in required else ""
        bits.append(f"{key}{mark}: {kind}")
    return ", ".join(bits) + "   (* = required)"


def build_prompt(tool: Tool, neighbour: Tool | None, plan: GenerationPlan) -> str:
    """The generation prompt. Neighbours appear by name only - never their descriptions."""
    neighbour_name = neighbour.qualified_name if neighbour else None
    lines = [
        "You are writing a test set for a tool-routing evaluation.",
        "",
        "The tool under test:",
        f"  tool: {tool.qualified_name}",
        f"  description: {tool.description or '(none)'}",
        f"  parameters: {_schema_summary(tool)}",
        "",
    ]
    if neighbour_name:
        lines += [
            f"The nearest other tool is named: {neighbour_name}",
            "(You are shown only its name on purpose. Do not guess what it does beyond its name.)",
            "",
        ]
    lines += [
        "Write realistic things a user would type to an assistant. Requirements:",
        f"  - {plan.straightforward} straightforward requests that clearly need"
        f" {tool.qualified_name}.",
        f"  - {plan.paraphrased} indirect or paraphrased requests that still need"
        f" {tool.qualified_name},",
        "    phrased the way a busy person types, without reusing the tool's own wording.",
    ]
    if neighbour_name and plan.hard_negative:
        lines += [
            f"  - {plan.hard_negative} requests that actually need {neighbour_name} instead,",
            f"    written to sound as close to {tool.qualified_name} as you can make them.",
            '    Mark these with kind "hard_negative".',
        ]
    lines += [
        "",
        "Rules: one sentence each. No tool names in the text. No numbering.",
        "Do not quote the description above.",
        "",
        "Reply with JSON only:",
        '{"queries": [{"text": "...", "kind": "straightforward|paraphrased|hard_negative"}]}',
    ]
    return "\n".join(lines)


def build_no_tool_prompt(tools: list[Tool], count: int) -> str:
    names = ", ".join(t.qualified_name for t in tools)
    return "\n".join(
        [
            "You are writing negative cases for a tool-routing evaluation.",
            "",
            f"An assistant has exactly these tools: {names}",
            "",
            f"Write {count} realistic user requests that NONE of those tools can satisfy,",
            "but which sound like they belong in the same product. A correct assistant would",
            "answer from its own knowledge or say it cannot help, and call no tool at all.",
            "",
            "Rules: one sentence each. No tool names. No numbering.",
            "",
            "Reply with JSON only:",
            '{"queries": [{"text": "..."}]}',
        ]
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply that may be wrapped in prose or fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Last resort: the outermost {...}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise BackendError(
        f"Could not read JSON from the model's reply ({len(text)} characters). "
        f"Reply began: {text[:300]!r}"
    )


#: One {"text": ..., "kind": ...} object, however the model spaced it.
_ITEM = re.compile(
    r'\{[^{}]*?"text"\s*:\s*"(?P<text>(?:[^"\\]|\\.)*)"'
    r'(?:[^{}]*?"kind"\s*:\s*"(?P<kind>[^"]*)")?[^{}]*?\}',
    re.DOTALL,
)


def salvage_queries(text: str) -> list[dict[str, Any]]:
    """Recover whatever complete query objects a malformed reply contains.

    Small local models truncate, trail off into prose, or forget a closing bracket. When the
    reply is 80% good, throwing all of it away and failing the run is the wrong trade - and
    re-asking costs another 30 seconds of someone's laptop. So we take the objects that did
    parse. Anything genuinely unusable still raises, upstream.
    """
    out: list[dict[str, Any]] = []
    for match in _ITEM.finditer(text):
        try:
            value = json.loads(f'"{match.group("text")}"')
        except json.JSONDecodeError:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        item: dict[str, Any] = {"text": value.strip()}
        if match.group("kind"):
            item["kind"] = match.group("kind")
        out.append(item)
    return out


def _parse_queries(reply: str) -> list[dict[str, Any]]:
    """Read the query list, salvaging a partly-broken reply rather than discarding it."""
    try:
        payload = _extract_json(reply)
    except BackendError:
        salvaged = salvage_queries(reply)
        if salvaged:
            return salvaged
        raise

    raw = payload.get("queries")
    if not isinstance(raw, list):
        salvaged = salvage_queries(reply)
        if salvaged:
            return salvaged
        raise BackendError("The model's reply had no 'queries' list.")
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"text": item})
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            out.append(item)
    if not out:
        return salvage_queries(reply)
    return out


async def generate_for_tool(
    backend: Backend,
    tool: Tool,
    tools: list[Tool],
    *,
    plan: GenerationPlan,
    seed: int | None = None,
    temperature: float = 0.8,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Query]:
    """Generate the query set for one tool."""
    neighbour = nearest_neighbour(tool, tools)
    prompt = build_prompt(tool, neighbour, plan)

    # One retry, cooler: a small local model that rambled once often complies when the
    # temperature drops. Failing a whole run because one of six calls trailed off mid-JSON
    # would waste every call before it.
    items: list[dict[str, Any]] = []
    last_error: BackendError | None = None
    for attempt in range(2):
        try:
            reply = await backend.generate(
                prompt,
                temperature=temperature if attempt == 0 else min(temperature, 0.2),
                seed=seed,
                max_tokens=max_tokens,
            )
            items = _parse_queries(reply)
        except BackendError as exc:
            last_error = exc
            items = []
        if items:
            break
    if not items:
        raise BackendError(
            f"Could not generate queries for {tool.qualified_name!r} after two attempts: "
            f"{last_error}"
        )

    queries: list[Query] = []
    counts = {"straightforward": 0, "paraphrased": 0, "hard_negative": 0}
    wanted = {
        "straightforward": plan.straightforward,
        "paraphrased": plan.paraphrased,
        "hard_negative": plan.hard_negative if neighbour else 0,
    }
    for item in items:
        kind = str(item.get("kind") or "straightforward").strip().lower()
        if kind not in counts:
            kind = "straightforward"
        if counts[kind] >= wanted[kind]:
            # The model over-delivered on one kind; keep the test set honest.
            continue
        counts[kind] += 1
        is_negative = kind == "hard_negative"
        expected = (
            neighbour.qualified_name if (is_negative and neighbour) else tool.qualified_name
        )
        queries.append(
            Query(
                id=f"{tool.qualified_name}#{kind}.{counts[kind]}",
                text=str(item["text"]).strip(),
                expected=expected,
                kind=kind,  # type: ignore[arg-type]
                neighbour=neighbour.qualified_name if neighbour else None,
            )
        )
    return queries


async def generate_no_tool(
    backend: Backend,
    tools: list[Tool],
    *,
    count: int = DEFAULT_NO_TOOL,
    seed: int | None = None,
    temperature: float = 0.8,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Query]:
    """Generate queries that no tool should answer."""
    if count <= 0:
        return []
    reply = await backend.generate(
        build_no_tool_prompt(tools, count),
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    try:
        items = _parse_queries(reply)
    except BackendError:
        return []
    return [
        Query(id=f"(none)#{i + 1}", text=str(item["text"]).strip(), expected=None, kind="no_tool")
        for i, item in enumerate(items[:count])
    ]


class QueryCache:
    """The editable YAML test set.

    Keyed per tool on the tool's fingerprint, so editing a description regenerates only
    that tool's queries and leaves everyone else's hand-edits alone.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": CACHE_VERSION, "tools": {}, "no_tool": []}
        if path.is_file():
            try:
                loaded = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                # A cache we cannot read is a cache we regenerate. Losing a generated test
                # set is annoying; refusing to run at all is worse.
                loaded = {}
            if isinstance(loaded, dict):
                self.data = {**self.data, **loaded}
        self.data.setdefault("tools", {})
        self.data.setdefault("no_tool", [])

    @classmethod
    def open(cls, directory: str | Path = CACHE_DIR) -> QueryCache:
        return cls(Path(directory) / CACHE_FILE)

    def fresh_for(self, tool: Tool) -> list[Query] | None:
        """Cached queries for this tool, if the tool has not changed since."""
        entry = self.data["tools"].get(tool.qualified_name)
        if not isinstance(entry, dict):
            return None
        if entry.get("fingerprint") != tool.fingerprint():
            return None
        raw = entry.get("queries") or []
        try:
            return [Query(**item) for item in raw]
        except Exception:
            return None

    def put(self, tool: Tool, queries: list[Query]) -> None:
        self.data["tools"][tool.qualified_name] = {
            "fingerprint": tool.fingerprint(),
            "queries": [q.model_dump() for q in queries],
        }

    def no_tool(self) -> list[Query]:
        try:
            return [Query(**item) for item in self.data.get("no_tool") or []]
        except Exception:
            return []

    def put_no_tool(self, queries: list[Query]) -> None:
        self.data["no_tool"] = [q.model_dump() for q in queries]

    def prune(self, tool_set: ToolSet) -> None:
        """Drop cached entries for tools that no longer exist."""
        live = set(tool_set.by_qualified_name())
        self.data["tools"] = {k: v for k, v in self.data["tools"].items() if k in live}

    def save(self, *, generator: str) -> Path:
        self.data["version"] = CACHE_VERSION
        self.data["generator"] = generator
        self.data["generated"] = dt.date.today().isoformat()
        self.data.setdefault(
            "_comment",
            "Edit these queries freely - they are your test set. A tool's queries are "
            "regenerated only when that tool's name, description or schema changes.",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True))
        return self.path


async def build_query_set(
    backend: Backend,
    tool_set: ToolSet,
    *,
    n: int = DEFAULT_N,
    no_tool_count: int = DEFAULT_NO_TOOL,
    cache: QueryCache | None = None,
    regenerate: bool = False,
    seed: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_progress: Any = None,
) -> list[Query]:
    """Get a full query set, generating only what the cache does not already have."""
    plan = GenerationPlan.for_n(n)
    tools = tool_set.sorted_tools()
    queries: list[Query] = []

    for tool in tools:
        cached = None if regenerate else (cache.fresh_for(tool) if cache else None)
        if cached:
            queries.extend(cached)
        else:
            fresh = await generate_for_tool(
                backend, tool, tools, plan=plan, seed=seed, max_tokens=max_tokens
            )
            if cache:
                cache.put(tool, fresh)
            queries.extend(fresh)
        if on_progress:
            on_progress(tool.qualified_name)

    cached_none = None if regenerate else (cache.no_tool() if cache else None)
    if cached_none:
        queries.extend(cached_none)
    elif no_tool_count > 0:
        fresh_none = await generate_no_tool(
            backend, tools, count=no_tool_count, seed=seed, max_tokens=max_tokens
        )
        if cache:
            cache.put_no_tool(fresh_none)
        queries.extend(fresh_none)

    if cache:
        cache.prune(tool_set)
    return queries
