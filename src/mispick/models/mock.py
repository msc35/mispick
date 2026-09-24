"""A deterministic backend, so the test suite runs offline.

It does not pretend to be a language model. It mimics the *failure* we care about: when
two tools have the same description, it cannot tell them apart and falls back to whichever
comes first alphabetically. That is enough to exercise the whole pipeline - confusion
matrix, stability, fix mode - with no network and no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re

from mispick.models.base import Backend, Selection, wire_name
from mispick.types import Tool

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


#: Words that carry no signal about which tool is meant.
_STOP = _words(
    "a an the of to for from in on at by with and or please can you i want need would like"
    " me my our us it this that there help how do does me the some any all new"
)


class MockBackend(Backend):
    """Scores tools by word overlap between the query and the description."""

    name = "mock"
    supports_seed = True

    def __init__(self, *, jitter: bool = False) -> None:
        #: When set, the K runs disagree for near-ties, so stability is measurable.
        self.jitter = jitter
        self.calls = 0

    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        self.calls += 1
        q = _words(query) - _STOP
        scored: list[tuple[float, str]] = []
        for tool in tools:
            haystack = _words(f"{tool.name} {tool.description or ''}") - _STOP
            overlap = len(q & haystack)
            # A name match is worth more than a description match.
            name_hits = len(q & (_words(tool.name) - _STOP))
            score = float(overlap + 2 * name_hits)
            if self.jitter and temperature > 0:
                digest = hashlib.sha256(
                    f"{seed}:{self.calls}:{tool.name}:{query}".encode()
                ).digest()
                score += (digest[0] / 255.0) * 1.5
            scored.append((score, tool.qualified_name))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        best_score, best = scored[0]
        if best_score <= 0:
            # Nothing matched: this is the "no tool fits" answer.
            return Selection(chosen=None, raw="no tool")

        tool = next(t for t in tools if t.qualified_name == best)
        return Selection(
            chosen=wire_name(tool),
            arguments=_plausible_args(tool, query),
            raw=f"mock chose {best} (score {best_score:.2f})",
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Return plausible JSON for whichever prompt shape asked for it.

        Queries are built from the tool's *description*, never its name - exactly as the
        real generation prompt demands ("No tool names in the text"). That matters: if the
        mock leaked the tool name into the query text, two tools with identical
        descriptions would still be trivially separable and the mock would never reproduce
        the confusion this whole tool exists to find.
        """
        if "REWRITE" in prompt:
            return json.dumps(
                {
                    "descriptions": {
                        name: f"{name.replace('_', ' ').capitalize()}. "
                        f"Use this only for {name.split('_')[-1]}, not for anything else."
                        for name in _quoted_names(prompt)
                    },
                    "cause": "unclear_purpose",
                }
            )

        if "NONE of those tools" in prompt:
            return json.dumps(
                {
                    "queries": [
                        {"text": "What is your refund policy in plain language?"},
                        {"text": "Explain how our escalation process works."},
                        {"text": "Who owns the billing roadmap this quarter?"},
                    ]
                }
            )

        description = _described(prompt)
        neighbour = _neighbour_name(prompt)
        subject = _subject(description)
        queries = [
            {"text": f"{description.rstrip('.')} about {subject}.", "kind": "straightforward"},
            {"text": f"I need to {description[0].lower()}{description[1:]}", "kind":
             "straightforward"},
            {"text": f"Can you {description[0].lower()}{description[1:].rstrip('.')} now?",
             "kind": "straightforward"},
            {"text": f"Deal with the {subject} please.", "kind": "straightforward"},
            {"text": f"Sort out {subject} for me, whatever that takes.", "kind": "paraphrased"},
            {"text": f"Something needs doing about {subject}.", "kind": "paraphrased"},
        ]
        if neighbour:
            other = _subject(neighbour.replace("_", " "))
            queries += [
                {"text": f"Actually this one is about {other}.", "kind": "hard_negative"},
                {"text": f"Handle the {other} side of things.", "kind": "hard_negative"},
            ]
        return json.dumps({"queries": queries})


def _quoted_names(prompt: str) -> list[str]:
    """Pull tool names out of a prompt, so the mock can answer about the right thing."""
    return re.findall(r"^\s*(?:name|tool):\s*([A-Za-z0-9_.:-]+)", prompt, re.MULTILINE)


def _described(prompt: str) -> str:
    """The description line from a generation prompt."""
    match = re.search(r"^\s*description:\s*(.+)$", prompt, re.MULTILINE)
    text = (match.group(1).strip() if match else "").strip()
    if not text or text == "(none)":
        return "Do the thing"
    return text


def _neighbour_name(prompt: str) -> str | None:
    match = re.search(r"nearest other tool is named:\s*([A-Za-z0-9_.:-]+)", prompt)
    return match.group(1) if match else None


def _subject(text: str) -> str:
    """The most content-bearing words of a description, for query filler."""
    words = [w for w in _words(text) - _STOP if len(w) > 3]
    return " ".join(sorted(words)[:2]) or "that"


def _plausible_args(tool: Tool, query: str) -> dict[str, object]:
    """Fill required properties with type-appropriate junk, so arg validation passes."""
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    args: dict[str, object] = {}
    for key in schema.get("required") or []:
        spec = props.get(key) or {}
        kind = spec.get("type", "string")
        if spec.get("enum"):
            args[key] = spec["enum"][0]
        elif kind == "integer":
            args[key] = 1
        elif kind == "number":
            args[key] = 1.0
        elif kind == "boolean":
            args[key] = True
        elif kind == "array":
            args[key] = []
        elif kind == "object":
            args[key] = {}
        else:
            args[key] = query[:40]
    return args


class ScriptedBackend(Backend):
    """Replays a fixed list of answers. For tests that need an exact sequence."""

    name = "scripted"
    supports_seed = True

    def __init__(self, answers: list[str | None], generations: list[str] | None = None) -> None:
        self.answers = list(answers)
        self.generations = list(generations or [])
        self.seen: list[str] = []

    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        self.seen.append(query)
        if not self.answers:
            return Selection(chosen=None, raw="script exhausted")
        answer = self.answers.pop(0)
        if answer is None:
            return Selection(chosen=None, raw="no tool")
        tool = next((t for t in tools if t.qualified_name == answer or t.name == answer), None)
        if tool is None:
            # Deliberate phantom.
            return Selection(chosen=answer, raw="phantom")
        return Selection(
            chosen=wire_name(tool), arguments=_plausible_args(tool, query), raw="scripted"
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> str:
        if self.generations:
            return self.generations.pop(0)
        return "{}"
