"""A shields-style SVG badge. Self-contained, no network, no shields.io call."""

from __future__ import annotations

from mispick.metrics import Metrics

#: Shields' palette.
GREEN = "#4c1"
YELLOW_GREEN = "#a4a61d"
YELLOW = "#dfb317"
ORANGE = "#fe7d37"
RED = "#e05d44"
GREY = "#555"


def colour_for(score: int) -> str:
    if score >= 95:
        return GREEN
    if score >= 85:
        return YELLOW_GREEN
    if score >= 70:
        return YELLOW
    if score >= 50:
        return ORANGE
    return RED


def _width(text: str) -> int:
    """Approximate rendered width at 11px DejaVu Sans, in tenths of a pixel."""
    narrow = set("iljtfIr1.,:;'\"!|()[]{} ")
    wide = set("mwMW@")
    total = 0
    for ch in text:
        if ch in narrow:
            total += 40
        elif ch in wide:
            total += 100
        else:
            total += 70
        total += 5
    return total


def render(metrics: Metrics, *, model: str, label: str = "mispick") -> str:
    """An SVG badge showing the score and the model it was measured with.

    The model belongs on the badge: a score without a model is not a fact about the server.
    """
    score = metrics.score
    right_text = f"{score}/100 · {model}"
    colour = colour_for(score)

    left_w = round(_width(label) / 10) + 10
    right_w = round(_width(right_text) / 10) + 10
    total = left_w + right_w

    left_centre = left_w * 5
    right_centre = left_w * 10 + right_w * 5

    return f"""<svg xmlns="http://www.w3.org/2000/svg" \
xmlns:xlink="http://www.w3.org/1999/xlink" width="{total}" height="20" \
role="img" aria-label="{label}: {right_text}">
  <title>{label}: {right_text}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{left_w}" height="20" fill="{GREY}"/>
    <rect x="{left_w}" width="{right_w}" height="20" fill="{colour}"/>
    <rect width="{total}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" \
font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="110" \
text-rendering="geometricPrecision" transform="scale(.1)">
    <text x="{left_centre}" y="150" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{left_centre}" y="140">{label}</text>
    <text x="{right_centre}" y="150" fill="#010101" fill-opacity=".3">{right_text}</text>
    <text x="{right_centre}" y="140">{right_text}</text>
  </g>
</svg>
"""
