"""A single self-contained HTML file. No CDN, no fonts, no JS framework - opens offline.

SPEC section 7 requires it to work from a file:// URL with no network, so everything is
inline and there is no external request of any kind.
"""

from __future__ import annotations

import html
import json

from mispick.metrics import NONE, PHANTOM, Metrics, top_confused_pairs
from mispick.report.provenance import provenance_of
from mispick.select import RunResult

_CSS = """
:root {
  --bg: #fbfbfd; --fg: #1c1c22; --muted: #6a6a78; --line: #e2e2ea;
  --card: #ffffff; --good: #14824a; --warn: #9a6b00; --bad: #b3261e;
  --heat0: #f4f4f8; --shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14141a; --fg: #ececf2; --muted: #9a9aab; --line: #2c2c38;
    --card: #1c1c24; --good: #4ade80; --warn: #fbbf24; --bad: #f87171;
    --heat0: #22222c; --shadow: none;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }
h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
     margin: 36px 0 12px; font-weight: 600; }
.prov { color: var(--muted); font-size: 13px; margin: 0 0 24px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 20px; box-shadow: var(--shadow); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile .label { font-size: 12px; color: var(--muted); text-transform: uppercase;
               letter-spacing: .06em; }
.tile .value { font-size: 26px; font-weight: 650; letter-spacing: -.02em; margin-top: 2px; }
.tile .ci { font-size: 12px; color: var(--muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.score .value { font-size: 40px; }
.good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); }
th { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
     font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.matrix { overflow-x: auto; }
.matrix table { width: auto; min-width: 100%; }
.matrix th.col { writing-mode: vertical-rl; text-orientation: mixed; height: 110px;
                 vertical-align: bottom; padding: 8px 4px; white-space: nowrap; }
.matrix td { text-align: center; border: 1px solid var(--line); padding: 0;
             width: 42px; height: 34px; }
.matrix td div { width: 100%; height: 100%; display: flex; align-items: center;
                 justify-content: center; font-variant-numeric: tabular-nums;
                 font-family: ui-monospace, monospace; font-size: 13px; }
.matrix th.row { white-space: nowrap; border: 1px solid var(--line); }
.zero { color: var(--muted); opacity: .45; }
.diag { outline: 2px solid var(--good); outline-offset: -2px; }
.legend { color: var(--muted); font-size: 13px; margin-top: 10px; }
.caveat { margin-top: 40px; padding: 14px 16px; border-left: 3px solid var(--warn);
          background: var(--card); color: var(--muted); font-size: 13px;
          border-radius: 0 8px 8px 0; }
.bar { display: inline-block; height: 8px; border-radius: 4px; background: var(--good);
       vertical-align: middle; }
.bar.w { background: var(--warn); } .bar.b { background: var(--bad); }
.track { display: inline-block; width: 90px; height: 8px; border-radius: 4px;
         background: var(--heat0); vertical-align: middle; }
"""


def _cls(value: float) -> str:
    return "good" if value >= 0.8 else "warn" if value >= 0.5 else "bad"


def _heat_style(count: int, row_total: int, correct: bool) -> str:
    """Background for one matrix cell."""
    if count == 0:
        return "background: var(--heat0);"
    share = count / row_total if row_total else 0.0
    alpha = 0.12 + 0.72 * share
    rgb = "20, 130, 74" if correct else "179, 38, 30"
    return f"background: rgba({rgb}, {alpha:.3f});"


def render(result: RunResult, metrics: Metrics) -> str:
    prov = provenance_of(result, metrics.token_estimate)
    e = html.escape
    score = metrics.score
    score_cls = "good" if score >= 90 else "warn" if score >= 75 else "bad"

    out: list[str] = []
    a = out.append
    a("<!doctype html>")
    a('<html lang="en"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width, initial-scale=1">')
    a(f"<title>mispick · {e(prov.servers)}</title>")
    a(f"<style>{_CSS}</style>")
    a("</head><body><main>")

    a(f"<h1>mispick · {e(prov.servers)}</h1>")
    a(f'<p class="prov">{e(prov.one_line)} · {prov.tool_count} tools</p>')

    # Summary tiles
    a('<div class="tiles">')
    a(
        f'<div class="card tile score"><div class="label">Score</div>'
        f'<div class="value {score_cls}">{score}<span style="font-size:18px;color:var(--muted)">'
        f"/100</span></div></div>"
    )
    for label, rate in (
        ("Accuracy", metrics.accuracy),
        (f"Stability (K={prov.k})", metrics.stability),
        ("Argument validity", metrics.arg_validity),
    ):
        if not rate.total:
            a(
                f'<div class="card tile"><div class="label">{e(label)}</div>'
                f'<div class="value" style="color:var(--muted)">n/a</div></div>'
            )
            continue
        low, high = rate.interval
        a(
            f'<div class="card tile"><div class="label">{e(label)}</div>'
            f'<div class="value {_cls(rate.value)}">{rate.pct:.0f}%</div>'
            f'<div class="ci">95% CI {low * 100:.0f}–{high * 100:.0f}% · '
            f"{rate.hits}/{rate.total}</div></div>"
        )
    a("</div>")

    # Confusion matrix
    a("<h2>Confusion matrix</h2>")
    a('<div class="card matrix"><table><thead><tr>')
    a('<th class="row">intended ↓ / chosen →</th>')
    for col in metrics.column_labels:
        a(f'<th class="col">{e(col)}</th>')
    a("</tr></thead><tbody>")
    for row in metrics.row_labels:
        cells = metrics.matrix.get(row, {})
        row_total = sum(cells.values())
        a(f'<tr><th class="row">{e(row)}</th>')
        for col in metrics.column_labels:
            count = cells.get(col, 0)
            correct = col == row
            style = _heat_style(count, row_total, correct)
            klass = "diag" if correct and count else ""
            inner = str(count) if count else '<span class="zero">·</span>'
            a(f'<td class="{klass}" style="{style}"><div>{inner}</div></td>')
        a("</tr>")
    a("</tbody></table>")
    a(
        '<p class="legend">Green on the diagonal is a correct pick. Red off the diagonal is a '
        "confusion; the darker the cell, the larger the share of that tool's trials it took. "
        f"<code>{NONE}</code> means no tool was chosen, <code>{PHANTOM}</code> means the model "
        "named a tool that does not exist.</p>"
    )
    a("</div>")

    # Cross-server
    from mispick.crossserver import analyse as analyse_cross

    cross = analyse_cross(result, metrics)
    if cross.is_multi_server:
        a("<h2>Across servers</h2>")
        a('<div class="card">')
        a('<table><thead><tr><th>Server</th><th class="n">Tools</th>'
          '<th class="n">Accuracy on its own tools</th><th class="n">Work lost</th>'
          '<th class="n">Work stolen</th></tr></thead><tbody>')
        for entry in sorted(cross.servers, key=lambda s: s.accuracy.value):
            a(
                f"<tr><td><code>{e(entry.label)}</code></td>"
                f'<td class="n">{entry.tool_count}</td>'
                f'<td class="n">{entry.accuracy}</td>'
                f'<td class="n">{entry.lost or "·"}</td>'
                f'<td class="n">{entry.stolen or "·"}</td></tr>'
            )
        a("</tbody></table>")
        if cross.collisions:
            a("<p><strong>Name collisions.</strong> The same bare tool name on more than one "
              "server:</p><ul>")
            for c in cross.collisions:
                note = (
                    " &mdash; and their descriptions are identical"
                    if c.identical_descriptions
                    else ""
                )
                servers = ", ".join(f"<code>{e(s)}</code>" for s in c.servers)
                a(f"<li><code>{e(c.name)}</code> on {servers}{note}</li>")
            a("</ul>")
            a('<p class="legend">The MCP spec tells clients to disambiguate these by prefixing '
              "the server name, and warns that <code>serverInfo.name</code> is not unique "
              "across servers &mdash; so mispick keys on your config label instead.</p>")
        if cross.leaks:
            a(f"<p><strong>{cross.cross_server_rate} of trials went to the wrong server.</strong>"
              "</p><ul>")
            for expected, chosen, count in cross.leaks[:8]:
                a(f"<li><code>{e(expected)}</code> &rarr; <code>{e(chosen)}</code> "
                  f"({count} trials)</li>")
            a("</ul>")
        else:
            a('<p class="good">No trial crossed from one server to another.</p>')
        a("</div>")

    # Confused pairs
    pairs = top_confused_pairs(metrics, limit=10)
    a("<h2>Most confused pairs</h2>")
    a('<div class="card">')
    if pairs:
        a('<table><thead><tr><th>Intended</th><th>Chosen instead</th>'
          '<th class="n">Trials</th><th class="n">Share</th></tr></thead><tbody>')
        for p in pairs:
            a(
                f"<tr><td><code>{e(p.expected)}</code></td>"
                f"<td><code>{e(p.chosen)}</code></td>"
                f'<td class="n">{p.count}</td>'
                f'<td class="n">{p.share * 100:.0f}%</td></tr>'
            )
        a("</tbody></table>")
        a('<p class="legend">Run <code>mispick fix</code> for a re-tested rewrite of these '
          "descriptions.</p>")
    else:
        a('<p class="good">No tool was mistaken for another tool.</p>')
    a("</div>")

    # Per tool
    a("<h2>Per tool</h2>")
    a('<div class="card"><table><thead><tr><th>Tool</th><th class="n">Recall</th>'
      '<th class="n">Precision</th><th>Recall</th></tr></thead><tbody>')
    for name, tool in sorted(metrics.per_tool.items(), key=lambda kv: kv[1].recall.value):
        width = round(tool.recall.value * 90)
        bar_cls = {"good": "", "warn": "w", "bad": "b"}[_cls(tool.recall.value)]
        a(
            f"<tr><td><code>{e(name)}</code></td>"
            f'<td class="n">{tool.recall}</td>'
            f'<td class="n">{tool.precision}</td>'
            f'<td><span class="track"><span class="bar {bar_cls}" '
            f'style="width:{width}px"></span></span></td></tr>'
        )
    a("</tbody></table></div>")

    # Notes
    notes: list[str] = []
    if metrics.over_trigger.total:
        notes.append(
            f"Called a tool on {metrics.over_trigger} of the &ldquo;no tool fits&rdquo; queries."
        )
    if metrics.phantom_rate.hits:
        notes.append(f"Named a tool that does not exist in {metrics.phantom_rate} of trials.")
    if metrics.unstable_queries:
        notes.append(
            f"{len(metrics.unstable_queries)} queries changed answer across {prov.k} runs."
        )
    if metrics.errored_trials:
        notes.append(
            f"{metrics.errored_trials} of {metrics.trials} trials errored and were excluded."
        )
    notes.append(
        f"The tool list costs roughly {metrics.token_estimate} tokens per request "
        "(an estimate, and a lower bound)."
    )
    if not prov.deterministic:
        notes.append(
            f"Temperature {prov.temperature:g}, so this run is not reproducible. "
            "Use <code>--temperature 0 --seed N</code>."
        )
    a("<h2>Notes</h2>")
    a('<div class="card"><ul style="margin:0;padding-left:20px">')
    for note in notes:
        a(f"<li>{note}</li>")
    a("</ul></div>")

    a(f'<div class="caveat"><strong>Read this before quoting the number.</strong> {e(prov.caveat)} '
      "Intervals are Wilson 95%.</div>")

    # The raw data, so the file is self-describing.
    from mispick.report import json as json_report

    # A tool description is untrusted input. Inside a <script> block, HTML escaping does not
    # apply but `</script>` still ends the element - so a description containing one would
    # break out and the rest of the payload would render as live markup. Escaping `<` to
    # \u003c is still valid JSON and closes that hole.
    payload = json.dumps(json_report.build(result, metrics)).replace("<", "\\u003c")
    a(f'<script type="application/json" id="mispick-data">{payload}</script>')
    a("</main></body></html>")
    return "\n".join(out) + "\n"
