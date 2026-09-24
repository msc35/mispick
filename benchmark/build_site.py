#!/usr/bin/env python3
"""Build the static leaderboard. Plain HTML, no framework, no external requests.

Wording rules, from benchmark/README.md: "most confused pair", never "worst server". Every
server links to its own repository, and every page states the model, N, K and the date, because
a score is a measurement of one model reading one tool surface - not a verdict on the people
who wrote it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from mispick.report import html as html_report
from mispick.report.load import from_payload

CSS = """
:root {
  --bg:#fbfbfd; --fg:#1c1c22; --muted:#6a6a78; --line:#e2e2ea; --card:#fff;
  --good:#14824a; --warn:#9a6b00; --bad:#b3261e; --accent:#5b4bd6;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#14141a;--fg:#ececf2;--muted:#9a9aab;--line:#2c2c38;--card:#1c1c24;
        --good:#4ade80;--warn:#fbbf24;--bad:#f87171;--accent:#a99bff;}
}
*{box-sizing:border-box}
body{margin:0;padding:40px 16px 72px;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
   margin:36px 0 12px;font-weight:600}
.lede{color:var(--muted);margin:0 0 6px;max-width:66ch}
.prov{color:var(--muted);font-size:13px;font-family:ui-monospace,Menlo,monospace;margin:0 0 28px}
a{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card);
  border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,monospace}
code{font-family:ui-monospace,Menlo,monospace;font-size:13px}
.score{font-weight:650}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.note{margin-top:32px;padding:14px 16px;border-left:3px solid var(--warn);background:var(--card);
  color:var(--muted);font-size:13px;border-radius:0 8px 8px 0;max-width:80ch}
.skipped td{color:var(--muted)}
footer{margin-top:44px;color:var(--muted);font-size:13px}
"""

CAVEAT = (
    "These are measurements of one model reading one tool surface on one day, not a judgement "
    "of the servers or the people who wrote them. A different model will produce different "
    "numbers, and a small local model is noisier than a large one. A tool surface that scores "
    "low here may be perfectly clear to a larger model, or may simply describe genuinely "
    "overlapping operations. Intervals are Wilson 95%."
)


def cls(value: float) -> str:
    return "good" if value >= 0.85 else "warn" if value >= 0.65 else "bad"


def render_index(index: dict[str, Any]) -> str:
    e = html.escape
    servers = index.get("servers") or []
    out: list[str] = []
    a = out.append
    a("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    a("<title>mispick benchmark</title>")
    a(f"<style>{CSS}</style></head><body><main>")
    a("<h1>Which MCP tools do models mix up?</h1>")
    a(
        "<p class='lede'>Each server's tool list was put in front of one model, which was asked "
        "to pick a tool for generated requests. No tool was ever called &mdash; only "
        "<code>initialize</code> and <code>tools/list</code>. Snapshots are committed, so every "
        "row can be reproduced.</p>"
    )
    a(
        f"<p class='prov'>model {e(str(index.get('model')))} &middot; "
        f"N={index.get('n')} &middot; K={index.get('k')} &middot; "
        f"temperature={index.get('temperature')} &middot; seed={index.get('seed')} &middot; "
        f"{e(str(index.get('generated')))}</p>"
    )

    a("<h2>Leaderboard</h2>")
    a("<table><thead><tr><th>Server</th><th class='n'>Tools</th><th class='n'>Score</th>"
      "<th class='n'>Accuracy</th><th>Most confused pair</th><th class='n'>Tool list</th>"
      "</tr></thead><tbody>")
    for server in servers:
        accuracy = server["accuracy"].get("value")
        acc_text = f"{accuracy * 100:.0f}%" if accuracy is not None else "n/a"
        pair = server.get("top_pair")
        pair_text = (
            f"<code>{e(pair['expected'])}</code> &rarr; <code>{e(pair['chosen'])}</code>"
            if pair
            else "<span class='good'>none</span>"
        )
        repo = server.get("repository")
        title = e(server["title"])
        name = f"<a href='{e(repo)}'>{title}</a>" if repo else title
        a(
            f"<tr><td>{name}<br><code style='color:var(--muted)'>"
            f"{e(server['registry_name'])}</code></td>"
            f"<td class='n'>{server['tool_count']}</td>"
            f"<td class='n score {cls(server['score'] / 100)}'>"
            f"<a href='servers/{e(server['slug'])}.html'>{server['score']}</a></td>"
            f"<td class='n'>{acc_text}</td>"
            f"<td>{pair_text}</td>"
            f"<td class='n'>~{server['tokens']} tok</td></tr>"
        )
    a("</tbody></table>")

    skipped = index.get("skipped") or []
    if skipped:
        a("<h2>Not measured</h2>")
        a(
            "<p class='lede'>These were in the candidate list but could not be measured. "
            "Listed so the coverage above is not overstated.</p>"
        )
        a("<table><thead><tr><th>Server</th><th class='n'>Tools</th><th>Why</th>"
          "</tr></thead><tbody>")
        for server in skipped:
            title = e(str(server.get("title") or server.get("registry_name") or "unknown"))
            repo = server.get("repository")
            name = f"<a href='{e(repo)}'>{title}</a>" if repo else title
            status = str(server.get("status") or "failed").replace("_", " ")
            reason = e(str(server.get("reason") or ""))
            count = server.get("tool_count")
            a(
                f"<tr class='skipped'><td>{name}</td>"
                f"<td class='n'>{count if count is not None else '&mdash;'}</td>"
                f"<td>{e(status)}{f' &mdash; {reason}' if reason else ''}</td></tr>"
            )
        a("</tbody></table>")
        a(
            "<p class='legend'>A server that would not start is a fact about running it on a "
            "clean machine, not a judgement of the project - it may need setup this benchmark "
            "did not do. Servers listed as <em>too many tools</em> were measured only for their "
            "token cost, which needs no model call.</p>"
        )

    a(f"<div class='note'><strong>Read this before quoting a number.</strong> {CAVEAT}</div>")
    a(
        "<footer>Measured with <a href='https://github.com/msc35/mispick'>mispick</a>. "
        "Something look wrong? "
        "<a href='https://github.com/msc35/mispick/issues/new?template=benchmark-rerun.md'>"
        "Request a re-run.</a></footer>"
    )
    a("</main></body></html>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/site"))
    args = parser.parse_args()

    index_path = args.results / "index.json"
    if not index_path.is_file():
        print(f"error: {index_path} not found. Run measure.py first.")
        return 2
    index = json.loads(index_path.read_text())

    site = args.out
    (site / "servers").mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text(render_index(index))
    # Tell GitHub Pages not to run Jekyll over it.
    (site / ".nojekyll").write_text("")

    for server in index.get("servers") or []:
        payload = json.loads((args.results / f"{server['slug']}.json").read_text())
        result, metrics = from_payload(payload)
        page = html_report.render(result, metrics)
        page = page.replace(
            "</main>",
            "<footer style='margin-top:40px;color:var(--muted);font-size:13px'>"
            "<a href='../index.html'>&larr; back to the leaderboard</a></footer></main>",
        )
        (site / "servers" / f"{server['slug']}.html").write_text(page)

    print(f"built {len(index.get('servers') or [])} pages -> {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
