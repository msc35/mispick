---
name: Request a benchmark re-run
about: Ask for a server to be re-measured, or point out a result that looks wrong
title: 'Re-run: <server name>'
labels: benchmark
---

**Server**
<!-- The registry name as it appears on the leaderboard, e.g. io.github.owner/repo -->

**Why**
<!-- Pick whichever applies:
 - the descriptions changed since the snapshot was captured
 - the snapshot is missing tools, or has tools that no longer exist
 - the result looks wrong for a reason you can name
 - the server was skipped but does start without credentials
-->

**Anything we should know about running it**
<!-- e.g. it needs a flag, or it only exposes its full tool list after some setup -->

---

A reminder on what the benchmark is: one model, one seed, one day, reading only
`initialize` and `tools/list`. A low score is a finding about how that model reads a tool
surface, not a judgement of the server. If you think the framing on your server's page is
unfair, say so here and we will fix the wording.
