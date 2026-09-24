# Where to submit mispick, and where not to

Prepared 2026-09-24. Nothing here has been submitted — each is a PR to somebody else's repository.

## The one that fits: `punkpeye/awesome-mcp-devtools`

484 stars. Describes itself as "developer tools, SDKs, libraries, and **testing utilities** for
Model Context Protocol (MCP) server development", and has a `## Testing Tools` section holding 18
entries including `mclenhard/mcp-evals`. Neither `whichtool` nor `toolfit` is listed, so mispick
would be the first measured-tool-selection entry there.

Add to **`## Testing Tools`**, keeping the section's existing order and format
(`- [owner/repo](url) <language emoji> - Description.`; 🐍 is the Python tag):

```markdown
- [msc35/mispick](https://github.com/msc35/mispick) 🐍 - Measures which of your tools a model actually mixes up. Generates test requests per tool including hard negatives, asks a model to choose, and reports a confusion matrix with per-tool precision and recall at Wilson 95% intervals. Proposes description rewrites for confused pairs and re-tests them, keeping only measured improvements. Also finds collisions across every server in a config. Never calls a tool — `initialize` and `tools/list` only — so it is safe against destructive servers. Free on a local Ollama model. `uvx mispick run --cmd "..."`
```

Steps, from their CONTRIBUTING: fork, branch (`add-mispick`), edit `README.md`, commit
("Add mispick devtool"), push, open a PR.

```bash
gh repo fork punkpeye/awesome-mcp-devtools --clone --remote
cd awesome-mcp-devtools
git checkout -b add-mispick
# edit README.md, adding the line above to ## Testing Tools
git commit -am "Add mispick devtool"
git push -u origin add-mispick
gh pr create --title "Add mispick devtool" --body "..."
```

## Where mispick does *not* belong

**`punkpeye/awesome-mcp-servers`** (95k stars) — tempting because of the reach, but its
CONTRIBUTING is explicit: *"This list is for servers with a public GitHub repository — something
you install and run yourself."* mispick is a client that reads `tools/list`; it is not a server.
Its `## Frameworks` section does carry some non-servers, and the maintainer points from there to
the devtools list above, which is the intended route. Submitting to the servers list would be
asking for a rejection and spending goodwill to get it.

**`wong2/awesome-mcp-servers`, `appcypher/awesome-mcp-servers`,
`jaw9c/awesome-remote-mcp-servers`** — all server lists, same reason.

**The official MCP registry** — declined for the same reason; see `docs/launch-checklist.md`.

## Re-check before submitting

- That `## Testing Tools` still exists and still uses this format.
- That nobody has added mispick already.
- Whether the list wants alphabetical order within the section — CONTRIBUTING mentions keeping
  categories alphabetical; entries in Testing Tools currently are not, so append rather than
  reorder someone else's list.
