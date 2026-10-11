---
name: se-demux
description: Reviews code for defects, steered by a neural engine that learns where to look. It starts from eight broad software-engineering domains, lights the directions that keep paying off across every review, explores uncertain ones, and grows new directions from what reviews find. Language- and framework-agnostic. Use when asked to review code, a diff, a PR, or a project. Not for formatting, linting, or writing fixes.
license: MIT
metadata:
  version: "8"
---

# se-demux

The engine tells you where to look. When a direction is shown with a recipe, that recipe is what you check; otherwise you already know. Run commands from this skill's directory. Write `payload.json` (and `changed.txt`) outside the project.

**1. Plan.**

```sh
uv run neural-engine plan --root <project>                          # the whole project
uv run neural-engine plan --root <project> --changed changed.txt    # a diff: changed paths, one per line
```

To review with another pack, set `NEURAL_ENGINE_PACK=<folder>` for every command.

If it says `unchanged since review N: ...`, stop: the earlier findings still stand.

**2. Look.** Read the code once, then walk every direction under `look`, each with equal care. Each walk is `hit` or `empty`, with a `site` (`relpath:line`) and a `quote` copied from that line. A direction shown with a recipe is checked against that recipe, and a hit's claim names the rule it breaks.
- A hit adds `sev` and a `claim`: the input or state, and the wrong outcome.
- An empty walk cites where you looked; in a diff review, a line in a changed file.

**3. Fan out.** Record defects the plan didn't point to, under the known name that says how it fails. If none fits, use a new name (1–4 lowercase words joined by hyphens) with `parents`: one or two known names it narrows. One root cause, one record: no two findings share a line. Report every finding, recorded or not.

**4. Verify.** Give the brief to a fresh subagent, unchanged. It shows the lines around each finding, so the refuter judges from those and doesn't read the project again.

```sh
uv run neural-engine verify --from payload.json
```

From its answer, add to the payload:
- `check` on every hit: `reproduced`, `plausible` or `refuted`;
- `same_as` on a new name it matched to a known one;
- `merge` for a recently born duplicate;
- the `audit` block, when the brief asked for one.

Downgrade or drop findings in your report to match its verdicts.

**5. Apply.** `uv run neural-engine schema payload` prints the exact shape.

```json
{"plan": "<id>",
 "walks": [{"node": "concurrency", "verdict": "hit", "sev": "major", "site": "src/orders:88",
            "quote": "order.status = POSTED", "claim": "two requests posting the same order both pass the status check at :80 and post it twice", "check": "reproduced"},
           {"node": "maintainability", "verdict": "empty", "site": "src/orders:12", "quote": "function post(order)"}],
 "fanout": [{"node": "role-checks", "parents": ["security"], "sev": "major", "site": "src/admin:31",
             "quote": "route GET /admin/export", "claim": "the export route checks login but not role, so any signed-in user downloads every customer", "check": "plausible"}],
 "merge": []}
```

```sh
uv run neural-engine apply --from payload.json
```

If it refuses, fix what it names. Never edit the engine's files by hand.

**Severity** is blast radius: critical (data loss, corruption, breach, outage), major (a likely defect with a concrete failure), minor (contained), info (unclear; recorded as nothing found).

**6. Report** findings worst-first: severity, `file:line`, direction, what breaks, the refuter's verdict, the fix. End with the `apply` output, and tell the user that `uv run neural-engine feedback <id> fixed` or `uv run neural-engine feedback <id> wrong` teaches the engine.
