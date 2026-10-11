"""Text the reviewer and the human read.

The reviewer's view is blind: shuffled names, no radii. A model told that a
walk is a long shot looks less hard, and the empty it returns would shrink the
direction for the wrong reason. The human's view (`show`) has every number.
"""

from __future__ import annotations

from .config import REVIEW
from .graph import Graph


def plan_view(g: Graph, pending: dict) -> str:
    known = sorted(d.name for d in g.live() if d.name not in pending["walk"])
    lines = [
        f"review {g.n + 1} · plan {pending['id']}",
    ]
    if pending.get("changed"):
        lines.append(f"scope    only the changed files: {', '.join(pending['changed'])}")
    lines.append("look     " + (", ".join(pending["walk"]) or "(nothing assigned; everything comes from fan-out)"))
    recipes = [(name, g.recipes[name]) for name in pending["walk"] if name in g.recipes]
    if recipes:
        lines.append("recipes  the rule to check in these directions, from the pack")
        for name, recipe in recipes:
            lines.append(f"  {name}")
            for label, value in recipe:
                lines.append(f"    {label + ':':<7}{value}")
    lines += [
        f"fan out  look where none of these point. Name what you find from: {', '.join(known) or '(none yet)'}",
        f"         or a new name with 1-{REVIEW['parents']} parents from the names above;"
        f" up to {REVIEW['births']} new names. Several findings may share one name.",
    ]
    return "\n".join(lines) + "\n"


def table(g: Graph, pack: str) -> str:
    """The network: families under their parents, lit or dark, plus health."""
    x = g.x()
    live = g.live()
    names = {d.name for d in live}
    kids: dict[str | None, list] = {}
    for d in live:
        kids.setdefault(d.parent if d.parent in names else None, []).append(d)
    out = [f"{pack} · {len(live)} directions · {g.n} reviews", ""]

    def walk(parent: str | None, depth: int) -> None:
        for d in sorted(kids.get(parent, []), key=lambda d: -d.radius(x)):
            r = d.radius(x)
            state = "dark" if r < REVIEW["dark"] else "lit "
            pad = "  " * depth + ("└ " if depth else "")
            bar = "█" * max(1, round(r * 40))
            # A second parent is not another nest: without this, the table hides it.
            also = f" (also under {d.parents[1]})" if len(d.parents) > 1 and d.parents[1] in names else ""
            out.append(f"{state} {bar:<16} {r:.2f}  {pad + d.name:<30} {d.hits:.1f}/{d.walks}{also}")
            walk(d.name, depth + 1)

    walk(None, 0)
    h = g.report()
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    out += [
        "",
        f"health (last reviews)  log-loss {fmt(h['loss'])} vs baseline {fmt(h['baseline'])}"
        f" · misses/review {fmt(h['misses_per_review'])}"
        f" · false-empty {fmt(h['false_empty_rate'])} over {h['audits']} audits"
        f" · human 'wrong' {fmt(h['feedback_wrong_rate'])} over {h['feedback']}",
    ]
    return "\n".join(out) + "\n"
