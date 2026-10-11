"""A synthetic world to test the network against the failures a real reviewer brings.

    python -m neural_engine.sim [reviews]

Code fails in the same ways everywhere: each direction has one true defect rate,
varied a little from project to project. Twelve specific directions the seed lacks
must be discovered by fan-out, and a few names a reviewer invents never pay and must
be deleted. Code keeps its defects until fixed, so reviews repeat, and some reviews
see only a diff. The simulated reviewer misses defects, invents some, notices some
directions more than others, uses synonyms, and sometimes files a specific defect
under its broad domain instead of naming it. A refuter catches part of that.
`Behavior` switches each failure on or off, so every design choice can be scored
against its absence.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Any

from . import pack as pack_mod
from .config import CHECK, REVIEW, SEV
from .graph import Graph, seeded

_SPEC = json.loads((pack_mod.DEFAULT / "sim.json").read_text(encoding="utf-8"))
PROJECTS: int = _SPEC["projects"]
SPREAD: float = _SPEC["spread"]  # log-normal sd of one project's rate around the true one
TRUTH: dict[str, float] = _SPEC["truth"]
NEAREST: dict[str, str] = _SPEC["nearest"]  # discovery -> the generic seed a reviewer might wrongly file it under
DISCOVERIES = list(NEAREST)
SYNONYMS: dict[str, str] = _SPEC["synonyms"]  # canonical -> other name
JUNK: dict[str, str] = _SPEC["junk"]  # invented name -> the seed a reviewer hangs it under; it never pays
PACK = pack_mod.load()


@dataclass
class Behavior:
    blind: bool = True  # the reviewer cannot see radii
    effort_follows_odds: bool = False  # ...and if it can, it slacks on long shots
    fix_rate: float = 0.3  # chance a defect is fixed before the next visit
    dedup: bool = True
    miss_rate: float = 0.2  # reviewer misses a real defect on a walk
    false_rate: float = 0.05  # reviewer invents a defect on a walk
    invent: float = 0.1  # per review, the reviewer files an invented defect under a name that never pays
    notice: float = 0.3  # chance fan-out notices an unassigned direction with a defect
    snap_rate: float = 0.4  # reviewer files a discovery under its nearest generic name instead of naming it
    synonym_rate: float = 0.5  # reviewer uses the other name for a synonym pair
    same_as_rate: float = 0.8  # refuter recognises the synonym at birth
    merge_rate: float = 0.5  # ...and, per review, a duplicate that slipped through earlier
    refute_false: float = 0.7  # refuter refutes an invented defect (else calls it plausible)
    reproduce_real: float = 0.8  # refuter reproduces a real one (else plausible)
    audit_catch: float = 0.5  # refuter finds a missed defect on the audited empty walk
    audit_share: float = REVIEW["audit_share"]  # reviews that audit one empty walk
    diff_share: float = 0.0  # reviews that examine only a diff...
    diff_seen: float = 0.25  # ...where each present defect is visible with this chance: the diff's share of the code
    diff_empties: str = "scaled"  # an empty in a diff counts for its share | "full" (counts as a whole review) | "drop"
    parents: str = "honest"  # births declare the nearest generic parent | "hostile" (an unrelated seed) | "none"
    shift: tuple[str, float, float, int] | None = None  # (direction, rate before, rate from review n on, n)


def truth(name: str, n: int = 0, b: Behavior | None = None) -> float:
    if b and b.shift and b.shift[0] == name:
        return b.shift[1] if n < b.shift[3] else b.shift[2]
    return TRUTH.get(name, 0.0)


def run(reviews: int = 150, seed: int = 7, b: Behavior | None = None, trace: dict | None = None) -> Graph:
    b = b or Behavior()
    return _run(reviews, random.Random(seed), b, trace if trace is not None else {})


def _run(reviews: int, rng: random.Random, b: Behavior, trace: dict) -> Graph:
    g = seeded(PACK)
    seeds = [s.name for s in PACK.seeds]
    lean = [{name: math.exp(rng.gauss(0, SPREAD)) for name in TRUTH} for _ in range(PROJECTS)]
    defects: dict[int, dict[str, int]] = {}
    next_id = 0
    trace.setdefault("detected", {})
    x = g.x()
    for n in range(1, reviews + 1):
        repo = rng.randrange(PROJECTS)
        present = defects.setdefault(repo, {})
        for name in list(present):
            if rng.random() < b.fix_rate:
                del present[name]
        for name in TRUTH:
            if name not in present and rng.random() < min(0.9, truth(name, n, b) * lean[repo][name]):
                next_id += 1
                present[name] = next_id

        def key(name: str, defect: int, repo: int = repo, n: int = n) -> str:
            return f"{repo}:{name}:{defect}" if b.dedup else f"{n}:{name}:{defect}"

        def verdict() -> float:
            return 1.0 if rng.random() < b.reproduce_real else CHECK["plausible"]

        part = 1.0
        if rng.random() < b.diff_share:
            part = b.diff_seen
            present = {k: v for k, v in present.items() if rng.random() < part}  # defects outside the diff stay
        walk = g.plan(rng)
        trace["walks"] = trace.get("walks", 0) + len(walk)
        walks: list[dict[str, Any]] = []
        empties: list[tuple[int, int | None]] = []
        for name in walk:
            d = g.require(name)
            effort = 1.0
            if not b.blind and b.effort_follows_odds:
                effort = 0.3 + 0.7 * min(1.0, d.radius(x) / 0.2)
            own = next((present[t] for t in (d.name, *d.aliases) if t in present), None)
            snapped = next(
                (
                    present[t]
                    for t, near in NEAREST.items()
                    if near == d.name and t in present and g.find(t) is None and rng.random() < b.snap_rate
                ),
                None,
            )
            real = own if own is not None else snapped
            if real is not None and rng.random() < (1 - b.miss_rate) * effort:
                walks.append({"node": name, "y": verdict(), "sev": "major", "key": key(name, real)})
            elif real is None and rng.random() < b.false_rate:
                y = 0.0 if rng.random() < b.refute_false else CHECK["plausible"]
                walks.append({"node": name, "y": y, **({"sev": "major", "key": f"false:{n}:{name}"} if y else {})})
            else:
                walks.append({"node": name, "y": 0.0})
                empties.append((len(walks) - 1, real))
        audit = None
        if empties and rng.random() < b.audit_share:
            trace["audits"] = trace.get("audits", 0) + 1
            i, real = empties[rng.randrange(len(empties))]
            found = real is not None and rng.random() < b.audit_catch
            audit = {"node": walks[i]["node"], "found": found}
            if found and real is not None:
                walks[i].update(y=1.0, sev="major", key=key(str(walks[i]["node"]), real))

        fanout: list[dict[str, Any]] = []
        births = 0
        recorded = {r["node"] for r in walks}
        for name, defect in present.items():
            known = g.find(name)
            if name in recorded or (known is not None and known.name in walk):
                continue
            if rng.random() >= b.notice:
                continue
            said = name
            if name in NEAREST and g.find(name) is None and rng.random() < b.snap_rate:
                said = NEAREST[name]  # filed under its broad domain instead of named
            elif name in SYNONYMS and rng.random() < b.synonym_rate:
                said = SYNONYMS[name]
            rec: dict[str, Any] = {"node": said, "y": verdict(), "sev": "major"}
            if g.find(said) is None:
                canonical = g.find(name)
                if said != name and canonical is not None and rng.random() < b.same_as_rate:
                    rec.update(node=canonical.name, alias=said)
                else:
                    if births >= min(REVIEW["births"], g.room()):
                        continue
                    births += 1
                    if b.parents == "honest" and name in NEAREST:
                        rec["parents"] = [NEAREST[name]]
                    elif b.parents == "hostile":
                        others = [s for s in seeds if s != NEAREST.get(name)]
                        rec["parents"] = [rng.choice(others)]
            found_node = g.find(rec["node"])
            node = found_node.name if found_node is not None else rec["node"]
            if node in recorded:
                continue
            recorded.add(node)
            rec["key"] = key(node, defect)
            fanout.append(rec)
        if rng.random() < b.invent and rng.random() >= b.refute_false:  # a refuted invention teaches nothing
            name = rng.choice(list(JUNK))
            alive = g.find(name)
            node = alive.name if alive else name
            if node not in recorded and (alive or births < min(REVIEW["births"], g.room())):
                rec = {"node": node, "y": CHECK["plausible"], "sev": "major", "key": f"junk:{n}:{name}"}
                if not alive and b.parents != "none":
                    rec["parents"] = [JUNK[name]]
                recorded.add(node)
                fanout.append(rec)

        if part < 1 and b.diff_empties == "drop":
            walks = [r for r in walks if r.get("y", 0.0) > 0]
        ev = {"op": "review", "n": n, "walks": walks, "fanout": fanout}
        if part < 1 and b.diff_empties == "scaled":
            ev["part"] = part
        merges = [
            [other, canonical]
            for canonical, other in SYNONYMS.items()
            if g.find(other)
            and g.find(canonical)
            and g.find(other) is not g.find(canonical)
            and rng.random() < b.merge_rate
        ]
        if merges:
            ev["merge"] = merges
        if audit:
            ev["audit"] = audit
        g.step(ev)
        if b.shift:
            shifted = g.find(b.shift[0])
            if n >= b.shift[3] and shifted and shifted.radius(x) >= REVIEW["dark"]:
                trace["detected"].setdefault("shift", n - b.shift[3])
    return g


# --- scoring -------------------------------------------------------------------


def score(seeds: range = range(1, 101), reviews: int = 300, b: Behavior | None = None) -> dict[str, float]:
    """Error counts over independent worlds; a cell is one direction.

    missed        a direction with real yield (>= 0.15) is dark
    phantom       an invented name that never pays is alive and lit
    unborn        a discovery never got a node of its own
    misplaced     a discovery born under a parent other than its true nearest one
    duplicates    a synonym pair ended up as two live nodes
    calibration   mean |radius - true rate x major severity| over every live cell
    walks, audits per review: what the plan costs the reviewer and the refuter
    """
    b = b or Behavior()
    out = {"missed": 0, "phantom": 0, "unborn": 0, "misplaced": 0, "duplicates": 0, "calibration": 0.0}
    cells = 0
    trace: dict = {}
    for s in seeds:
        g = run(reviews, s, b, trace)
        x = g.x()
        for canonical, other in SYNONYMS.items():
            a, o = g.find(canonical), g.find(other)
            out["duplicates"] += a is not None and o is not None and a is not o
        for name in JUNK:
            d = g.find(name)
            out["phantom"] += d is not None and d.radius(x) >= REVIEW["dark"]
        for name in TRUTH:
            d = g.find(name)
            if d is None or (name in DISCOVERIES and d.name != name):
                out["unborn"] += name in DISCOVERIES
                continue
            if name in DISCOVERIES and b.parents == "honest" and d.parent != NEAREST[name]:
                out["misplaced"] += 1
            r, y = d.radius(x), truth(name, reviews, b)
            out["missed"] += y >= 0.15 and r < REVIEW["dark"]
            out["calibration"] += abs(r - y * SEV["major"])
            cells += 1
    out["calibration"] = round(out["calibration"] / max(cells, 1), 4)
    out["walks"] = round(trace.get("walks", 0) / (reviews * len(seeds)), 2)
    out["audits"] = round(trace.get("audits", 0) / (reviews * len(seeds)), 3)
    return out


def table(g: Graph) -> str:
    x = g.x()
    rows = sorted(g.live(), key=lambda d: -d.radius(x))
    out = [f"after {g.n} reviews (dark < {REVIEW['dark']})", f"{'direction':<24}{'radius':>8}{'true':>8}   yield/walks"]
    for d in rows:
        true = f"{TRUTH.get(d.name, 0.0) * SEV['major']:>8.2f}"
        born = (
            f"  (born under {' + '.join(d.parents)})"
            if d.origin == "birth" and d.parents
            else "  (born)"
            if d.origin == "birth"
            else ""
        )
        out.append(f"{d.name:<24}{d.radius(x):>8.2f}{true}   {d.hits:.1f}/{d.walks}{born}")
    h = g.report()
    out.append(
        f"\nlog-loss {h['loss']:.3f} vs baseline {h['baseline']:.3f} · false-empty {h['false_empty_rate'] or 0:.2f}"
        f" over {h['audits']} audits · live {len(g.live())}"
    )
    return "\n".join(out)


if __name__ == "__main__":
    print(table(run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)))
    print("\nerrors over 100 worlds:", score(range(1, 101)))
