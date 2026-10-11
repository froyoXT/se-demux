"""The graph of directions: planning, learning from reviews and feedback, birth, deletion, replay.

There are no levels. Every direction is a peer node; a born node remembers the
parents it inherited from (at most two), which is all the structure there is.
Depth grows wherever reviews keep finding something under a node.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import LEARN, PRIOR, REVIEW, SEV
from .model import BIAS, Direction, born, features, sigma
from .pack import Pack
from .schema import EVENT


@dataclass
class Health:
    """Per-review numbers, all computed from the log at no extra cost."""

    n: int
    loss: float = 0.0  # prequential log-loss of the graph's p on planned walks
    base: float = 0.0  # ...and of a bias-only baseline (one global rate)
    walks: int = 0
    misses: int = 0  # fan-out findings on directions that were dark here
    audit: int | None = None  # 1 if the audited empty walk had a defect, 0 if not


@dataclass
class Graph:
    directions: dict[str, Direction] = field(default_factory=dict)
    n: int = 0
    findings: dict[str, dict] = field(default_factory=dict)  # key -> {node, sev, feedback}
    health: list[Health] = field(default_factory=list)
    base_hits: float = 0.0
    base_walks: int = 0
    seen: dict[str, int] = field(default_factory=dict)  # salted hash of reviewed bytes and pack -> the review
    recipes: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)

    def x(self) -> dict[str, float]:
        return features()

    def live(self) -> list[Direction]:
        return [d for d in self.directions.values() if not d.deleted]

    def require(self, name: str) -> Direction:
        """A live direction by name or alias; an error if there is none (callers have validated the name)."""
        d = self.find(name)
        if d is None:
            raise ValueError(f"no live direction named {name!r}")
        return d

    def find(self, name: str) -> Direction | None:
        for d in self.live():
            if name == d.name or name in d.aliases:
                return d
        return None

    def plan(self, rng: random.Random) -> list[str]:
        """Thompson draw per direction; walk the best draws that clear the dark line.

        An uncertain direction sometimes draws high and is walked, so a dark one is re-tested
        in proportion to how unsure the graph is about it; fan-out catches what draws miss.
        """
        x = self.x()
        draws = sorted(((d.sample(x, rng), d.name) for d in self.live()), reverse=True)
        return [name for draw, name in draws[: int(REVIEW["walk"])] if draw >= REVIEW["dark"]]

    def room(self) -> int:
        """Births this graph can take right now: free budget plus evictable newborns."""
        return max(0, int(REVIEW["max_live"]) - len(self.live())) + len(self._evictable())

    # --- one review --------------------------------------------------------

    def step(self, ev: dict) -> list[str]:
        if ev.get("op") == "feedback":
            return self._feedback(ev)
        if ev.get("op") == "pack":
            return [f"pack {ev['previous']} -> {ev['fingerprint']} acknowledged"]
        if ev.get("n") != self.n + 1:
            raise ValueError(f"review {ev.get('n')} is not {self.n + 1}")
        n, notes = self.n + 1, list[str]()
        x = self.x()
        part = ev.get("part", 1.0)
        for d in self.live():
            d.forget(part)
        h = Health(n, audit=None if "audit" not in ev else int(ev["audit"]["found"]))

        for rec in ev["walks"]:
            d = self.find(rec["node"]) or self._birth(rec["node"], [], n, notes)
            self._alias(d, rec, notes)
            y = rec.get("y", 0.0)
            # A defect seen in a diff is a whole observation; nothing seen in a diff speaks only for the part examined.
            seen = 1.0 if y > 0 else part
            w = (LEARN["repeat_weight"] if self._repeat(rec, notes) else 1.0) * seen
            if w == 0:
                continue
            # health scores every walk: p predicts "this walk finds a defect", and a re-found one is found
            p, base = min(max(d.p(x), 1e-6), 1 - 1e-6), (self.base_hits + 1) / (self.base_walks + 2)
            h.loss += _xent(y, p)
            h.base += _xent(y, base)
            h.walks += 1
            self.base_hits += y
            self.base_walks += 1
            before = d.radius(x)
            d.learn(x, y, SEV.get(rec.get("sev"), 0.0), w=w)
            d.walks += 1
            d.looks += seen
            self._remember(rec, d)
            notes.append(_line(d, rec, y, before, d.radius(x)))

        for rec in ev.get("fanout", []):
            existing = self.find(rec["node"])
            if existing is not None and existing.radius(x) < REVIEW["dark"]:
                h.misses += 1
            d = existing or self._birth(rec["node"], _parents(rec), n, notes)
            self._alias(d, rec, notes)
            w = LEARN["fanout_weight"] * (LEARN["repeat_weight"] if self._repeat(rec, notes) else 1.0)
            if w == 0:
                continue
            before = d.radius(x)
            d.learn(x, rec["y"], SEV[rec["sev"]], w=w)
            self._remember(rec, d)
            notes.append(_line(d, rec, rec["y"], before, d.radius(x), "fan-out"))

        for drop, keep in ev.get("merge", []):
            a, b = self.find(drop), self.find(keep)
            if a is not None and b is not None and a is not b:
                b.absorb(a)
                for d in self.directions.values():  # its children now hang under the survivor
                    moved = (b.name if p == a.name else p for p in d.parents)
                    d.parents = list(dict.fromkeys(p for p in moved if p != d.name))
                notes.append(f"merge {a.name} into {b.name}")

        for d in self.live():
            if self._stillborn(d, x):
                d.deleted = True
                notes.append(f"delete {d.name} (dark even optimistically, after {d.looks:g} walks)")

        if ev.get("tree"):
            self.seen[ev["tree"]] = n
        self.health.append(h)
        self.n = n
        return notes

    def unchanged(self, tree: str) -> int | None:
        """The review that already saw these exact bytes under this pack (both are in the hash): nothing new to see."""
        return self.seen.get(tree)

    def _birth(self, name: str, parents: list[str], n: int, notes: list[str]) -> Direction:
        """Born under the parents the reviewer chose from existing names."""
        ups = [d for d in (self.find(p) for p in parents[: int(REVIEW["parents"])]) if d is not None]
        if len(self.live()) >= REVIEW["max_live"]:
            victims = self._evictable()
            if not victims:
                raise ValueError("node budget is full and no dark newborn can be replaced")
            victim = min(victims, key=lambda d: (sum(d.q.values()) + sum(d.prior_q.values()), d.name))
            victim.deleted = True
            notes.append(f"replace {victim.name} (least certain dark newborn; node budget)")
        d = born(name, ups, n)
        self.directions[name] = d
        notes.append(f"birth {name}" + (f" under {' + '.join(u.name for u in ups)}" if ups else ""))
        return d

    @staticmethod
    def _stillborn(d: Direction, x: dict[str, float]) -> bool:
        """A newborn that, while still young, stays dark even at `delete_z` standard deviations of optimism.

        Deletion is a judgment about infancy only. A direction that survives it behaves like a seed:
        it may go dark, never deleted, so a noisy run of empties cannot kill a proven direction.
        """
        if d.origin != "birth" or not LEARN["delete_walks"] <= d.looks <= 3 * LEARN["delete_walks"]:
            return False
        mu, var = d.logit(x)
        return sigma(mu + LEARN["delete_z"] * math.sqrt(var)) * d.value() < REVIEW["dark"]

    def _evictable(self) -> list[Direction]:
        return [d for d in self.live() if d.origin == "birth" and d.radius(self.x()) < REVIEW["dark"]]

    @staticmethod
    def _alias(d: Direction, rec: dict, notes: list[str]) -> None:
        if rec.get("alias") and rec["alias"] not in d.aliases:
            d.aliases.append(rec["alias"])
            notes.append(f"alias {rec['alias']} -> {d.name}")

    def _repeat(self, rec: dict, notes: list[str]) -> bool:
        """A finding already learned from (same project, direction, line) is evidence, but not new evidence.

        It is learned at `repeat_weight`, so one unresolved finding reviewed daily cannot own an edge,
        while the fact that it is still there still counts for something.
        """
        key = rec.get("key")
        if key and key in self.findings:
            notes.append(f"{rec['node']:<24} repeat of {key}, learned at weight {LEARN['repeat_weight']}")
            return True
        return False

    def _remember(self, rec: dict, d: Direction) -> None:
        if key := rec.get("key"):
            prev = self.findings.get(key)
            # A later sighting of the same finding must not clear a verdict already given.
            self.findings[key] = {
                "node": d.name,
                "sev": rec.get("sev"),
                "feedback": prev["feedback"] if prev else None,
            }

    def _feedback(self, ev: dict) -> list[str]:
        f = self.findings.get(ev["key"])
        if f is None or f["feedback"] is not None:
            raise ValueError(f"finding {ev['key']} is unknown or already has feedback")
        d = self.find(f["node"])
        f["feedback"] = ev["verdict"]
        if d is None:
            return [f"feedback {ev['key']} {ev['verdict']}: {f['node']} no longer exists"]
        x = self.x()
        before = d.radius(x)
        if ev["verdict"] == "fixed":
            d.learn(x, 1.0, SEV.get(f["sev"], 0.0), w=LEARN["fixed_weight"])
        else:
            d.learn(x, 0.0, 0.0, w=LEARN["wrong_weight"])
        return [f"feedback {ev['key']} {ev['verdict']}: {d.name} r {before:.2f} -> {d.radius(x):.2f}"]

    # --- health ------------------------------------------------------------

    def report(self) -> dict[str, float | None]:
        recent = self.health[-int(LEARN["window"]) :]
        walks = sum(h.walks for h in recent)
        audits = [h.audit for h in recent if h.audit is not None]
        judged = [f for f in self.findings.values() if f["feedback"]]
        return {
            "loss": sum(h.loss for h in recent) / walks if walks else None,
            "baseline": sum(h.base for h in recent) / walks if walks else None,
            "misses_per_review": sum(h.misses for h in recent) / len(recent) if recent else None,
            "false_empty_rate": sum(audits) / len(audits) if audits else None,
            "audits": len(audits),
            "feedback_wrong_rate": sum(f["feedback"] == "wrong" for f in judged) / len(judged) if judged else None,
            "feedback": len(judged),
        }


def _parents(rec: dict) -> list[str]:
    p = rec.get("parents", rec.get("parent"))
    return [p] if isinstance(p, str) else list(p or [])


def _xent(y: float, p: float) -> float:
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _line(d: Direction, rec: dict, y: float, before: float, after: float, kind: str = "") -> str:
    tag = f"hit {rec.get('sev')} y={y:g}" if y > 0 else "empty"
    return f"{d.name:<24} {tag:<20} r {before:.2f} -> {after:.2f}" + (f"  ({kind})" if kind else "")


def seeded(pack: Pack) -> Graph:
    g = Graph()
    for s in pack.seeds:
        # A child is a seed the pack author asserted. It does not inherit belief: the specialisation
        # offset would start that inheritance near the dark line.
        g.directions[s.name] = Direction(
            s.name,
            "seed",
            parents=list(s.parents),
            prior={BIAS: PRIOR["seed_bias"]},
            value_prior=s.value,
        )
    g.recipes = {s.name: s.recipe for s in pack.seeds if s.recipe}
    return g


def replay(pack: Pack, events: list[dict]) -> Graph:
    """Rebuild the graph from the pack and the log. Every log line is validated before it is used."""
    g = seeded(pack)
    for i, raw in enumerate(events, 1):
        try:
            ev = EVENT.validate_python(raw).model_dump(by_alias=True, exclude_none=True)
        except ValueError as exc:
            raise ValueError(f"log line {i} is not a valid event: {exc}") from exc
        g.step(ev)
    return g
