"""Refuse a payload before it reaches the graph.

The reviewed code is untrusted, and the graph steers every later review. What
the graph keeps from a payload is only: direction names, a parent chosen from
existing names, the refuter's verdict, a severity, and a salted hash of each
finding's line. Sites, quotes and claims are checked here and then dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from .config import CHECK, FINDING, REVIEW, SEV
from .graph import Graph
from .schema import SITE_RE, Payload, errors, name_error, normalize_site

SITE = re.compile(SITE_RE)
WALKS_PER_LINE = 3
CLAIM = (20, 600)


def normalize(raw: object) -> dict:
    """Structure first: a payload that does not match schema.Payload is refused before any meaning is read."""
    try:
        payload = Payload.model_validate(raw).model_dump(by_alias=True, exclude_none=True)
    except ValidationError as exc:
        raise ValueError("\n".join(errors(exc))) from exc
    for rec in (*payload.get("walks", ()), *payload.get("fanout", ())):
        rec["site"] = normalize_site(rec["site"])
    if (audit := payload.get("audit")) and audit.get("site"):
        audit["site"] = normalize_site(audit["site"])
    return payload


def audit_target(pending: dict, walks: list[dict]) -> str | None:
    """The direction fixed at plan time, audited only when that walk came back empty."""
    node = pending.get("audit_node")
    if not node:
        return None
    walk = next((w for w in walks if w.get("node") == node), None)
    if walk is None or walk.get("verdict") != "empty":
        return None
    return str(node)


def payload_errors(g: Graph, pending: dict, raw: dict, verified: bool = True) -> list[str]:
    """`verified=False` checks a payload before the refuter has run (for `neural-engine verify`)."""
    try:
        raw = normalize(raw)
    except ValueError as exc:
        return str(exc).splitlines()
    if raw.get("plan") != pending["id"]:
        return [f"plan {raw.get('plan')} is not the pending plan {pending['id']}; run plan again"]
    if pending["n"] != g.n:
        return ["the graph moved since this plan; run plan again"]
    root = Path(pending["root"])
    walks, fanout = raw.get("walks") or [], raw.get("fanout") or []

    names = [str(w.get("node")) for w in walks]
    if sorted(names) != sorted(pending["walk"]):
        missing = sorted(set(pending["walk"]) - set(names))
        unplanned = sorted(set(names) - set(pending["walk"]))
        return [f"walks must be exactly the plan; missing {missing}, unplanned {unplanned}"]

    errs: list[str] = []
    hit_sites: set[str] = set()
    per_line: dict[str, int] = {}
    for w in walks:
        label, verdict = w["node"], w.get("verdict")
        if verdict not in ("hit", "empty"):
            errs.append(f"{label}: verdict is hit or empty")
            continue
        errs += anchor_errors(label, w, root)
        site = w.get("site")
        if verdict == "hit":
            errs += finding_errors(label, w, verified, SEV)
            if site in hit_sites:
                errs.append(f"{label}: {site} is already another finding's; one root cause, one record")
            hit_sites.add(site)
        else:
            errs += empty_errors(label, w, pending)
        per_line[site] = per_line.get(site, 0) + 1
        if per_line[site] == WALKS_PER_LINE + 1:
            errs.append(f"{site} anchors more than {WALKS_PER_LINE} walks; anchor each where you looked")

    if len(fanout) > REVIEW["fanout"]:
        errs.append(f"at most {REVIEW['fanout']} fan-out records")
    new: set[str] = set()
    for f in fanout:
        name = f.get("node")
        if err := name_error(name):
            errs.append(f"fan-out {err}")
            continue
        label = f"fan-out {name}"
        errs += anchor_errors(label, f, root) + finding_errors(label, f, verified, FINDING)
        if f.get("site") in hit_sites:
            errs.append(f"{label}: {f.get('site')} is already another finding's; one root cause, one record")
        hit_sites.add(f.get("site"))
        parents = f.get("parents", f.get("parent"))
        parents = [parents] if isinstance(parents, str) else parents
        same = f.get("same_as")
        if same is not None:
            target = g.find(str(same))
            if g.find(name) is not None:
                errs.append(f"{label}: same_as is only for a new name")
            elif target is None:
                errs.append(f"{label}: same_as {same!r} is not an existing direction")
            elif twin := near_twin(g, name, target.name):
                errs.append(f"{label} is {twin}; name it {twin}")
            continue
        if g.find(name) is not None:
            if parents:
                errs.append(f"{label}: parents are only for a new name")
            continue  # a known direction, recorded again or for the first time: merged at apply
        new.add(name)
        if twin := near_twin(g, name):
            errs.append(f"{label} is {twin}; name it {twin}")
        if parents is not None:
            if not isinstance(parents, list) or not 1 <= len(parents) <= REVIEW["parents"]:
                errs.append(f"{label}: parents is a list of 1-{REVIEW['parents']} existing directions")
            else:
                errs += [
                    f"{label}: parent {p!r} is not an existing direction" for p in parents if g.find(str(p)) is None
                ]
    if len(new) > REVIEW["births"]:
        errs.append(f"{len(new)} new names; at most {REVIEW['births']} per review, file the rest under existing names")
    if len(new) > g.room():
        errs.append(f"{len(new)} new directions but the node budget has room for {g.room()}")

    for pair in raw.get("merge") or []:
        a, b = (g.find(n) for n in pair)
        pair = list(pair)
        if a is None or b is None or a is b:
            errs.append(f"merge {pair}: both must be distinct existing directions")
        elif a.origin != "birth" or g.n - a.born >= 10:
            errs.append(f"merge {pair}: only a recently born direction can be merged away")
        elif _descends(b, a, g):
            errs.append(f"merge {pair}: {b.name} descends from {a.name}")
    if verified:
        errs += audit_errors(pending, walks, raw.get("audit"), root)
    return errs


def empty_errors(label: str, w: dict, pending: dict) -> list[str]:
    """An empty walk says where the reviewer looked, so it must be a place a defect could have been seen."""
    errs = []
    site = str(w.get("site"))
    if w.get("sev") is not None:
        errs.append(f"{label}: empty carries no sev")
    changed = pending.get("changed")
    if changed and site.rpartition(":")[0] not in changed:
        errs.append(f"{label}: this review covers only the changed files; cite where you looked in them")
    return errs


def finding_errors(label: str, rec: dict, verified: bool, sevs) -> list[str]:
    errs = []
    if rec.get("sev") not in sevs:
        errs.append(f"{label}: sev must be one of {'/'.join(sevs)}")
    claim = rec.get("claim")
    if not isinstance(claim, str) or not CLAIM[0] <= len(claim.strip()) <= CLAIM[1]:
        errs.append(f"{label}: claim of {CLAIM[0]}-{CLAIM[1]} characters: the input or state, and the wrong outcome")
    if verified and rec.get("check") not in CHECK:
        errs.append(f"{label}: check must be the refuter's verdict, one of {'/'.join(CHECK)}")
    return errs


def audit_errors(pending: dict, walks: list[dict], audit: object, root: Path) -> list[str]:
    target = audit_target(pending, walks)
    if target is None:
        return [] if not audit else ["no audit on this review; drop the audit block"]
    if not isinstance(audit, dict) or audit.get("node") != target:
        return [f"audit must re-examine {target} (from `neural-engine verify`)"]
    if audit.get("found") is True:
        return anchor_errors("audit", audit, root) + finding_errors("audit", audit, False, FINDING)
    if audit.get("found") is not False:
        return ["audit.found is true or false"]
    return []


def anchor_errors(label: str, rec: dict, root: Path) -> list[str]:
    site, quote = rec.get("site"), rec.get("quote")
    if not isinstance(site, str) or not SITE.fullmatch(site):
        return [f"{label}: site must be relpath:line"]
    rel, _, line = site.rpartition(":")
    path = (root / rel).resolve()
    if ".." in Path(rel).parts or not path.is_relative_to(root) or not path.is_file():
        return [f"{label}: {rel} is not a file under the reviewed root"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(line)
    if n > len(lines):
        return [f"{label}: {site} is past the end of the file"]
    if not isinstance(quote, str) or not 3 <= len(quote.strip()) <= 120:
        return [f"{label}: quote 3-120 characters from the cited line"]
    if quote.strip() not in lines[n - 1]:
        near = [i for i in range(max(1, n - 3), min(len(lines), n + 3) + 1) if quote.strip() in lines[i - 1]]
        hint = f"; it is on {', '.join(f'{rel}:{i}' for i in near)}" if near else ""
        return [f"{label}: quote is not at {site}{hint}"]
    return []


def excerpt(root: Path, site: str, around: int) -> list[str]:
    """Numbered lines either side of a site (already checked by anchor_errors), each cut to a readable width."""
    rel, _, line = site.rpartition(":")
    lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(line)
    lo, hi = max(1, n - around), min(len(lines), n + around)
    return [f"{i:>5}{'>' if i == n else ' '} {lines[i - 1][:160]}" for i in range(lo, hi + 1)]


def site_text(root: Path, site: str) -> str:
    """Whitespace-collapsed text of the cited line and up to two lines either side."""
    rel, _, line = site.rpartition(":")
    lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(line)
    return " ".join(" ".join(lines[max(0, n - 3) : n + 2]).split())


def near_twin(g: Graph, name: str, skip: str | None = None) -> str | None:
    """A new name too close to a live direction or an alias, other than `skip`."""
    mine = _norm(name)
    pool = [(d.name, other) for d in g.live() for other in (d.name, *d.aliases)]
    for owner, other in pool:
        if owner == skip:
            continue
        theirs = _norm(other)
        if mine == theirs or (min(len(mine), len(theirs)) > 5 and _lev(mine, theirs) <= 1):
            return owner
    return None


def _descends(child, ancestor, g: Graph) -> bool:
    """True when `child` reaches `ancestor` by following parent links."""
    seen: set[str] = set()
    stack = list(child.parents)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name == ancestor.name:
            return True
        parent = g.directions.get(name) or g.find(name)
        if parent is not None:
            stack.extend(parent.parents)
    return False


def _norm(s: str) -> str:
    return "".join(t[:-1] if t.endswith("s") and len(t) > 3 else t for t in s.split("-"))


def _lev(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]
