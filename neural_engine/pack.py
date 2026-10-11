"""A pack: everything the engine knows about one field, loaded from a folder.

The engine contains no knowledge of any field. A pack is the seed it starts from:
directions, the parents a child narrows, and an optional recipe for that child.
The shipped pack in `pack/` is one field; another field is another folder.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import REVIEW
from .schema import name_error

DEFAULT = Path(__file__).resolve().parent.parent / "pack"
_LABELS = ("rule", "wrong", "right", "cite")
_TOKEN = re.compile(r"[0-9A-Za-z]+(?:-[0-9A-Za-z]+)*")


@dataclass(frozen=True)
class Seed:
    name: str
    value: float
    parents: tuple[str, ...] = ()
    recipe: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Pack:
    name: str
    version: str
    seeds: tuple[Seed, ...]
    fingerprint: str = ""
    sim: Path | None = None


def load(folder: Path = DEFAULT) -> Pack:
    pack, _names = _load(folder)
    return pack


def _load(folder: Path, stack: tuple[Path, ...] = ()) -> tuple[Pack, tuple[str, ...]]:
    folder = folder.resolve()
    if folder in stack:
        raise ValueError("extends cycle")
    meta = tomllib.loads((folder / "pack.toml").read_text(encoding="utf-8"))
    name = meta["pack"]["name"]
    if err := name_error(name):
        raise ValueError(f"pack.toml: {err}")
    version = str(meta["pack"]["version"])
    extends = meta["pack"].get("extends")
    base: Pack | None = None
    base_names: tuple[str, ...] = ()
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ValueError("pack.toml: extends must name a pack or a relative path")
        extends = extends.strip()
        base_folder = _extends_folder(folder, extends)
        if base_folder in (*stack, folder):
            raise ValueError(f"extends cycle ({extends})")
        base, base_names = _load(base_folder, (*stack, folder))
        if name in base_names:
            raise ValueError(f"pack name {name!r} is the same as a pack it extends")
    files = meta["files"]
    earlier = [s.name for s in base.seeds] if base else []
    by = {s.name: s for s in base.seeds} if base else {}
    order = list(earlier)
    for row_name, value, parents in _rows((folder / str(files["seed"])).read_text(encoding="utf-8"), earlier):
        if parents is None:
            prev = by[row_name]
            by[row_name] = Seed(row_name, value, prev.parents, prev.recipe)
        else:
            by[row_name] = Seed(row_name, value, parents)
            order.append(row_name)
    if not order:
        raise ValueError("seed.tsv: no directions")
    recipe_key = files.get("recipes")
    recipe_dir = folder / str(recipe_key) if recipe_key else None
    for seed_name, recipe in _recipes(recipe_dir, by, set(earlier)).items():
        prev = by[seed_name]
        by[seed_name] = Seed(prev.name, prev.value, prev.parents, recipe)
    seeds = tuple(by[n] for n in order)
    _exclusive({s.name: s.recipe for s in seeds if s.recipe})
    sim_key = files.get("sim")
    if sim_key:
        sim: Path | None = (folder / str(sim_key)).resolve()
    else:
        sim = base.sim if base else None
    return Pack(name, version, seeds, _fingerprint(seeds), sim), (*base_names, name)


def _extends_folder(folder: Path, extends: str) -> Path:
    if extends.startswith(".") or "/" in extends:
        if Path(extends).is_absolute():
            raise ValueError(f"extends {extends!r} must be relative to the pack")
        path = (folder / extends).resolve()
        if not (path / "pack.toml").is_file():
            raise ValueError(f"extends {extends!r} is not a pack folder")
        return path
    shipped = tomllib.loads((DEFAULT / "pack.toml").read_text(encoding="utf-8"))["pack"]["name"]
    if extends == shipped:
        return DEFAULT.resolve()
    raise ValueError(f"extends {extends!r} is not a known pack")


def _rows(text: str, earlier: list[str]) -> list[tuple[str, float, tuple[str, ...] | None]]:
    """Rows of this file. `parents is None` is an override of a base seed: value only."""
    defined = list(earlier)
    seen: set[str] = set()
    out: list[tuple[str, float, tuple[str, ...] | None]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = raw.split("\t")
        name = cols[0].strip()
        if err := name_error(name):
            raise ValueError(f"seed.tsv: {err}")
        if name in seen:
            raise ValueError(f"seed.tsv: duplicate name {name}")
        seen.add(name)
        value = _value(name, cols[1].strip() if len(cols) > 1 else "")
        parents_raw = cols[2].strip() if len(cols) > 2 else ""
        if name in earlier:
            if parents_raw:
                raise ValueError(f"seed.tsv: {name} overrides a base seed and cannot set parents")
            out.append((name, value, None))
            continue
        out.append((name, value, _parents(name, parents_raw, defined)))
        defined.append(name)
    return out


def _value(name: str, raw: str) -> float:
    if not raw:
        return 0.5
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"seed.tsv: {name} value must be in (0, 1]") from None
    if not 0 < value <= 1:
        raise ValueError(f"seed.tsv: {name} value must be in (0, 1]")
    return value


def _parents(name: str, raw: str, defined: list[str]) -> tuple[str, ...]:
    if not raw:
        return ()
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"seed.tsv: {name} has an empty parent")
    if len(parts) > int(REVIEW["parents"]):
        raise ValueError(f"seed.tsv: {name} has {len(parts)} parents; at most {int(REVIEW['parents'])}")
    if name in parts:
        raise ValueError(f"seed.tsv: {name} is its own parent")
    if len(set(parts)) != len(parts):
        raise ValueError(f"seed.tsv: {name} lists a parent more than once")
    known = set(defined)
    for part in parts:
        if part not in known:
            raise ValueError(f"seed.tsv: parent {part} of {name} is not defined earlier")
    return tuple(parts)


def _recipes(
    folder: Path | None, seeds: dict[str, Seed], base_names: set[str]
) -> dict[str, tuple[tuple[str, str], ...]]:
    if folder is None:
        return {}
    if not folder.is_dir():
        raise ValueError(f"{folder.name}: recipe folder is missing")
    found: dict[str, tuple[tuple[str, str], ...]] = {}
    names = set(seeds)
    # A root carries no recipe, so its name is an ordinary word, not a reference to one.
    children = {name for name, seed in seeds.items() if seed.parents}
    for entry in sorted(folder.iterdir(), key=lambda path: path.name):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            raise ValueError(f"{entry.name}: a recipe folder cannot contain a subfolder")
        if not entry.is_file() or not entry.name.endswith(".md"):
            raise ValueError(f"{entry.name}: a recipe file must be named <seed>.md")
        stem = entry.name[:-3]
        if stem not in names:
            raise ValueError(f"{entry.name}: {stem} is not a seed")
        if stem in base_names:
            raise ValueError(f"{entry.name}: a recipe for a base seed is not allowed")
        if not seeds[stem].parents:
            raise ValueError(f"{entry.name}: a root is a universal domain; it carries no recipe")
        found[stem] = _recipe(entry, children - {stem})
    return found


def _recipe(path: Path, others: set[str]) -> tuple[tuple[str, str], ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != len(_LABELS):
        raise ValueError(f"{path.name}: a recipe is exactly four lines (rule, wrong, right, cite)")
    parsed: list[tuple[str, str]] = []
    folded = {name.casefold() for name in others}
    for label, line in zip(_LABELS, lines, strict=True):
        got, sep, value = line.partition(":")
        if sep != ":" or got != label:
            raise ValueError(f"{path.name}: expected {label} next, found {got or 'no label'}")
        value = value.strip()
        if not value:
            raise ValueError(f"{path.name}: {label} is empty")
        if len(line) > int(REVIEW["recipe_line"]):
            raise ValueError(f"{path.name}: {label} is over {int(REVIEW['recipe_line'])} characters")
        if "://" in line:
            raise ValueError(f"{path.name}: a recipe cannot contain a link")
        if hit := _mentioned(line, folded):
            raise ValueError(f"{path.name}: mentions another child ({hit})")
        parsed.append((label, value))
    if parsed[1][1] == parsed[2][1]:
        raise ValueError(f"{path.name}: wrong and right must differ")
    return tuple(parsed)


def _mentioned(line: str, folded: set[str]) -> str | None:
    for token in _TOKEN.findall(line):
        if token.casefold() in folded:
            return token.casefold()
    return None


def _exclusive(recipes: dict[str, tuple[tuple[str, str], ...]]) -> None:
    names = list(recipes)
    for i, left in enumerate(names):
        lv = dict(recipes[left])
        for right in names[i + 1 :]:
            rv = dict(recipes[right])
            for field in ("rule", "wrong", "right"):
                if _norm(lv[field]) == _norm(rv[field]):
                    raise ValueError(f"{left}.md and {right}.md share a {field}")
            if _jaccard(_words(lv["rule"]), _words(rv["rule"])) >= float(REVIEW["recipe_overlap"]):
                raise ValueError(f"{left}.md and {right}.md rules overlap")


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _fingerprint(seeds: tuple[Seed, ...]) -> str:
    """Resolved seeds, not the file bytes: a comment or a blank line changes nothing."""
    payload = [[s.name, s.value, list(s.parents), [list(pair) for pair in s.recipe]] for s in seeds]
    return hashlib.sha1(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:12]
