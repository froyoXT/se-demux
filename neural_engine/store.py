"""Where the graph lives on disk.

    pack/                                        the field's knowledge: the seed
    $NEURAL_ENGINE_HOME/<pack>/log.jsonl         one validated event per line: the only truth
    $NEURAL_ENGINE_HOME/<pack>/state.json        the replayed graph, for humans; never read back
    $NEURAL_ENGINE_HOME/<pack>/pending.json      the plan awaiting its review
    $NEURAL_ENGINE_HOME/<pack>/salt              per-store secret for finding keys

The root is global (default ~/.neural-engine) so experience crosses projects, and each
pack has its own home under that root. The log records the pack fingerprint each review
was learned under; if the pack changes, planning and applying stop until `repack`
acknowledges it, because replay would otherwise re-learn old reviews under a seed they
never saw.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import pack as packs
from .graph import Graph, replay
from .schema import LOG_VERSION

_SALT = re.compile(r"^[0-9a-f]{32}$")


class Store:
    def __init__(self, home: Path | None = None, pack: packs.Pack | None = None):
        root = home or Path(os.environ.get("NEURAL_ENGINE_HOME", Path.home() / ".neural-engine"))
        self.pack = pack or packs.load()
        self.home = root / self.pack.name
        self.home.mkdir(parents=True, exist_ok=True)
        self.log = self.home / "log.jsonl"
        self.state = self.home / "state.json"
        self.pending = self.home / "pending.json"

    def events(self) -> list[dict]:
        if not self.log.exists():
            return []
        text = self.log.read_text(encoding="utf-8")
        torn = bool(text) and not text.endswith("\n")
        lines = text.splitlines()
        out = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if torn and i == len(lines):
                    continue
                raise ValueError(f"log line {i} is not JSON") from exc
        return out

    def graph(self) -> Graph:
        return replay(self.pack, self.events())

    def learned_under(self) -> str | None:
        """The pack fingerprint the log was last learned or acknowledged under."""
        last = None
        for ev in self.events():
            if ev.get("op") == "pack":
                last = ev["fingerprint"]
            elif ev.get("op") == "review" and ev.get("pack"):
                last = ev["pack"]
        return last

    def require_current_pack(self) -> None:
        last = self.learned_under()
        if last and last != self.pack.fingerprint:
            raise ValueError(
                f"the pack changed ({last} -> {self.pack.fingerprint}) since the log was learned; "
                "replay would re-learn old reviews under a seed they never saw. "
                "Run `neural-engine repack` to accept the change."
            )

    def repack(self) -> list[str]:
        with self.locked():
            last = self.learned_under()
            if not last or last == self.pack.fingerprint:
                return [f"pack {self.pack.fingerprint} is already current"]
            event = {"op": "pack", "v": LOG_VERSION, "fingerprint": self.pack.fingerprint, "previous": last}
            g = self.graph()
            notes = g.step(event)
            self.commit(event, g)
        return notes

    @contextmanager
    def locked(self):
        with open(self.home / ".lock", "a+b") as fh:
            _lock(fh)
            yield

    def commit(self, event: dict, g: Graph) -> None:
        _cut_torn_tail(self.log)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _atomic(self.state, snapshot(g))
        if event.get("op") == "review":
            self.pending.unlink(missing_ok=True)

    def salt(self) -> str:
        """A per-store secret, so a stored line hash cannot be reversed by guessing short lines."""
        path = self.home / "salt"
        text = _salt_text(path)
        if _SALT.fullmatch(text):
            return text
        if self.events():
            raise ValueError(
                "the salt file is missing or invalid but the log already has events; restore the salt file"
            )
        return _write_salt(path)

    def save_pending(self, pending: dict) -> None:
        _atomic(self.pending, json.dumps(pending, indent=1) + "\n")

    def load_pending(self) -> dict:
        if not self.pending.exists():
            raise ValueError("no pending plan; run plan first")
        return json.loads(self.pending.read_text(encoding="utf-8"))

    def check(self) -> list[str]:
        fails = []
        try:
            g = self.graph()
        except (OSError, ValueError, KeyError) as exc:
            return [f"replay failed: {exc}"]
        if self.state.exists() and self.state.read_text(encoding="utf-8") != snapshot(g):
            fails.append("state.json is not the pack replayed through log.jsonl")
        last = self.learned_under()
        if last and last != self.pack.fingerprint:
            fails.append(f"pack changed ({last} -> {self.pack.fingerprint}) and is not acknowledged; run repack")
        return fails


def snapshot(g: Graph) -> str:
    rows = {}
    for d in sorted(g.directions.values(), key=lambda d: d.name):
        rows[d.name] = {
            "origin": d.origin,
            "born": d.born,
            "parents": d.parents,
            "deleted": d.deleted,
            "walks": d.walks,
            "hits": round(d.hits, 3),
            "value": round(d.value(), 3),
            "aliases": d.aliases,
            "weights": {k: [round(d.m[k], 3), round(d.q[k], 3)] for k in sorted(d.m)},
        }
    state = {"reviews": g.n, "health": g.report(), "directions": rows}
    return json.dumps(state, indent=1) + "\n"


if sys.platform == "win32":
    import errno
    import msvcrt

    def _lock(fh) -> None:
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)  # waits ~10 s, then raises EDEADLOCK if still held
                return
            except OSError as exc:
                if exc.errno != errno.EDEADLOCK:
                    raise

else:
    import fcntl

    def _lock(fh) -> None:
        fcntl.flock(fh, fcntl.LOCK_EX)


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _salt_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return ""


def _write_salt(path: Path) -> str:
    """Publish one salt through a private temp file and return the bytes now in `path`.

    A shared ``salt.tmp`` let two first callers race: one's replace moved the file and the
    other raised FileNotFoundError. Each caller writes its own temp (mode 0o600, fsynced).
    A link publishes it only when the salt is absent, so racers keep the first one and the
    re-read returns that value; replace would let a later caller discard it. An invalid salt
    is replaced.
    """
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        try:
            os.chmod(name, 0o600)
            os.write(fd, os.urandom(16).hex().encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(name, path)
        except FileExistsError:
            if not _SALT.fullmatch(_salt_text(path)):
                os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return _salt_text(path)


def _cut_torn_tail(path: Path) -> None:
    """A crashed append leaves a partial last line. Keep a tail that parses as JSON; drop one that does not."""
    if not path.exists():
        return
    with open(path, "rb+") as fh:
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        if end == 0:
            return
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
        nl = data.rfind(b"\n")
        tail = data[nl + 1 :]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            fh.truncate(0 if nl < 0 else nl + 1)
        else:
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
        fh.flush()
        os.fsync(fh.fileno())
