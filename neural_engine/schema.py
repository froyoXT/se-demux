"""Typed boundaries: what an agent may hand the engine, and what the log may hold.

Structure is checked here by pydantic; meaning (does this direction exist, is the
quote really at that line, was this walk planned) is checked against the graph in
guard.py. `neural-engine schema payload` prints the JSON Schema so an agent can follow it.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

LOG_VERSION = 1
NAME_RE = r"^[a-z][a-z0-9]*(-[a-z0-9]+){0,3}$"
# The path is any non-control characters (the line is the part after the last colon); traversal is checked later.
SITE_RE = r"^[^\x00-\x1f\x7f-\x9f]{1,240}:[1-9][0-9]{0,6}$"
DENY = {
    "ignore",
    "skip",
    "safe",
    "trusted",
    "none",
    "instruction",
    "instructions",
    "allow",
    "disable",
    "prompt",
    "approve",
    "approved",
    "exempt",
    "override",
    "system",
    "assistant",
    "user",
    "clean",
    "pass",
}


def normalize_site(site: str) -> str:
    """One spelling of a site path: no leading ./, no redundant separators, posix."""
    rel, sep, line = site.rpartition(":")
    if not sep:
        return site
    return f"{PurePosixPath(rel).as_posix()}:{line}"


def _claim(value: str) -> str:
    if any(unicodedata.category(ch) == "Cc" or ch in "\u2028\u2029" for ch in value):
        raise ValueError("claim must not contain newlines or control characters")
    return value


Name = Annotated[str, StringConstraints(pattern=NAME_RE, max_length=32)]
Site = Annotated[str, StringConstraints(pattern=SITE_RE)]
Quote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=120)]
Claim = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=20, max_length=600),
    AfterValidator(_claim),
]
Sev = Literal["critical", "major", "minor", "info"]
Finding = Literal["critical", "major", "minor"]
Check = Literal["reproduced", "plausible", "refuted"]


def name_error(name: object) -> str | None:
    """Why a name may not enter the graph, or None."""
    if not isinstance(name, str) or not re.fullmatch(NAME_RE, name) or len(name) > 32:
        return f"{name!r} must be 1-4 lowercase words joined by hyphens"
    if DENY & set(name.split("-")):
        return f"{name!r} uses a reserved word"
    return None


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- what the agent hands in --------------------------------------------------


class Walk(Strict):
    node: Name
    verdict: Literal["hit", "empty"]
    site: Site
    quote: Quote
    sev: Sev | None = None
    claim: Claim | None = None
    check: Check | None = None


class Fanout(Strict):
    node: Name
    site: Site
    quote: Quote
    sev: Finding
    claim: Claim
    check: Check | None = None
    parents: list[Name] | None = Field(None, min_length=1, max_length=2)
    same_as: Name | None = None


class Audit(Strict):
    node: Name
    found: bool
    site: Site | None = None
    quote: Quote | None = None
    sev: Finding | None = None
    claim: Claim | None = None


class Payload(Strict):
    plan: str
    walks: list[Walk]
    fanout: list[Fanout] = Field(default_factory=list, max_length=12)
    audit: Audit | None = None
    merge: list[tuple[Name, Name]] = Field(default_factory=list)


# --- what the log holds ---------------------------------------------------------


class WalkRecord(Strict):
    node: str
    y: float = Field(0.0, ge=0, le=1)
    sev: Sev | None = None
    key: str | None = None
    alias: str | None = None  # a new name the refuter matched to this direction


class FanoutRecord(WalkRecord):
    sev: Finding
    parents: list[str] | None = None


class AuditRecord(Strict):
    node: str
    found: bool


class ReviewEvent(Strict):
    op: Literal["review"]
    v: int = LOG_VERSION
    n: int = Field(ge=1)
    pack: str = ""
    part: float = Field(1.0, gt=0, le=1)  # share of the project the review examined; 1 is the whole of it
    tree: str = ""  # salted hash of every file's bytes, so an unchanged project is not reviewed again
    walks: list[WalkRecord]
    fanout: list[FanoutRecord] = Field(default_factory=list)
    audit: AuditRecord | None = None
    merge: list[tuple[str, str]] = Field(default_factory=list)


class FeedbackEvent(Strict):
    op: Literal["feedback"]
    v: int = LOG_VERSION
    key: str
    verdict: Literal["fixed", "wrong"]


class PackEvent(Strict):
    """Acknowledges that the pack changed: replay from here on uses the new one, by explicit choice."""

    op: Literal["pack"]
    v: int = LOG_VERSION
    fingerprint: str
    previous: str


Event = Annotated[ReviewEvent | FeedbackEvent | PackEvent, Field(discriminator="op")]
EVENT = TypeAdapter(Event)

SCHEMAS: dict[str, type[BaseModel]] = {"payload": Payload}


FRIENDLY = {
    NAME_RE: "must be 1-4 lowercase words joined by hyphens",
    SITE_RE: "must be relpath:line",
}


def errors(exc: Exception) -> list[str]:
    """Pydantic errors as short lines an agent can act on: where, and what to do."""
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return [str(exc)]
    out = []
    for e in exc.errors():
        where = ".".join(str(p) for p in e["loc"])
        msg = e["msg"]
        if e["type"] == "string_pattern_mismatch":
            msg = f"{e.get('input')!r} {FRIENDLY.get(e['ctx']['pattern'], msg)}"
        elif e["type"] == "value_error" and e["loc"] and e["loc"][-1] == "claim":
            err = e.get("ctx", {}).get("error")
            msg = str(err) if err else "claim must not contain newlines or control characters"
        elif e["loc"] and e["loc"][-1] == "claim":
            msg = "claim of 20-600 characters: the input or state, and the wrong outcome"
        out.append(f"{where}: {msg}")
    return out
