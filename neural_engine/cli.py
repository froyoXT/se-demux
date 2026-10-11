"""neural-engine: plan | verify | apply | feedback | show | check | repack | schema   (see README)"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import pack, render, review
from .schema import SCHEMAS
from .store import Store


def _json(path: Path) -> dict:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neural-engine")
    ap.add_argument(
        "--home",
        type=Path,
        default=None,
        help="root of pack homes (default: $NEURAL_ENGINE_HOME or ~/.neural-engine)",
    )
    ap.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="pack folder (default: $NEURAL_ENGINE_PACK, else the shipped pack)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="choose this review's directions")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--changed", type=Path, help="a diff review: a file listing the changed paths, one per line")
    v = sub.add_parser("verify", help="print the refuter's brief for a payload")
    v.add_argument("--from", dest="src", type=Path, required=True)
    a = sub.add_parser("apply", help="learn from a finished, verified review")
    a.add_argument("--from", dest="src", type=Path, required=True)
    a.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("feedback", help="a human verdict on a finding: fixed or wrong")
    f.add_argument("key")
    f.add_argument("verdict", choices=["fixed", "wrong"])
    sub.add_parser("show", help="the network: every direction's radius, family and health")
    sub.add_parser("check", help="replay the log and verify the cache and pack")
    sub.add_parser("repack", help="accept a changed pack: replay continues under it")
    sc = sub.add_parser("schema", help="print the JSON Schema an agent must follow")
    sc.add_argument("name", choices=sorted(SCHEMAS))
    args = ap.parse_args(argv)
    try:
        if args.cmd == "schema":
            print(json.dumps(SCHEMAS[args.name].model_json_schema(by_alias=True), indent=1))
            return 0
        folder = args.pack
        if folder is None and (env := os.environ.get("NEURAL_ENGINE_PACK")):
            folder = Path(env)
        store = Store(args.home, pack.load(folder or pack.DEFAULT))
        if args.cmd == "plan":
            changed = None
            if args.changed:
                changed = [ln.strip() for ln in args.changed.read_text(encoding="utf-8").splitlines() if ln.strip()]
            try:
                print(render.plan_view(*review.plan(store, args.root, changed)), end="")
            except review.Unchanged as exc:
                print(exc)
        elif args.cmd == "verify":
            print(review.verify(store, _json(args.src)), end="")
        elif args.cmd == "apply":
            print("\n".join(review.apply(store, _json(args.src), args.dry_run)))
        elif args.cmd == "feedback":
            print("\n".join(review.feedback(store, args.key, args.verdict)))
        elif args.cmd == "show":
            print(render.table(store.graph(), store.pack.name), end="")
        elif args.cmd == "repack":
            print("\n".join(store.repack()))
        else:
            fails = store.check()
            print("PASS" if not fails else "FAIL\n  " + "\n  ".join(fails))
            return 1 if fails else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"refused:\n{exc}", file=sys.stderr)
        return 1
    return 0
