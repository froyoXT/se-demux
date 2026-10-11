import json
import re
import shutil
import tempfile
import tomllib
import unittest
from contextlib import contextmanager
from pathlib import Path

from neural_engine import guard, pack, render, review
from neural_engine.config import REVIEW
from neural_engine.graph import seeded
from neural_engine.schema import name_error
from neural_engine.store import Store


def copy_pack() -> Path:
    folder = Path(tempfile.mkdtemp()) / "pack"
    shutil.copytree(pack.DEFAULT, folder)
    return folder


def recipe_text(rule: str, wrong: str, right: str, cite: str) -> str:
    return f"rule: {rule}\nwrong: {wrong}\nright: {right}\ncite: {cite}\n"


HOUSE = recipe_text(
    "a handoff names the one person who receives the work",
    "the group will pick it up later",
    "one person receives the work",
    "the line that names who receives the work",
)
STEP = recipe_text(
    "counters move by a single step",
    "the counter jumps ahead by two",
    "the counter advances by one",
    "the line that changes the counter",
)


def write_pack(
    folder: Path,
    name: str,
    rows: str,
    extends: str | None = None,
    recipes: dict[str, str] | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    extends_line = f'extends = "{extends}"\n' if extends else ""
    files = 'seed = "seed.tsv"\n'
    if recipes is not None:
        files += 'recipes = "recipes"\n'
    (folder / "pack.toml").write_text(
        f'[pack]\nname = "{name}"\nversion = "1"\n{extends_line}\n[files]\n{files}',
        encoding="utf-8",
    )
    text = rows if rows.endswith("\n") or not rows else rows + "\n"
    (folder / "seed.tsv").write_text(text, encoding="utf-8")
    if recipes is not None:
        rec = folder / "recipes"
        rec.mkdir(exist_ok=True)
        for fname, body in recipes.items():
            dest = rec / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
    return folder


def apply_empty(store: Store, root: Path) -> None:
    _, pending = review.plan(store, root)
    walks = [
        {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
        for i, n in enumerate(pending["walk"])
    ]
    raw: dict = {"plan": pending["id"], "walks": walks}
    if target := guard.audit_target(pending, walks):
        raw["audit"] = {"node": target, "found": False}
    review.apply(store, raw)


class Pack(unittest.TestCase):
    def test_shipped_pack_loads_and_is_consistent(self):
        p = pack.load()
        self.assertIsNone(name_error(p.name))
        self.assertEqual(p.name, "software-engineering")
        self.assertEqual(len(p.seeds), 8)
        self.assertEqual(len(p.fingerprint), 12)
        self.assertTrue(all(s.parents == () and s.recipe == () for s in p.seeds))

    def test_broken_pack_is_refused(self):
        folder = copy_pack()
        with open(folder / "seed.tsv", "a") as fh:
            fh.write("Not A Name\t0.5\n")
        with self.assertRaises(ValueError):
            pack.load(folder)

    def test_engine_holds_no_field_knowledge(self):
        loaded = pack.load()
        # `logic` is a seed and also the start of the math term "logistic".
        words = [loaded.name, *(s.name for s in loaded.seeds), "store-api", "device-loop"]
        engine = Path(pack.__file__).parent
        for path in sorted(engine.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for word in words:
                if word == "logic":
                    hit = re.search(r"(?<![A-Za-z0-9-])logic(?![A-Za-z0-9-])", text)
                    self.assertIsNone(hit, f"{path.name} mentions {word}")
                else:
                    self.assertNotIn(word, text, f"{path.name} mentions {word}")


class Repack(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.root = d / "repo"
        self.root.mkdir()
        (self.root / "app.src").write_text("x = 1\n" * 50)
        self.home = d / "home"
        self.folder = copy_pack()

    def review_once(self, store):
        _, pending = review.plan(store, self.root)
        walks = [
            {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
            for i, n in enumerate(pending["walk"])
        ]
        raw = {"plan": pending["id"], "walks": walks}
        if target := guard.audit_target(pending, walks):
            raw["audit"] = {"node": target, "found": False}
        review.apply(store, raw)

    def test_changed_pack_stops_planning_until_repacked(self):
        self.review_once(Store(self.home, pack.load(self.folder)))
        with open(self.folder / "seed.tsv", "a") as fh:
            fh.write("new-direction\t0.5\n")
        store = Store(self.home, pack.load(self.folder))
        with self.assertRaises(ValueError) as cm:
            review.plan(store, self.root)
        self.assertIn("repack", str(cm.exception))
        self.assertTrue(store.check())
        store.repack()
        self.assertEqual(store.check(), [])
        self.review_once(store)

    def test_feedback_stops_when_the_pack_changed(self):
        store = Store(self.home, pack.load(self.folder))
        _, pending = review.plan(store, self.root)
        claim = "a concrete input that leads to a concrete wrong outcome"
        walks = []
        for i, name in enumerate(pending["walk"]):
            w = {"node": name, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
            if i == 0:
                w.update(verdict="hit", sev="major", claim=claim, check="reproduced")
            walks.append(w)
        raw = {"plan": pending["id"], "walks": walks}
        if target := guard.audit_target(pending, walks):
            raw["audit"] = {"node": target, "found": False}
        out = "\n".join(review.apply(store, raw))
        key = out.split("finding ids (for neural-engine feedback): ")[1].split("=")[1].split(",")[0].strip()
        with open(self.folder / "seed.tsv", "a", encoding="utf-8") as fh:
            fh.write("new-direction\t0.5\n")
        moved = Store(self.home, pack.load(self.folder))
        with self.assertRaises(ValueError) as cm:
            review.feedback(moved, key, "wrong")
        self.assertIn("repack", str(cm.exception))
        self.assertFalse(any(ev.get("op") == "feedback" for ev in moved.events()))

    def _diverge(self, store: Store) -> Store:
        """Change the pack on disk. The next time `store` takes its lock, a second store repacks first."""
        with open(self.folder / "seed.tsv", "a", encoding="utf-8") as fh:
            fh.write("new-direction\t0.5\n")
        fresh = Store(self.home, pack.load(self.folder))
        real = store.locked

        @contextmanager
        def locked():
            fresh.repack()
            with real():
                yield

        store.locked = locked
        return fresh

    def test_a_repack_while_plan_waits_is_not_saved(self):
        store = Store(self.home, pack.load(self.folder))
        self.review_once(store)
        (self.root / "app.src").write_text("x = 2\n" * 50)
        fresh = self._diverge(store)
        with self.assertRaises(ValueError) as cm:
            review.plan(store, self.root)
        self.assertIn("repack", str(cm.exception))
        self.assertFalse(store.pending.exists())
        self.assertEqual([ev["op"] for ev in store.events()], ["review", "pack"])
        self.assertEqual(store.events()[-1]["fingerprint"], fresh.pack.fingerprint)
        self.assertEqual(fresh.check(), [])

    def test_a_repack_while_apply_waits_is_not_committed(self):
        store = Store(self.home, pack.load(self.folder))
        self.review_once(store)
        (self.root / "app.src").write_text("x = 2\n" * 50)
        _, pending = review.plan(store, self.root)
        walks = [
            {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 2"}
            for i, n in enumerate(pending["walk"])
        ]
        raw: dict = {"plan": pending["id"], "walks": walks}
        if target := guard.audit_target(pending, walks):
            raw["audit"] = {"node": target, "found": False}
        fresh = self._diverge(store)
        with self.assertRaises(ValueError) as cm:
            review.apply(store, raw)
        self.assertIn("repack", str(cm.exception))
        self.assertEqual([ev["op"] for ev in store.events()], ["review", "pack"])
        self.assertEqual(store.events()[-1]["fingerprint"], fresh.pack.fingerprint)
        self.assertTrue(store.pending.exists())
        self.assertEqual(fresh.check(), [])

    def test_a_repack_while_feedback_waits_is_not_committed(self):
        store = Store(self.home, pack.load(self.folder))
        _, pending = review.plan(store, self.root)
        claim = "a concrete input that leads to a concrete wrong outcome"
        walks = []
        for i, name in enumerate(pending["walk"]):
            w = {"node": name, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
            if i == 0:
                w.update(verdict="hit", sev="major", claim=claim, check="reproduced")
            walks.append(w)
        raw: dict = {"plan": pending["id"], "walks": walks}
        if target := guard.audit_target(pending, walks):
            raw["audit"] = {"node": target, "found": False}
        out = "\n".join(review.apply(store, raw))
        key = out.split("finding ids (for neural-engine feedback): ")[1].split("=")[1].split(",")[0].strip()
        fresh = self._diverge(store)
        with self.assertRaises(ValueError) as cm:
            review.feedback(store, key, "wrong")
        self.assertIn("repack", str(cm.exception))
        self.assertEqual([ev["op"] for ev in store.events()], ["review", "pack"])
        self.assertFalse(any(ev.get("op") == "feedback" for ev in store.events()))
        self.assertEqual(fresh.check(), [])

    def test_lockfile_matches_the_project(self):
        root = Path(__file__).resolve().parents[1]
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        lock = (root / "uv.lock").read_text(encoding="utf-8")
        self.assertIn(f'name = "se-demux"\nversion = "{version}"', lock)

    def test_log_events_are_versioned_and_validated(self):
        store = Store(self.home, pack.load(self.folder))
        self.review_once(store)
        event = json.loads(store.log.read_text().splitlines()[0])
        self.assertEqual((event["op"], event["v"]), ("review", 1))
        with open(store.log, "a") as fh:
            fh.write(json.dumps({"op": "review", "n": 2, "scope": "", "walks": [{"node": "x", "y": 7}]}) + "\n")
        self.assertIn("not a valid event", store.check()[0])


def _bare(folder: Path, name: str, extends: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pack.toml").write_text(
        f'[pack]\nname = "{name}"\nversion = "1"\nextends = "{extends}"\n',
        encoding="utf-8",
    )
    return folder


def _pair(left: str, right: str, field: str = "rule") -> Path:
    """Two child recipes. `field` is the one whose text is `left` and `right`; the others differ."""
    bits = {
        "rule": ("counters move by a single step", "a tray holds one batch"),
        "wrong": ("the counter jumps ahead by two", "the tray is shared across batches"),
        "right": ("the counter advances by one", "the tray holds a single batch"),
    }
    bits[field] = (left, right)
    rows = "root\nwest-note\t0.5\troot\neast-note\t0.5\troot\n"
    cite_l, cite_r = "the line that changes the counter", "the line that fills the tray"
    return write_pack(
        Path(tempfile.mkdtemp()) / "pack",
        "field-notes",
        rows,
        recipes={
            "west-note.md": recipe_text(bits["rule"][0], bits["wrong"][0], bits["right"][0], cite_l),
            "east-note.md": recipe_text(bits["rule"][1], bits["wrong"][1], bits["right"][1], cite_r),
        },
    )


class Extends(unittest.TestCase):
    def test_extends_by_name_adds_a_child_and_overrides_a_value(self):
        folder = Path(tempfile.mkdtemp()) / "northwind"
        write_pack(
            folder,
            "northwind",
            "security\t0.4\nhandoff-notes\t0.5\tlogic\n",
            extends="software-engineering",
            recipes={"handoff-notes.md": HOUSE, ".keep": "hidden\n"},
        )
        (folder / "recipes" / ".notes").mkdir()
        (folder / "recipes" / ".notes" / "secret.md").write_text(HOUSE, encoding="utf-8")
        loaded = pack.load(folder)
        base = pack.load()
        self.assertEqual([s.name for s in loaded.seeds], [s.name for s in base.seeds] + ["handoff-notes"])
        security = next(s for s in loaded.seeds if s.name == "security")
        self.assertEqual((security.value, security.parents, security.recipe), (0.4, (), ()))
        child = loaded.seeds[-1]
        self.assertEqual(child.parents, ("logic",))
        self.assertEqual(child.recipe[0], ("rule", "a handoff names the one person who receives the work"))
        self.assertEqual([label for label, _ in child.recipe], ["rule", "wrong", "right", "cite"])

    def test_extends_by_relative_path_chains(self):
        root = Path(tempfile.mkdtemp())
        gamma = recipe_text(
            "a tray holds one batch",
            "the tray is shared across batches",
            "the tray holds a single batch",
            "the line that fills the tray",
        )
        write_pack(root / "base", "base-field", "alpha\nbeta\t0.6\n")
        write_pack(
            root / "mid",
            "mid-field",
            "gamma\t0.5\talpha\n",
            extends="../base",
            recipes={"gamma.md": gamma},
        )
        write_pack(root / "leaf", "leaf-field", "delta\t\tgamma,beta\n", extends="../mid")
        loaded = pack.load(root / "leaf")
        self.assertEqual([s.name for s in loaded.seeds], ["alpha", "beta", "gamma", "delta"])
        by = {s.name: s for s in loaded.seeds}
        self.assertEqual(by["alpha"].value, 0.5)
        self.assertEqual(by["beta"].value, 0.6)
        self.assertEqual(by["gamma"].parents, ("alpha",))
        self.assertEqual(by["gamma"].recipe[0][0], "rule")
        self.assertEqual(by["delta"].parents, ("gamma", "beta"))
        self.assertEqual(by["delta"].value, 0.5)
        self.assertEqual(by["delta"].recipe, ())

    def test_the_loader_refuses_a_bad_pack(self):
        tmp = Path(tempfile.mkdtemp())

        def refused(folder: Path, needle: str) -> None:
            with self.assertRaises(ValueError) as cm:
                pack.load(folder)
            self.assertIn(needle, str(cm.exception), str(cm.exception))

        child = "alpha\nbeta\t0.5\talpha\n"
        refused(
            write_pack(tmp / "root-recipe", "field-notes", "alpha\n", recipes={"alpha.md": STEP}),
            "a root is a universal domain",
        )
        refused(
            write_pack(tmp / "unknown-recipe", "field-notes", "alpha\n", recipes={"beta.md": STEP}),
            "beta is not a seed",
        )
        refused(
            write_pack(tmp / "unknown-parent", "field-notes", "alpha\nbeta\t0.5\tlater\n"),
            "parent later of beta is not defined earlier",
        )
        refused(
            write_pack(tmp / "later-parent", "field-notes", "alpha\t0.5\tbeta\nbeta\n"),
            "parent beta of alpha is not defined earlier",
        )
        refused(
            write_pack(tmp / "three", "field-notes", "a\nb\nc\nd\t0.5\ta,b,c\n"),
            "has 3 parents",
        )
        refused(write_pack(tmp / "self", "field-notes", "alpha\t0.5\talpha\n"), "is its own parent")
        refused(
            write_pack(tmp / "over-parents", "northwind", "logic\t0.4\tsecurity\n", extends="software-engineering"),
            "cannot set parents",
        )
        refused(write_pack(tmp / "dup", "field-notes", "alpha\nalpha\t0.6\n"), "duplicate name alpha")
        cycle_a, cycle_b = tmp / "cyc-a", tmp / "cyc-b"
        _bare(cycle_a, "pack-alpha", "../cyc-b")
        _bare(cycle_b, "pack-beta", "../cyc-a")
        refused(cycle_a, "extends cycle")
        refused(_bare(tmp / "same-name", "software-engineering", "software-engineering"), "same as a pack it extends")
        refused(_bare(tmp / "unknown-ext", "field-notes", "not-a-pack"), "is not a known pack")
        refused(_bare(tmp / "absolute", "field-notes", "/no/such"), "must be relative")
        refused(
            write_pack(
                tmp / "base-recipe",
                "northwind",
                "security\t0.4\n",
                extends="software-engineering",
                recipes={"logic.md": STEP},
            ),
            "a recipe for a base seed is not allowed",
        )
        refused(
            write_pack(tmp / "not-md", "field-notes", child, recipes={"notes.txt": "x\n", "beta.md": STEP}),
            "notes.txt: a recipe file must be named",
        )
        refused(
            write_pack(tmp / "subdir", "field-notes", child, recipes={"beta.md": STEP, "extra/x.md": "x\n"}),
            "extra: a recipe folder cannot contain a subfolder",
        )
        short = (
            "rule: counters move by a single step\n"
            "wrong: the counter jumps ahead by two\n"
            "right: the counter advances by one\n"
        )
        refused(write_pack(tmp / "missing", "field-notes", child, recipes={"beta.md": short}), "exactly four lines")
        repeated = "rule: one\nrule: two\nwrong: the counter jumps ahead by two\nright: the counter advances by one\n"
        refused(
            write_pack(tmp / "repeated", "field-notes", child, recipes={"beta.md": repeated}),
            "expected wrong next, found rule",
        )
        refused(
            write_pack(tmp / "extra-line", "field-notes", child, recipes={"beta.md": STEP + "note: extra\n"}),
            "exactly four lines",
        )
        swapped = (
            "wrong: the counter jumps ahead by two\n"
            "rule: counters move by a single step\n"
            "right: the counter advances by one\n"
            "cite: the line that changes the counter\n"
        )
        refused(
            write_pack(tmp / "order", "field-notes", child, recipes={"beta.md": swapped}),
            "expected rule next, found wrong",
        )
        empty = (
            "rule:\n"
            "wrong: the counter jumps ahead by two\n"
            "right: the counter advances by one\n"
            "cite: the line that changes the counter\n"
        )
        refused(write_pack(tmp / "empty-value", "field-notes", child, recipes={"beta.md": empty}), "rule is empty")
        same = recipe_text(
            "counters move by a single step",
            "the counter jumps",
            "the counter jumps",
            "the line that changes the counter",
        )
        refused(
            write_pack(tmp / "same-fix", "field-notes", child, recipes={"beta.md": same}),
            "wrong and right must differ",
        )
        long = "rule: " + ("n" * 295) + "\n" + STEP.split("\n", 1)[1]
        refused(write_pack(tmp / "long", "field-notes", child, recipes={"beta.md": long}), "over 300")
        linked = recipe_text(
            "see http://example.test/rule for the counter",
            "the counter jumps ahead by two",
            "the counter advances by one",
            "the line that changes the counter",
        )
        refused(write_pack(tmp / "link", "field-notes", child, recipes={"beta.md": linked}), "cannot contain a link")
        refused(_pair("Alpha   Beta", "alpha beta"), "west-note.md and east-note.md share a rule")
        refused(
            _pair("The Counter Jumps", "the   counter   jumps", "wrong"),
            "west-note.md and east-note.md share a wrong",
        )
        refused(_pair("Same Fix", "same   fix", "right"), "west-note.md and east-note.md share a right")
        refused(
            _pair("one two three four five six seven eight", "one two three four five six seven nine"),
            "west-note.md and east-note.md rules overlap",
        )
        distinct = _pair("red box on the shelf", "blue cup in the drawer")
        self.assertEqual(len([s for s in pack.load(distinct).seeds if s.recipe]), 2)
        base = tmp / "cross-base"
        leaf = tmp / "cross-leaf"
        eight = "one two three four five six seven eight"
        nine = "one two three four five six seven nine"
        write_pack(
            base,
            "base-field",
            "domain\nbase-note\t0.5\tdomain\n",
            recipes={
                "base-note.md": recipe_text(eight, "the tray is shared", "the tray is private", "the filling line")
            },
        )
        write_pack(
            leaf,
            "leaf-field",
            "child-note\t0.5\tdomain\n",
            extends="../cross-base",
            recipes={
                "child-note.md": recipe_text(
                    nine, "the counter jumps ahead", "the counter steps once", "the counter line"
                )
            },
        )
        refused(leaf, "base-note.md and child-note.md rules overlap")
        write_pack(
            base,
            "base-field",
            "domain\nbase-note\t0.5\tdomain\n",
            recipes={
                "base-note.md": recipe_text(
                    "alpha beta", "the tray is shared", "the tray is private", "the filling line"
                )
            },
        )
        write_pack(
            leaf,
            "leaf-field",
            "child-note\t0.5\tdomain\n",
            extends="../cross-base",
            recipes={
                "child-note.md": recipe_text(
                    "Alpha   Beta", "the counter jumps ahead", "the counter steps once", "the counter line"
                )
            },
        )
        refused(leaf, "base-note.md and child-note.md share a rule")

    def test_a_root_name_is_an_ordinary_word(self):
        rule = "business logic lives in one place, not in screen code"
        loaded = pack.load(
            write_pack(
                Path(tempfile.mkdtemp()) / "northwind",
                "northwind",
                "handoff-notes\t0.5\tlogic\n",
                extends="software-engineering",
                recipes={
                    "handoff-notes.md": recipe_text(
                        rule,
                        "the screen computes the total itself",
                        "the screen calls the one place that owns the total",
                        "the line in the screen that computes it",
                    )
                },
            )
        )
        by = {s.name: s for s in loaded.seeds}
        self.assertEqual(dict(by["handoff-notes"].recipe)["rule"], rule)

    def test_a_recipe_that_names_another_child_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        named = recipe_text("gamma must stay put", "gamma moves aside", "it stays put", "the line that moves it")
        with self.assertRaises(ValueError) as cm:
            pack.load(
                write_pack(
                    tmp / "siblings",
                    "field-notes",
                    "alpha\nbeta\t0.5\talpha\ngamma\t0.5\talpha\n",
                    recipes={"beta.md": named},
                )
            )
        self.assertIn("mentions another child (gamma)", str(cm.exception))
        write_pack(
            tmp / "base",
            "base-field",
            "domain\nbase-note\t0.5\tdomain\n",
            recipes={"base-note.md": STEP},
        )
        mentions = recipe_text(
            "base-note must stay put", "base-note moves aside", "it stays put", "the line that moves it"
        )
        with self.assertRaises(ValueError) as cm:
            pack.load(
                write_pack(
                    tmp / "leaf",
                    "leaf-field",
                    "child-note\t0.5\tdomain\n",
                    extends="../base",
                    recipes={"child-note.md": mentions},
                )
            )
        self.assertIn("mentions another child (base-note)", str(cm.exception))

    def test_a_child_name_inside_a_hyphenated_token_is_accepted(self):
        rule = "a payload pre-color-shift stays inside one word"
        loaded = pack.load(
            write_pack(
                Path(tempfile.mkdtemp()) / "pack",
                "field-notes",
                "pay\ncolor-shift\t0.5\tpay\nnote-taking\t0.5\tpay\n",
                recipes={
                    "note-taking.md": recipe_text(
                        rule,
                        "the payload splits the pre-color-shift",
                        "the payload keeps the pre-color-shift whole",
                        "the line that builds the payload",
                    )
                },
            )
        )
        by = {s.name: s for s in loaded.seeds}
        self.assertEqual(dict(by["note-taking"].recipe)["rule"], rule)


class Fingerprint(unittest.TestCase):
    def test_a_comment_does_not_change_it_and_a_recipe_does(self):
        d = Path(tempfile.mkdtemp())
        root = d / "repo"
        root.mkdir()
        (root / "app.src").write_text("x = 1\n" * 30, encoding="utf-8")
        folder = write_pack(d / "pack", "field-notes", "alpha\nbeta\t0.5\talpha\n", recipes={"beta.md": STEP})
        dark = REVIEW["dark"]
        REVIEW["dark"] = 0
        try:
            store = Store(d / "home", pack.load(folder))
            before = store.pack.fingerprint
            apply_empty(store, root)
            (folder / "seed.tsv").write_text("# note\n\nalpha\t\nbeta\t0.5\talpha\n", encoding="utf-8")
            commented = pack.load(folder)
            self.assertEqual(commented.fingerprint, before)
            (root / "app.src").write_text("x = 2\n" * 30, encoding="utf-8")
            review.plan(Store(d / "home", commented), root)
            text = (folder / "recipes" / "beta.md").read_text(encoding="utf-8")
            (folder / "recipes" / "beta.md").write_text(text.replace("single", "single counted"), encoding="utf-8")
            moved = pack.load(folder)
            self.assertNotEqual(moved.fingerprint, before)
            changed = Store(d / "home", moved)
            with self.assertRaises(ValueError) as cm:
                review.plan(changed, root)
            self.assertIn("repack", str(cm.exception))
            changed.repack()
            review.plan(changed, root)
        finally:
            REVIEW["dark"] = dark


class PlanView(unittest.TestCase):
    def test_a_recipe_is_shown_only_for_a_walked_direction(self):
        d = Path(tempfile.mkdtemp())
        folder = write_pack(d / "pack", "field-notes", "alpha\nbeta\t0.5\talpha\n", recipes={"beta.md": STEP})
        root = d / "repo"
        root.mkdir()
        (root / "app.src").write_text("x = 1\n" * 20, encoding="utf-8")
        rule = "counters move by a single step"
        dark = REVIEW["dark"]
        REVIEW["dark"] = 0
        try:
            store = Store(d / "home", pack.load(folder))
            g, pending = review.plan(store, root)
            self.assertEqual(sorted(pending["walk"]), ["alpha", "beta"])
            view = render.plan_view(g, pending)
            self.assertIn("recipes  the rule to check", view)
            self.assertIn("\n  beta\n", view)
            self.assertIn(rule, view)
            self.assertLess(view.index("look"), view.index("recipes"))
            self.assertLess(view.index("recipes"), view.index("fan out"))
            quiet = dict(pending)
            quiet["walk"] = ["alpha"]
            hidden = render.plan_view(g, quiet)
            self.assertNotIn("recipes  ", hidden)
            self.assertNotIn(rule, hidden)
            walks = [
                {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
                for i, n in enumerate(pending["walk"])
            ]
            brief = review.verify(store, {"plan": pending["id"], "walks": walks})
            self.assertNotIn(rule, brief)
        finally:
            REVIEW["dark"] = dark


class ChildSeed(unittest.TestCase):
    def test_a_child_seed_stays_a_seed(self):
        d = Path(tempfile.mkdtemp())
        folder = write_pack(
            d / "pack",
            "northwind",
            "security\t0.4\nhandoff-notes\t0.5\tlogic\n",
            extends="software-engineering",
            recipes={"handoff-notes.md": HOUSE},
        )
        loaded = pack.load(folder)
        g = seeded(loaded)
        child = g.require("handoff-notes")
        self.assertEqual(child.origin, "seed")
        self.assertEqual(child.parents, ["logic"])
        self.assertGreater(child.radius(g.x()), REVIEW["dark"])
        shown = render.table(g, loaded.name)
        self.assertTrue(shown.startswith("northwind · "))
        self.assertIn("└ handoff-notes", shown)
        for _ in range(40):
            g.step({"op": "review", "n": g.n + 1, "walks": [{"node": "handoff-notes", "y": 0.0}]})
        child = g.require("handoff-notes")
        self.assertFalse(child.deleted)
        self.assertLess(child.radius(g.x()), REVIEW["dark"])

        root = d / "repo"
        root.mkdir()
        (root / "app.src").write_text("x = 1\n" * 30, encoding="utf-8")
        claim = "a concrete input that leads to a concrete wrong outcome"
        dark = REVIEW["dark"]
        REVIEW["dark"] = 0
        try:
            store = Store(d / "home", loaded)
            _, pending = review.plan(store, root)
            walks = [
                {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 1"}
                for i, n in enumerate(pending["walk"])
            ]
            raw: dict = {
                "plan": pending["id"],
                "walks": walks,
                "fanout": [
                    {
                        "node": "desk-owner",
                        "parents": ["handoff-notes"],
                        "sev": "major",
                        "site": "app.src:20",
                        "quote": "x = 1",
                        "claim": claim,
                        "check": "reproduced",
                    }
                ],
            }
            if target := guard.audit_target(pending, walks):
                raw["audit"] = {"node": target, "found": False}
            review.apply(store, raw)
            self.assertEqual(store.graph().require("desk-owner").parents, ["handoff-notes"])
            (root / "app.src").write_text("x = 2\n" * 30, encoding="utf-8")
            _, pending = review.plan(store, root)
            walks = [
                {"node": n, "verdict": "empty", "site": f"app.src:{i + 1}", "quote": "x = 2"}
                for i, n in enumerate(pending["walk"])
            ]
            raw = {"plan": pending["id"], "walks": walks, "merge": [["handoff-notes", "logic"]]}
            if target := guard.audit_target(pending, walks):
                raw["audit"] = {"node": target, "found": False}
            with self.assertRaises(ValueError) as cm:
                review.apply(store, raw)
            self.assertIn("recently born", str(cm.exception))
            self.assertFalse(store.graph().require("handoff-notes").deleted)
        finally:
            REVIEW["dark"] = dark


class Homes(unittest.TestCase):
    def test_two_packs_on_one_root_keep_separate_logs(self):
        d = Path(tempfile.mkdtemp())
        root = d / "repo"
        root.mkdir()
        (root / "app.src").write_text("x = 1\n" * 20, encoding="utf-8")
        home = d / "home"
        store_a = Store(home, pack.load(write_pack(d / "a", "field-notes", "alpha\n")))
        store_b = Store(home, pack.load(write_pack(d / "b", "other-notes", "beta\n")))
        self.assertNotEqual(store_a.home, store_b.home)
        apply_empty(store_a, root)
        self.assertEqual(store_a.check(), [])
        self.assertEqual(store_b.check(), [])
        review.plan(store_b, root)
        self.assertTrue(store_a.log.exists())
        self.assertFalse(store_b.log.exists())


if __name__ == "__main__":
    unittest.main()
