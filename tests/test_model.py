import random
import unittest
from dataclasses import replace

from neural_engine import pack, render
from neural_engine.config import PRIOR, REVIEW
from neural_engine.graph import Graph, seeded
from neural_engine.model import BIAS, Direction, born, features
from neural_engine.pack import Seed

PACK = pack.load()


X = features()


def fresh():
    return Direction("x", "seed", prior={BIAS: PRIOR["seed_bias"]})


class Unit(unittest.TestCase):
    def test_hit_raises_and_empty_lowers(self):
        up, down = fresh(), fresh()
        p0 = up.p(X)
        up.learn(X, 1.0, 0.6)
        down.learn(X, 0.0, 0.0)
        self.assertGreater(up.p(X), p0)
        self.assertLess(down.p(X), p0)

    def test_soft_label_and_weight_move_less(self):
        full, soft, light = fresh(), fresh(), fresh()
        full.learn(X, 1.0, 0.6)
        soft.learn(X, 0.5, 0.6)
        light.learn(X, 1.0, 0.6, w=0.25)
        self.assertGreater(full.p(X), soft.p(X))
        self.assertGreater(full.p(X), light.p(X))
        self.assertGreater(light.p(X), fresh().p(X))

    def test_certainty_accumulates_and_forgetting_returns_it_to_the_prior(self):
        d, x = fresh(), X
        _, v0 = d.logit(x)
        for _ in range(10):
            d.learn(x, 0.0, 0.0)
        _, v1 = d.logit(x)
        for _ in range(1000):
            d.forget()
        _, v2 = d.logit(x)
        self.assertLess(v1, v0)
        self.assertAlmostEqual(v2, v0, places=2)

    def test_rare_but_critical_stays_lit(self):
        rare, quiet = fresh(), fresh()
        for i in range(20):
            rare.learn(X, 1.0 if i % 10 == 0 else 0.0, 1.0)
            quiet.learn(X, 0.0, 0.0)
        self.assertGreater(rare.radius(X), REVIEW["dark"])
        self.assertLess(quiet.radius(X), REVIEW["dark"])

    def test_newborn_inherits_two_parents_less_surely_and_rarer(self):
        a, b = fresh(), fresh()
        for _ in range(10):
            a.learn(X, 1.0, 0.6)
        kid = born("kid", [a, b], 1)
        self.assertEqual(kid.parents, ["x", "x"])
        self.assertLess(kid.p(X), a.p(X))  # specialisation starts rarer
        self.assertGreater(kid.logit(X)[1], a.logit(X)[1])  # and more uncertain


class Table(unittest.TestCase):
    def test_a_second_live_parent_is_named_on_the_line(self):
        g = Graph()
        g.directions["logic"] = Direction("logic", "seed")
        g.directions["security"] = Direction("security", "seed")
        g.directions["handoff"] = Direction("handoff", "birth", parents=["logic", "security"])
        shown = render.table(g, "field")
        line = next(ln for ln in shown.splitlines() if "handoff" in ln)
        self.assertIn("└ handoff", line)
        self.assertIn("(also under security)", line)
        self.assertEqual(sum("handoff" in ln for ln in shown.splitlines()), 1)
        security = next(ln for ln in shown.splitlines() if "security" in ln and "also under" not in ln)
        self.assertNotIn("└", security)
        self.assertLess(shown.index("logic"), shown.index("└ handoff"))
        self.assertLess(shown.index("└ handoff"), shown.index(security))

        g.directions["security"].deleted = True
        hidden = render.table(g, "field")
        self.assertIn("└ handoff", hidden)
        self.assertNotIn("also under", hidden)


class Seeding(unittest.TestCase):
    def setUp(self):
        self.g = seeded(PACK)

    def test_seed_is_lit_peers(self):
        self.assertEqual(len(self.g.live()), len(PACK.seeds))
        self.assertTrue(all(d.parents == [] for d in self.g.live()))
        self.assertTrue(all(d.radius(X) > REVIEW["dark"] for d in self.g.live()))

    def test_plan_respects_the_cap(self):
        walk = self.g.plan(random.Random(1))
        self.assertTrue(0 < len(walk) <= REVIEW["walk"])

    def test_empty_graph_plans_nothing(self):
        self.assertEqual(Graph().plan(random.Random(1)), [])


class Step(unittest.TestCase):
    def review(self, g, walks, fanout=(), audit=None, merge=None, **extra):
        ev = {"op": "review", "n": g.n + 1, "walks": list(walks), "fanout": list(fanout)}
        ev.update(extra)
        if audit:
            ev["audit"] = audit
        if merge:
            ev["merge"] = merge
        return g.step(ev)

    def graph(self, *names):
        tiny = replace(
            PACK,
            seeds=tuple(Seed(n, 0.5) for n in names),
        )
        return seeded(tiny)

    def test_repeat_finding_is_learned_at_reduced_weight(self):
        once, twice = self.graph("x"), self.graph("x")
        hit = {"node": "x", "y": 1.0, "sev": "major", "key": "k1"}
        self.review(once, [hit])
        self.review(once, [{"node": "x", "y": 1.0, "sev": "major", "key": "k2"}])
        self.review(twice, [hit])
        self.assertTrue(any("repeat of k1" in n for n in self.review(twice, [hit])))
        self.assertGreater(once.find("x").p(X), twice.find("x").p(X))

    def test_birth_alias_and_merge(self):
        g = self.graph("n-plus-one")
        self.review(
            g, [], fanout=[{"node": "orm-batching", "parents": ["n-plus-one"], "y": 1.0, "sev": "minor", "key": "a"}]
        )
        self.assertEqual(g.find("orm-batching").parents, ["n-plus-one"])
        self.review(
            g, [], fanout=[{"node": "n-plus-one", "alias": "query-in-loop", "y": 1.0, "sev": "major", "key": "b"}]
        )
        self.assertIs(g.find("query-in-loop"), g.find("n-plus-one"))
        self.review(
            g, [], fanout=[{"node": "orm-prefetch", "parents": ["orm-batching"], "y": 1.0, "sev": "minor", "key": "c"}]
        )
        self.review(g, [], merge=[["orm-batching", "n-plus-one"]])
        self.assertIs(g.find("orm-batching"), g.find("n-plus-one"))
        self.assertEqual(g.find("orm-prefetch").parents, ["n-plus-one"])  # children follow the merge
        self.assertEqual(g.find("n-plus-one").parents, [])

    def test_stillborn_newborn_is_deleted_but_a_proven_one_is_not(self):
        g = self.graph()
        self.review(
            g,
            [],
            fanout=[
                {"node": "junk-idea", "y": 1.0, "sev": "minor", "key": "j"},
                {"node": "real-idea", "y": 1.0, "sev": "major", "key": "r"},
            ],
        )
        for i in range(30):
            walks = [
                {"node": "real-idea", **({"y": 1.0, "sev": "major", "key": f"r{i}"} if i % 3 == 0 else {"y": 0.0})}
            ]
            if g.find("junk-idea"):  # a plan never assigns a deleted direction
                walks.append({"node": "junk-idea", "y": 0.0})
            self.review(g, walks)
        self.assertIsNone(g.find("junk-idea"))
        self.assertIsNotNone(g.find("real-idea"))

    def test_node_budget_replaces_a_dark_newborn(self):
        g = self.graph()
        REVIEW["max_live"], saved = 2, REVIEW["max_live"]
        try:
            self.review(g, [], fanout=[{"node": "first-idea", "y": 1.0, "sev": "minor", "key": "1"}])
            for _ in range(5):
                self.review(g, [{"node": "first-idea", "y": 0.0}])
            self.review(g, [], fanout=[{"node": "second-idea", "y": 1.0, "sev": "major", "key": "2"}])
            notes = self.review(g, [], fanout=[{"node": "third-idea", "y": 1.0, "sev": "major", "key": "3"}])
            self.assertTrue(any(n.startswith("replace first-idea") for n in notes))
            self.assertLessEqual(len(g.live()), 2)
        finally:
            REVIEW["max_live"] = saved

    def test_feedback_wrong_lowers_and_fixed_raises(self):
        g = self.graph("x", "y")
        self.review(
            g,
            [
                {"node": "x", "y": 1.0, "sev": "major", "key": "kx"},
                {"node": "y", "y": 1.0, "sev": "major", "key": "ky"},
            ],
        )
        before = g.find("x").p(X)
        g.step({"op": "feedback", "key": "kx", "verdict": "wrong"})
        g.step({"op": "feedback", "key": "ky", "verdict": "fixed"})
        self.assertLess(g.find("x").p(X), before)
        self.assertGreater(g.find("y").p(X), before)
        self.assertRaises(ValueError, g.step, {"op": "feedback", "key": "kx", "verdict": "fixed"})

    def test_a_later_observation_keeps_feedback(self):
        g = self.graph("x")
        hit = {"node": "x", "y": 1.0, "sev": "major", "key": "k"}
        self.review(g, [hit])
        g.step({"op": "feedback", "key": "k", "verdict": "wrong"})
        self.review(g, [hit])
        self.assertEqual(g.findings["k"]["feedback"], "wrong")
        self.assertRaises(ValueError, g.step, {"op": "feedback", "key": "k", "verdict": "fixed"})

    def test_a_diff_empty_speaks_for_its_part_and_a_diff_hit_for_itself(self):
        whole, diff = self.graph("x"), self.graph("x")
        self.review(whole, [{"node": "x", "y": 0.0}])
        self.review(diff, [{"node": "x", "y": 0.0}], part=0.25)
        before = self.graph("x").find("x").p(X)
        drop_whole, drop_diff = before - whole.find("x").p(X), before - diff.find("x").p(X)
        self.assertAlmostEqual(drop_diff / drop_whole, 0.25, delta=0.1)  # the Newton step is not quite linear in w
        whole, diff = self.graph("x"), self.graph("x")
        self.review(whole, [{"node": "x", "y": 1.0, "sev": "major"}])
        self.review(diff, [{"node": "x", "y": 1.0, "sev": "major"}], part=0.25)
        self.assertAlmostEqual(whole.find("x").p(X), diff.find("x").p(X), places=3)

    def test_bytes_already_reviewed_are_recognised(self):
        g = self.graph("x")
        self.review(g, [{"node": "x", "y": 0.0}], tree="t1")
        self.assertEqual(g.unchanged("t1"), 1)
        self.assertIsNone(g.unchanged("t2"))

    def test_health_report(self):
        g = self.graph("x")
        self.review(g, [{"node": "x", "y": 0.0}], audit={"node": "x", "found": True})
        h = g.report()
        self.assertEqual((h["audits"], h["false_empty_rate"]), (1, 1.0))


if __name__ == "__main__":
    unittest.main()
