"""The design's claims, and each mechanism against its absence, on the adversarial simulator.

Code fails the same ways everywhere: one true rate per direction, varied a little per project.
The reviewer misses, invents, snaps discoveries to generic names and uses synonyms, and a
refuter catches part of it. A claim about one world must hold in at least 4 of 5; a gate
compares totals over 100 worlds of 300 reviews, because over 20 worlds the same mechanism
read as doubling misses on one set of seeds and as nothing on the next.
"""

import unittest

from neural_engine import config, sim
from neural_engine.config import REVIEW
from neural_engine.sim import Behavior as B

DARK = REVIEW["dark"]
WORLDS = range(1, 101)


def score(b=None, **prior):
    saved = {k: config.PRIOR[k] for k in prior}
    config.PRIOR.update(prior)
    try:
        return sim.score(WORLDS, 300, b)
    finally:
        config.PRIOR.update(saved)


class Claims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = []
        for s in range(1, 6):
            trace = {}
            cls.runs.append((sim.run(300, s, trace=trace), trace))

    def holds(self, claim, at_least=4):
        passed = sum(1 for g, t in self.runs if claim(g, t))
        self.assertGreaterEqual(passed, at_least, f"held in {passed} of {len(self.runs)} worlds")

    def test_discoveries_that_pay_are_born_under_their_own_names_and_lit(self):
        paying = [n for n in sim.DISCOVERIES if sim.TRUTH[n] >= 0.15]

        def claim(g, _):
            lit = [n for n in paying if (d := g.find(n)) is not None and d.name == n and d.radius(g.x()) >= DARK]
            return len(lit) >= len(paying) - 2

        self.holds(claim)

    def test_invented_names_mostly_do_not_survive_lit(self):
        """Reviewers keep inventing them, so they keep being reborn; most are deleted or dark at any moment."""
        self.holds(lambda g, _: sum(bool((d := g.find(n)) and d.radius(g.x()) >= DARK) for n in sim.JUNK) <= 1)

    def test_every_seed_stays_lit(self):
        self.holds(lambda g, _: all(g.find(s.name).radius(g.x()) >= DARK for s in sim.PACK.seeds))


class Gates(unittest.TestCase):
    """Each shipped mechanism must beat its absence on the failure it exists for."""

    @classmethod
    def setUpClass(cls):
        cls.shipped = score()

    def test_blind_view_beats_a_reviewer_that_slacks_on_long_shots(self):
        self.assertLess(self.shipped["missed"], score(B(blind=False, effort_follows_odds=True))["missed"])

    def test_a_direction_that_starts_paying_after_going_dark_is_lit_again(self):
        lags = []
        for s in range(1, 21):
            trace = {}
            sim.run(400, s, B(shift=("logic", 0.0, 0.3, 200)), trace)  # no discovery is filed under logic
            lags.append(trace["detected"].get("shift", 999))
        self.assertLess(sorted(lags)[16], 200)  # in at least 17 of 20 worlds, within the remaining 200 reviews

    def test_specialisation_offset_reduces_phantoms(self):
        self.assertLess(self.shipped["phantom"], score(specialize=0.0)["phantom"])

    def test_refuter_merges_remove_duplicates(self):
        self.assertLess(self.shipped["duplicates"], score(B(merge_rate=0.0, same_as_rate=0.5))["duplicates"])
        self.assertEqual(self.shipped["misplaced"], 0)

    def test_partial_repeat_weight_is_better_calibrated_than_full(self):
        saved = config.LEARN["repeat_weight"]
        try:
            config.LEARN["repeat_weight"] = 1.0
            full = sim.score(WORLDS, 300)["calibration"]
        finally:
            config.LEARN["repeat_weight"] = saved
        self.assertLess(self.shipped["calibration"], full)

    def test_a_diff_empty_speaks_only_for_its_share(self):
        """Half the reviews see a quarter of the code. Counted as whole reviews, their empties hide real directions;
        dropped, invented names are never pushed dark. Counted for their share, neither happens."""
        diffs = {k: score(B(diff_share=0.5, diff_empties=k)) for k in ("scaled", "full", "drop")}
        hidden = {k: v["missed"] + v["unborn"] for k, v in diffs.items()}
        self.assertLess(hidden["scaled"], hidden["full"])
        self.assertLess(diffs["scaled"]["phantom"], diffs["drop"]["phantom"])

    def test_audit_is_a_meter_not_a_cost(self):
        self.assertLess(self.shipped["audits"], 0.25)

    def test_hostile_parents_are_survived(self):
        hostile = score(B(parents="hostile"))
        self.assertLessEqual(hostile["unborn"], 1.5 * self.shipped["unborn"] + 3)
        self.assertLess(hostile["missed"], 1.5 * self.shipped["missed"] + 5)


if __name__ == "__main__":
    unittest.main()
