"""One direction: a logistic unit with a Gaussian weight.

    p = sigma(b)                    how often walking this direction finds something
    radius = p x value              value = mean severity of what it finds

Nothing is specific to a project: code fails in the same ways everywhere, so every
review teaches the one network. The structure that matters is the topology: which
directions exist, and which parents a newborn inherits its belief from.

Each weight is a Gaussian (mean m, precision q), updated online by the Laplace
step of Chapelle & Li (2011). The update's error term (y - p) is
backpropagation through one sigmoid. Labels are soft (y in [0, 1]) and every
observation carries a weight, so a half-trusted signal moves the graph half as far.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import LEARN, PRIOR

BIAS = "*"


def sigma(s: float) -> float:
    return 1.0 / (1.0 + math.exp(-s)) if s >= 0 else math.exp(s) / (1.0 + math.exp(s))


def features() -> dict[str, float]:
    """The input every review presents: the unit's bias. The engine knows nothing else about the code."""
    return {BIAS: 1.0}


def base_q(k: str) -> float:
    return PRIOR["bias_q"]


@dataclass
class Direction:
    name: str
    origin: str  # seed | birth
    born: int = 0
    parents: list[str] = field(default_factory=list)  # at most REVIEW["parents"]; inheritance happens once, at birth
    value_prior: float = LEARN["value_prior"]
    prior: dict[str, float] = field(default_factory=dict)  # prior means
    prior_q: dict[str, float] = field(default_factory=dict)  # prior precisions where not the base
    m: dict[str, float] = field(default_factory=dict)
    q: dict[str, float] = field(default_factory=dict)
    hits: float = 0.0  # sum of trusted soft labels
    sev_sum: float = 0.0  # sum of label x severity
    walks: int = 0  # planned walks
    looks: float = 0.0  # planned walks, an empty in a diff counting for its part: what infancy is judged on
    aliases: list[str] = field(default_factory=list)
    deleted: bool = False

    # --- reading ---------------------------------------------------------

    def weight(self, k: str) -> tuple[float, float]:
        q0 = self.prior_q.get(k, base_q(k))
        return self.m.get(k, self.prior.get(k, 0.0)), self.q.get(k, q0)

    def logit(self, x: dict[str, float]) -> tuple[float, float]:
        mu = var = 0.0
        for k, v in x.items():
            m, q = self.weight(k)
            mu += m * v
            var += v * v / q
        return mu, var

    def p(self, x: dict[str, float]) -> float:
        """What the graph believes. Uncertainty never lights a direction; it only explores it, via sample()."""
        return sigma(self.logit(x)[0])

    @property
    def parent(self) -> str | None:
        return self.parents[0] if self.parents else None

    def value(self) -> float:
        return (self.sev_sum + self.value_prior * LEARN["value_k"]) / (self.hits + LEARN["value_k"])

    def radius(self, x: dict[str, float]) -> float:
        return self.p(x) * self.value()

    def sample(self, x: dict[str, float], rng: random.Random) -> float:
        """Thompson draw: uncertain directions sometimes sample high, and so get explored."""
        s = 0.0
        for k, v in x.items():
            m, q = self.weight(k)
            s += (m + rng.gauss(0, 1) / math.sqrt(q)) * v
        return sigma(s) * self.value()

    # --- learning --------------------------------------------------------

    def learn(self, x: dict[str, float], y: float, sev_value: float, w: float = 1.0) -> None:
        """One observation of soft label y with trust w. Severity feeds value only in proportion to y."""
        self._update(x, y, w)
        self.hits += w * y
        self.sev_sum += w * y * sev_value

    def _update(self, x: dict[str, float], y: float, w: float) -> None:
        """Weighted MAP step for one logistic observation under a diagonal Gaussian prior."""
        cur = {k: self.weight(k) for k in x}
        s0 = sum(m * x[k] for k, (m, _) in cur.items())
        spread = sum(x[k] ** 2 / q for k, (_, q) in cur.items())
        s = s0
        for _ in range(20):  # Newton on the 1-D logit: s = s0 + w (y - sigma(s)) spread
            p = sigma(s)
            s -= (s - s0 - w * (y - p) * spread) / (1 + w * p * (1 - p) * spread)
        p = sigma(s)
        for k, (m, q) in cur.items():
            self.m[k] = m + w * (y - p) * x[k] / q
            self.q[k] = q + w * x[k] ** 2 * p * (1 - p)

    def absorb(self, other: Direction) -> None:
        """Merge a synonym in: precision-weighted means, the surer precision, summed history; it becomes an alias."""
        for k in set(self.m) | set(other.m):
            sm, sq = self.weight(k)
            om, oq = other.weight(k)
            self.m[k] = (sm * sq + om * oq) / (sq + oq)
            self.q[k] = max(sq, oq)
        self.hits += other.hits
        self.sev_sum += other.sev_sum
        self.walks += other.walks
        self.aliases += [other.name, *other.aliases]
        other.deleted = True

    def forget(self, part: float = 1.0) -> None:
        """Precision relaxes toward its prior per unit of code examined: a review of a tenth of a project, a tenth."""
        g = LEARN["forget"] ** part
        for k, q in self.q.items():
            q0 = self.prior_q.get(k, base_q(k))
            self.q[k] = q0 + g * (q - q0)


def born(name: str, parents: list[Direction], review: int) -> Direction:
    """A newborn inherits its parents' precision-weighted belief, held four times less surely.

    Inheritance happens once, here. Nothing flows between parent and child afterwards, so a
    bad parent costs a slow start, never a feedback loop.
    """
    if not parents:
        return Direction(name, "birth", born=review, prior={BIAS: PRIOR["root_bias"]})
    keys = {BIAS}.union(*(set(p.prior) | set(p.m) for p in parents))
    prior, prior_q = {}, {}
    for k in keys:
        ws = [p.weight(k) for p in parents]
        total = sum(q for _, q in ws)
        prior[k] = sum(m * q for m, q in ws) / total
        prior_q[k] = total / len(ws) * PRIOR["inherit_q"]
    prior[BIAS] += PRIOR["specialize"]
    value = sum(p.value() for p in parents) / len(parents)
    return Direction(
        name,
        "birth",
        born=review,
        parents=[p.name for p in parents],
        prior=prior,
        prior_q=prior_q,
        value_prior=value,
    )
