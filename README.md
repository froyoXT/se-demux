# se-demux

A **neural engine** that learns where a model should look, and a skill that uses it for code review.

The engine is field-agnostic. It holds no guideline text: only named directions and numbers. A **pack** seeds those directions. Roots are universal domains; a child narrows one or two of them and may carry a four-line recipe, and one pack may `extends` another ([CONSTITUTION.md](CONSTITUTION.md)). The network grows from there. Each direction learns only from its own observations; a newborn copies its parents' belief once, and nothing flows back. Like NEAT, the topology evolves: directions are born, merged and deleted as verified results come in. The shipped pack is software-engineering review; another field is another pack.

## The network

Each direction is one logistic unit with a Gaussian weight:

```
p_i = σ(b_i)          how often walking this direction finds something
r_i = p_i · v_i       radius: that chance × the mean severity of what it finds
```

Nothing is specific to a project. Code fails in the same ways everywhere, so every review, of any project, teaches the one network. Each observation is one weighted Bayesian logistic step (Chapelle & Li, 2011):

```
m_j ← m_j + w (y − σ(s)) x_j / q_j        error × input / precision
q_j ← q_j + w x_j² σ(s)(1 − σ(s))         certainty accumulates; it relaxes 1.5% by the review's share
```

| Signal | Trust `w` |
|---|---|
| planned walk | 1 |
| fan-out finding | 0.5 |
| repeat (same direction, file and line) | × 0.25 |
| empty walk in a diff review | × the diff's share of the bytes, at least 5% |
| human "wrong" / "fixed" | 1 / 0.5 |

**Topology.**
- **Birth:** a finding no direction names becomes a node under one or two parents. It inherits their belief at ¼ certainty and starts rarer.
- **Merge:** the refuter folds duplicates, and the survivor takes over their children.
- **Delete:** a newborn still dark after 8–24 looks is deleted; an empty walk in a diff counts as the diff's share. Above 90 nodes, a birth replaces the least certain dark newborn.

## The shipped seed

Eight broad domains of how code fails: `logic`, `error-handling`, `security`, `data-integrity`, `concurrency`, `resource-lifetime`, `performance`, `maintainability`. Each is a root with a prior severity and no recipe. A pack that extends this one may add children and recipes ([CONSTITUTION.md](CONSTITUTION.md)). Everything specific is left to births.

## One review

```
plan → look + fan out → verify (one blind refuter, from excerpts) → apply → later: feedback
```

- **plan** draws a Thompson sample per direction and assigns up to 10 above the dark line, shuffled, with no numbers shown.
  - If exactly these bytes were already reviewed under this pack, wherever they sit, it stops: nothing is read, and nothing is learned twice.
  - `--changed` makes it a diff review of only those files.
- **look:** every walk is `hit` or `empty`, anchored by a site and a quote checked against the file. A direction shown with a recipe is checked against that recipe.
- **verify:** a fresh refuter sees each finding with 15 lines either side, and no direction names. On one review in five it also searches behind one empty walk, as a check on glances.
- **apply** learns from the verdicts; `feedback` adds the human's.

## Evidence

`neural_engine/sim.py` simulates code that fails the same ways everywhere: one true rate per direction, varied a little per project. The reviewer misses, invents, uses synonyms and files specific defects under broad names. Twelve specific directions must be discovered, and names invented that never pay must be deleted. `tests/test_learning.py` holds each mechanism against its absence over 100 worlds of 300 reviews. Fewer worlds are too noisy to decide anything.

| 100 worlds | missed | invented names lit | unborn | duplicates |
|---|---|---|---|---|
| **shipped**, 300 reviews | **0** | **101** | **1** | **0** |
| no specialisation offset for newborns | 0 | 155 | 0 | 0 |
| repeats learned at full weight | 0 | 204 | 1 | 0 |
| no refuter merges | 0 | 98 | 1 | 137 |
| shipped, 600 reviews | 0 | 59 | 0 | 0 |

Each world has 3 invented names that never pay. Reviewers keep reinventing them, so about one in three is alive and lit at any moment, fewer with experience.

What the simulator can't show is whether steering pays on real code. One fixture, a 13-file ledger service with 16 known defects and 7 decoys, was reviewed once by an independent frontier model with the skill and once without, on a fresh network:

| one fixture, one review | without the skill | with the skill |
|---|---|---|
| known defects found | 14/16 | 11/16 |
| criticals found | 5/5 | 5/5 |
| decoys reported | 0 | 0 |
| tokens | 1.4M | 4.1M |

On a first review every seed is walked, so this measures the skill's cost, not its steering. The payload, the refuter and the apply roughly tripled the tokens, and the refuter dropped one real finding. A follow-up study found that, on a strong reviewer, directions don't go dark, so steering is unlikely to repay that cost; it proposes a redesign and a protocol to settle it ([RESEARCH.md](RESEARCH.md)).

## Use

```sh
uv run neural-engine [--home <dir>] [--pack <folder>] plan --root <project> [--changed changed.txt]
uv run neural-engine verify --from payload.json
uv run neural-engine apply --from payload.json [--dry-run]
uv run neural-engine feedback <id> fixed
uv run neural-engine feedback <id> wrong
uv run neural-engine show
uv run neural-engine check
uv run neural-engine repack
uv run neural-engine schema payload
uv run python -m neural_engine.sim
uv run python -m unittest discover -s tests -t .
```

`--pack <folder>` selects a pack. If it is omitted, `$NEURAL_ENGINE_PACK` does, and otherwise the shipped pack is used. `--home` is the root of the pack homes.

**Where the learning lives.** The pack is only the seed. Everything the network learns, its weights and every direction born, merged or deleted, lives in that pack's home under the root `~/.neural-engine/<pack name>/` (or `$NEURAL_ENGINE_HOME`, or `--home`): `log.jsonl` holds every review, and replaying it rebuilds the network; `state.json` is that network, for reading. One home per pack serves every project you review with that pack. The log stores no free text, only names, verdicts, severities and salted hashes. A changed pack stops planning until `repack` records it.
