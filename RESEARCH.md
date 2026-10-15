# Research: does steering pay? (2026-10-15)

This is the handoff after se-demux's first real-code evaluation. It records what was measured, what three independent research passes found, the redesign they converge on, and how to settle the question. **No code has changed since the evaluation. The open decision is at the end.**

## 1. The result

One hand-built fixture, a 13-file Python/SQLite ledger service with 16 known defects and 7 decoys, was reviewed once by an independent frontier model (grok-4.7, high effort) with the skill and once without, both on a fresh network.

| one fixture, one review | without the skill | with the skill |
|---|---|---|
| known defects found (strict / grader) | 14/16 / 15/16 | 11/16 / 12/16 |
| criticals | 5/5 | 5/5 |
| decoys reported | 0 | 0 |
| tokens | 1.39M | 4.12M |
| model calls | 13 | 47 |
| billed cost | 1.0× | **2.0×** |
| wall clock | 29 min | 32 min |

- **Tokens and cost tell different stories.** Most input was cached re-reads (85% without the skill, 92% with it), billed at a quarter of the uncached rate. So the bill doubled while the tokens tripled. Uncached input rose 1.57× and output 1.12×. The extra money is extra turns.
- **The grader is loose by one in each arm.** It matches path and line ±2 and ignores the summary. In both arms, a finding about tied-timestamp balances was credited as the quadratic statement query, so the strict counts are one lower. The gap between the arms is 3 either way.
- **Significance:** McNemar on 3 discordant defects against 0 gives exact two-sided p = 0.25. One fixture cannot decide this; the fixture was also a design input.

### Where the cost went (guided run)

The per-step shares are estimated from per-call prompt sizes; the refuter's share and the call counts are measured.

| step | calls | share of cost |
|---|---|---|
| read skill, docs and plan (plan ran twice) | 5 | 5% |
| read the fixture | 2 | 2% |
| read engine schema, guard and config | 3 | 6% |
| reproduce by running code | 10 | 20% |
| payload, quote fixes, `verify` | 9 | 23% |
| spawn the refuter and wait | 3 | 9% |
| `apply`, then clean up a stray `__pycache__` | 4 | 11% |
| final report | 1 | 2% |
| **refuter** (its own session) | 10 | **22%** |

Reading and reproducing look like the unguided run's 13 calls. The other 34 calls are the skill's ceremony.

### Where the four guided misses were lost

| defect | the guided run… |
|---|---|
| regex backtracking DoS in the memo check | read the file and never wrote a note about it. Performance's one hit was a blocking HTTP call elsewhere. |
| request signature omits method and path | read the file and filed two other defects in it, but never mentioned the signature. |
| payout across recipients is not atomic | found it and reproduced the partial batch. The refuter then marked the finding `refuted` because one sentence ("a retry pays again") was wrong for the amounts chosen, and the skill says a refuted finding is dropped. |
| idempotency cache grows without bound | found it, but filed it at the dict's declaration under a restart framing. The unguided run missed it entirely. |

**Correction to an earlier hypothesis.** The claim that "one verdict per walk made the model satisfice" is only half true. Several directions also carried extra findings under the same name. What was saturated was the recording ceiling:
- 8 of 8 walks hit;
- 12 of 12 fan-out slots were used;
- 3 of 3 births;
- 0 empty walks.

The misses sat inside directions that had already paid once.

### What the network learned from it

- `apply` keeps only the strongest observation per direction. 19 reported findings became 11 updates, all positive.
- Nothing went dark, and every belief moved up.
- A corpus where every project has a bug in every root teaches the network to walk every root forever.

## 2. Findings from three research passes

Three independent research agents (the same frontier model, separate clones, no answer key, no reviews run) each took one angle: field evidence, mechanism, and forensics. They agreed on the following.

### The skip premise fails on the engine's own dynamics (measured in simulation)

- A direction's hit bit turns on at the first hit and stays on for a strong reviewer.
- At a true hit rate of 0.20 over 300 reviews, **no seed ever went dark**, and late reviews walked more seeds than early ones (7.84 against 7.38, over 20 worlds).
- After one critical on every seed, a hit rate of 0.9 still walks 7.999 of 8.
- Even after 40 forced empties, Thompson sampling still walks security about 40% of the time.

The promised savings, fewer walks once directions go dark, would almost never arrive.

### The field evidence points the same way

These figures are as cited by the research agent and have not been independently re-checked. Treat them as leads.

- **Checklists and category procedures** did not beat ad hoc reading in inspection studies (Porter, Votta and Basili, 1995, on requirements documents, and its replication). Scenario- and perspective-based reading did better.
  - For LLM review, naming categories gives a small lift, and adding procedure on top of that cuts it.
  - Long guidance did worse than a bare category name.
- **Critics without an oracle** reject true answers.
  - Intrinsic self-correction lowered accuracy (Huang et al., ICLR 2024).
  - Majority voting and reward models fail to find the rare correct sample that plain coverage does (Brown et al., "Large Language Monkeys").
  - Systems that ship drop a finding only on an execution check: a crash, a test, or a sandbox (Project Zero's Naptime; OpenAI's Aardvark).
- **The union of independent passes scales; voting doesn't.** Coverage rises roughly log-linearly with samples. A second independent unguided pass is where the last one or two defects live.
- **Learned "where to look" signals.**
  - Change-level defect prediction works modestly inside one project (Kamei et al., 2013) and rarely transfers across projects.
  - Cross-project prediction largely fails (Zimmermann et al., 2009).
  - Fuzzing schedulers give *more* energy to paths that haven't paid off and never drop them (AFLFast, 2016), the opposite of darkening.
  - The signal that works best, change-level risk inside one project, is the per-project identity the constitution rules out.

### Verdict

**The learned map of what to skip should be shelved.** On a strong reviewer the evidence says it won't save anything and may hide rare, severe defects. What isn't ruled out is a cheap way to *add* recall.

## 3. The redesign that survives: learn the reviewer's blind spots

Two independent proposals converged on this design: "residual hint" from the research, and "credit only marginal recall" from the author's own notes.

1. **Open review first, unchanged.** The reviewer reviews as if the engine weren't there and finishes the draft list. On a cold network the engine assigns nothing. This keeps recall at least at the unguided level.
2. **`hint` after the draft.** It prints at most three directions that the draft didn't tag but that history says tend to add findings. The reviewer makes one short pass on those only and appends what it finds.
3. **No ceremony on the hot path.**
   - No quote check, and no `verify` step.
   - Any verifier may keep, repair or downgrade a finding. It drops one only when running the code shows the claimed outcome doesn't happen.
4. **Learn only the margin.** `apply` takes tagged notes: `{node, n, sev}`, plus births.
   - A hinted direction that adds a finding is a hit.
   - A hinted direction searched and empty is an empty.
   - Directions the draft covered teach nothing new.
   - A lit direction therefore means a real blind spot, and a dark one means the open read already covers it.
5. **Optional, later:** store a finding count per direction, and fire one extra pass only when this review falls far below that direction's own history. Leave this out until the hint shows value.

The promise changes from "cheaper later reviews" to **more recall at no more than 1.3× the unguided cost.** The engine stays field-agnostic and the pack stays data. Nothing in [CONSTITUTION.md](CONSTITUTION.md) changes.

### Engine changes, sketched

| file | change |
|---|---|
| `review.plan` | Seals the tree; assigns nothing; prints "review first". |
| `review.hint` (new) | Takes the draft's tags and returns at most 3 lit directions not among them. |
| `schema.Payload` | Becomes notes plus births; `site`, `quote`, `check` and `audit` are removed. |
| `guard` | Drops the anchor and quote checks. |
| `review.apply` | Learns once per note; a hinted empty is y = 0. |
| `review.verify` | Removed, or kept only as an optional execution-backed check. |
| `SKILL.md` | Rewritten to the six steps above. |
| `README.md` | Updated in the same change. |

The simulator and `tests/test_learning.py` must be adapted so that a hit means "added beyond the open read".

## 4. How to settle it

**Stop deciding on the tally fixture.** Keep it as a frozen smoke test only.

**Decision set.**
- About 20 real fix commits from after the reviewing model's knowledge cutoff, across Python, Java and TypeScript. Each one is reviewed as the parent's diff, and the oracle is the lines the fix changed plus the tests that went from failing to passing.
- Refactor, rename and dependency-bump commits from the same repos serve as decoys.
- Rebuild GitBug-Java's method (reproducible, recent commits) rather than using Defects4J, BugsInPy or SWE-bench Verified, which are likely in training data or have known flaws.

**Arms.**
1. Unguided.
2. The current skill.
3. The thin residual-hint skill.
4. **Two unguided runs, unioned.** The skill costs 2×, so this is the fair same-budget competitor.

**Variance pilot first.** 2 fixtures × 4 paired repeats. Sample size is N = (2.80 σ / δ)² for a two-sided 5% test at 80% power.

| gap to detect | σ = 2 | σ = 3 |
|---|---|---|
| 3 defects | 4 fixtures | 8 fixtures |
| 2 defects | 8 fixtures | 18 fixtures |

If the pilot's σ is above 4, a 20-fixture study can't see a 2-defect effect: scope it to diffs or stop.

**Budget**, in units of this run (unguided about $5.50, guided about $10.90):

| stage | estimate |
|---|---|
| pilot | ~$260 |
| cold three arms on 20 diffs | ~$460 |
| a steering curve (4 homes × 15 diff reviews) | ~$740 |

The steering curve is only worth running if the thin skill passes.

**Pass and kill criteria, written before any new fixture is opened.**
- **Thin skill passes** if, on held-out diffs, it matches or beats unguided recall within one defect at no more than 1.3× billed cost.
- **Kill the current skill** if the mean gap (unguided − guided) is more than 1 defect at more than 1.5× cost, with a 90% interval that excludes 0.
- **Kill the project** if the thin skill also fails to beat unguided on recall per dollar, or loses to two unguided runs unioned at the same budget.

**Harness work** (in the git-ignored `eval/`, not part of this repository):
- record billed cost (`costUsdTicks`), cached and uncached input, output, and calls, with the parent and refuter separate;
- log cap saturation (fan-out used, births, empties);
- grade line-match and claim-match side by side;
- aggregate paired runs into mean gap, sd and interval;
- for a sequence, record walks assigned, dark count and dark-hides (a fix site inside a dark direction that nothing caught).

## 5. Open decision and notes for the next agent

**The owner hasn't yet chosen between:**
- **(a) Rebuild** around section 3, then run the variance pilot, and kill on the pre-registered bar; or
- **(b) Shelve** the project, with this document as the record.

Ask before starting either.

**Working rules carried over from the owner:**
- The engine is field-agnostic, like a CNN; a pack is its data.
- Recipes are optional, atomic, mutually exclusive, and never on universal roots.
- No linter or analyzer rules, and no per-project identity.
- Docs stay lean and are never stale: README, SKILL.md and this file change with the code.
- Implementation, tests and measurements are done by an independent frontier model on scratch clones. The lead writes the spec and reviews every diff before applying it.
- Commits contain no machine paths, secrets or local identifiers.

**What's not in the repository:** the evaluation fixture, its answer key, the run transcripts and the harness all live in the git-ignored `eval/` folder on the original machine. A fresh clone has none of them.

### Sources cited by the research (not independently re-checked)

- Porter, Votta, Basili, *Comparing detection methods for software requirements inspections*, TSE 1995: https://www.computer.org/csdl/journal/ts/1995/06/e0563/13rRUxC0SFB
- Replication: https://link.springer.com/article/10.1023/A:1009776104355
- Perspective-based reading: https://dl.acm.org/doi/10.1109/32.922713
- Chapelle & Li, *An empirical evaluation of Thompson sampling*, NeurIPS 2011: https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
- GitBug-Java, MSR 2024: https://arxiv.org/abs/2402.02961
- Multi-fault Defects4J/BugsInPy mining: https://arxiv.org/html/2403.19171
- Training-data overlap of bug benchmarks: https://arxiv.org/pdf/2411.13323
- OpenAI on SWE-bench Verified: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Also named without links: Huang et al. 2023 (arXiv:2310.01798); Brown et al. 2024 (arXiv:2407.21787); Yu et al. 2024 (arXiv:2401.16310); PrimeVul (arXiv:2403.18624); GPTLens (arXiv:2310.01152); Agentless (arXiv:2407.01489); CriticGPT (arXiv:2407.00215); Kamei et al., TSE 2013; Zimmermann et al., ESEC/FSE 2009; Böhme et al., AFLFast, CCS 2016; Project Zero's Naptime (2024) and Big Sleep (2024).
