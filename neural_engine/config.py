"""Every tunable number, in one place. Field knowledge lives in the pack, never here."""

SEV = {"critical": 1.0, "major": 0.6, "minor": 0.3, "info": 0.1}
FINDING = ("critical", "major", "minor")  # info teaches nothing and is learned as empty

PRIOR = {
    "seed_bias": -1.0,  # a seed starts at p ~ 0.27 everywhere
    "root_bias": -2.0,  # a newborn with no parent starts unlikely but uncertain
    "inherit_q": 0.25,  # a newborn holds its parents' belief four times less surely
    "specialize": -1.0,  # ...and starts rarer: a child names a narrower failure than its parents
    "bias_q": 2.0,  # prior precision of a direction's rate: how many reviews' worth the prior is
}

# The refuter's verdict on a hit becomes its soft label.
CHECK = {"reproduced": 1.0, "plausible": 0.5, "refuted": 0.0}

LEARN = {
    "forget": 0.985,  # per review, precision relaxes toward its prior; trades noise against adaptation (README)
    "value_prior": 0.5,  # mean severity assumed before any finding (a seed row may override)
    "value_k": 2.0,  # ...worth this many findings
    "repeat_weight": 0.25,  # a finding already learned from (same repo, direction, line) is old evidence
    "fanout_weight": 0.5,  # a fan-out is selected on success, so it is trusted half as much as a planned walk
    "fixed_weight": 0.5,  # a human saying "fixed" confirms the finding
    "wrong_weight": 1.0,  # a human saying "wrong" counts as a full empty observation
    "delete_walks": 8,  # a newborn walked this often, and still dark, is deleted...
    "delete_z": 0.5,  # ...even this many standard deviations optimistic: deletes junk in ~16 walks, keeps 1-in-5 payers
    "window": 50,  # reviews in the rolling health metrics
}

REVIEW = {
    "walk": 10,  # most directions assigned per review
    "fanout": 12,  # findings recorded outside the plan
    "births": 3,  # new names per review
    "parents": 2,  # most parents a newborn may declare
    "max_live": 90,  # node budget: room to grow well past the seed, small enough to plan from
    "dark": 0.06,  # radius below which a direction is dark: not assigned unless a draw lifts it
    "audit_share": 0.2,  # reviews whose plan audits one empty walk: a meter of glances, not a learner (README)
    "span": 15,  # lines either side of a hit that the refuter is shown; it judges from these, not the whole tree
    "min_part": 0.05,  # a diff review counts as at least this share of the project
    "recipe_line": 300,  # most characters on one line of a recipe
    "recipe_overlap": 0.7,  # token-set Jaccard at which two recipe rules are the same rule
}
