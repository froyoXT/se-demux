# Seeding constitution

Rules for writing a pack. The loader enforces the ones it can check.

1. The engine is a network; a pack is its data. A field changes the pack, never the engine.
2. A seed names a direction: a way the field's work fails, a place to look. It is not a checklist item.
3. Roots are universal domains any capable reviewer already understands. A root never carries a recipe.
4. A rule a reviewer cannot know, such as a house or organisation guideline, is a child of the domain it narrows, with one or two parents. Only a child may carry a recipe. An organisation layer on an industry norm is a child.
5. A recipe holds only what the reviewer cannot know, in exactly four lines: `rule`, `wrong`, `right`, `cite`. It never says how to review, how likely a defect is, or how severe it is.
6. Every recipe is atomic (one rule; two rules are two directions), self-contained (it reads alone: no links, and no references to other recipes, directions or documents) and mutually exclusive (a defect fits at most one; overlap means merge or split). Recipe text lives only in the recipes folder, never in `seed.tsv`, the docs or the engine. The loader refuses a bad four-line shape, a link, another child's name, and a duplicate or overlapping rule.
7. Nothing a compiler, analyzer or linter already enforces belongs in a pack. Seed the part that needs judgement.
8. Seed few and broad. Specifics are born from reviews; do not seed what births will find.
9. `value` is the prior severity of what the direction finds: 0.7 high-impact, 0.5 default, 0.3 low.
10. One field, one pack, one home. `extends` builds on another pack. An override changes only a value; nothing is removed.
11. Changing a seed or a recipe changes the fingerprint. `repack` acknowledges it, and nothing learned is lost.
12. A name is 1–4 lowercase words joined by hyphens.

## Example

Northwind is a made-up organisation. Its pack extends the shipped one, lowers one root, and adds one house rule.

```
pack.toml
[pack]
name = "northwind"
version = "1"
extends = "software-engineering"

[files]
seed = "seed.tsv"
recipes = "recipes"

seed.tsv
logic	0.4
handoff-notes	0.5	logic

recipes/handoff-notes.md
rule: a handoff names the one person who receives the work
wrong: the group will pick it up later
right: one person receives the work
cite: the line that names who receives the work
```
