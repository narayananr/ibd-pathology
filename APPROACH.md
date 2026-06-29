# APPROACH.md — the whole pipeline, explained end to end

> Plain-language walkthrough of *the approach*, start to finish — from a raw slide sitting on
> a scanner to the final coloured map and an honest accuracy number. Written 2026-06-28 as the
> companion narrative to the slide deck (`slides/index.html`, Slides 11–12) and the per-slide
> deep-dive (`slides/NOTES.md`). Project rules live in `CLAUDE.md`; the ordered to-do in
> `BUILD_PLAN.md`; "where are we right now" in `PROGRESS.md`.
>
> **How to read this:** every term is defined the first time it appears. One example slide,
> `107_HE`, is carried through the whole thing so it stays concrete.

**The goal, in one line:** turn a biopsy slide into a picture where *inflamed* tissue glows red
and *healed* tissue stays cool — plus a trustworthy number for "how often is the model right?"

---

## Stage 0 — What we start with, and the one label we have

A **whole-slide image (WSI)** is the biopsy scanned at microscope resolution. It is *gigapixel* —
like a photo billions of pixels wide, 1–4 GB on disk. No model can swallow that whole; it is far
too big.

The **only label** attached to `107_HE` is **one word**: `active` or `inactive`. "Active" means a
pathologist found neutrophils (immune cells) invading the gut lining *somewhere* on the slide.
That is it — one word for the entire slide.

> ⚠️ **The "label gap"** is *only* about this one word. The dataset already decided active-vs-inactive
> years ago; it just stored the word inside the WSI **filename** (`ID-X_Y.ndpi`). The pre-tiled patch
> files we downloaded kept only `107_HE` and dropped the word. So "closing the gap" = a spreadsheet
> lookup to recover one word per slide. **No tiles, no drawing, no annotation.**

So our raw material is: **one giant image + one word.** Everything below is the bridge from that to
a coloured map.

---

## Stage 1 — Cut the slide into tiles (TRIDENT does this)

Because the slide is too big, we chop it into thousands of small squares called **tiles** (or
*patches*) — think 256×256 pixels each. `107_HE` might become **~1,000 tiles**.

**TRIDENT** (an off-the-shelf tool from a Harvard lab) does this automatically in three moves:

1. **Tissue segmentation** — find the actual tissue and ignore the blank glass that fills most of a
   slide, so we never waste effort on empty space.
2. **Tiling** — lay a grid over the tissue and cut the tiles, *remembering each tile's (x, y)
   position* on the slide (the **coordinates**, or `coords` — we need them later to paint the map
   back in the right spots).
3. (Stage 2) hand each tile to the foundation model.

We use TRIDENT instead of writing our own tiler because WSIs are a minefield of formats and
coordinate systems, and it lets us swap foundation models with one flag.

**After Stage 1:** `107_HE` = ~1,000 little images, each tagged with where it sits on the slide.

---

## Stage 2 — Turn each tile into numbers (the *frozen* foundation model)

A computer cannot reason about a picture directly; it needs numbers. So we pass each tile through a
**foundation model** — a large neural network that was *already* trained (by someone else, on
millions of pathology images) to look at a tile and summarise it.

Its output for one tile is an **embedding**: a list of about **1,500 numbers** describing what the
tile looks like — the tile's "fingerprint," or its coordinates in a vast space where look-alike
tiles land near each other. Inflamed tiles cluster in one neighbourhood, calm tiles in another,
*even though the model was never told anything about IBD.*

**"Frozen"** is the key word: we do **not** train or change this model at all. We use it as a fixed
translator from picture → numbers. That is what makes this cheap and runnable without a big GPU —
the expensive learning was already done by someone else, and we **cache** the numbers so we never
redo it.

**After Stage 2:** `107_HE` = ~1,000 embeddings (rows of ~1,500 numbers) + the one word
`active`/`inactive`. *This pile of numbers is the raw material for every model below.* Stages 1–2
are solved by tools; from here on is the small part we own.

---

## Stage 3 — The modelling, in three steps of rising sophistication

The actual question: given ~1,000 tile-embeddings and one word, how do we score the tiles? We build
up slowly.

> 🔑 **Two kinds of label — keep them straight:**
> - **Slide label** — one word per slide (`active`/`inactive`). **The only label we ever need.**
> - **Tile label** — saying which *individual* tiles are inflamed. **We never make these by hand.**
>
> Both steps below run on the slide label alone. The whole reason for the second step (MIL) is so we
> *never* have to label tiles.

### Step 3a — Mean-pool baseline (the cheap "is there any signal?" check)

**Mean-pool** = average all ~1,000 embeddings into a *single* vector — the "average tile" of the
slide. Now each slide is just one vector + one word.

Feed those to **logistic regression** — the simplest classifier; it learns a weighted line that best
separates "active" vectors from "inactive" ones and outputs a probability.

We score it with **AUROC** (Area Under the ROC Curve): one number from 0.5 (useless coin-flip) to
1.0 (perfect) for how well the two classes separate. If AUROC is **≳ 0.70**, the embeddings genuinely
carry the inflammation signal and we proceed. If it is near 0.5, something upstream is broken and we
fix *that* first.

Its weakness: averaging **dilutes** a focal hotspot (a few inflamed tiles get drowned out by the
calm majority), and it gives **no map**. That is fine — it is a fast go/no-go *gate*, not the product.

### Step 3b — Attention-MIL (the real deliverable: the map)

Now we stop averaging. **Attention-MIL** (Multiple-Instance Learning) keeps all ~1,000 tiles separate
and learns a **weight** for each one — *how much this tile should count toward the slide's verdict.*
Then it takes a **weighted** vote: suspicious tiles count a lot, boring calm tiles count ~zero.

It trains on **only the one word per slide** — no tile labels, ever. The trick: the only way the
model can correctly call a focal slide `active` is to learn to put its weight on the few genuinely
inflamed tiles. So it *teaches itself* which tiles matter.

The payoff: **those learned weights are the map.** The tiles the model leaned on are exactly the
inflamed spots — localization for free, from slide-level labels only.

> **Analogy.** Imagine a box of 1,000 photos, and you are told only *"this box contains at least one
> fire-drill photo"* — but **not** which ones. Attention-MIL is a method that, from just that
> box-level statement, teaches itself to spot which photos are the fire-drill ones. Hand it 1,000
> tiles + the single word "active," and it figures out on its own which tiles earned that word. This
> matches the data's own definition — "active if ≥1 spot is inflamed" — which is precisely what MIL is
> built for. We sanity-check our numbers against Owkin's published **IMILIA** on this same dataset.

### Step 3c — QuPath polygons (optional, last, take-it-or-leave-it)

*Optionally*, on a handful of slides, you hand-draw a few "this region is inflamed / this one is
healed" outlines in **QuPath** (a slide-viewer). That gives a small set of *clean, human-verified*
tile labels — a yardstick to check the heatmap against. This is the **only** place a human marks
anything, it is optional, and it is what the existing `ibd_inflamed_probe.py` already does.

---

## Stage 4 — Stitch the heatmap (the visible deliverable)

Take the slide **thumbnail** (a small overview image). For each tile we now have a score — P(active)
between 0 and 1. Using the (x, y) coordinates saved back in Stage 1, we **paint each tile's spot**
with a colour: cool blue/green = healed → hot red = inflamed, laid semi-transparently over the
thumbnail.

Result: quiet tissue tints cool, inflamed hotspots glow red. A pathologist can glance at it and
check the model is firing on *real* cryptitis/abscesses — not on folds, blur, or slide edges. That
trust is the entire point.

---

## Stage 5 — Honest validation (the number we can defend)

A heatmap that merely *looks* right proves nothing. So we measure honestly with **leave-slides-out**
validation:

- Split slides into a training group and a held-out group, **keeping all of a slide's tiles together
  on one side.** Train on the training slides, test on slides the model has *never seen*.
- **Why never split by tile:** two tiles from the same slide are near-twins (same patient, stain,
  scanner). If twins land on both sides of the split, the model "recognizes" rather than
  "understands," and the score is fake-high. Splitting by slide is non-negotiable — it is the
  difference between an honest number and a lie.
- Report **AUROC** plus **agreement** (e.g. *kappa* — how often the model and a pathologist agree,
  beyond chance), and write down *where it fails* (focal disease it missed, artifacts it tripped on).

---

## The whole journey at a glance

```
ONE giant slide  +  ONE word (active/inactive)
        │
   [TRIDENT]  segment tissue → cut ~1,000 tiles (+ remember positions)
        │
   [frozen foundation model]  each tile → ~1,500 numbers (embedding)
        │
   ┌──────────────────────────── the part we build ────────────────────────────┐
   │  3a  mean-pool → logistic regression   →  "is the signal even there?" (AUROC)│
   │  3b  attention-MIL (slide labels only) →  weights = which tiles are inflamed │
   │  4   paint tile scores onto thumbnail  →  the red/blue heatmap              │
   │  5   leave-slides-out test             →  honest AUROC + agreement          │
   └────────────────────────────────────────────────────────────────────────────┘
```

Stages 1–2 are off-the-shelf and **cached once**. Only Stages 3–5 are ours, and they are small,
cheap, and CPU-friendly.

---

## Where we are on this journey (2026-06-28)

- ✅ **Environment** ready; ✅ **data downloaded** (the pre-tiled H&E patch set, MD5-verified).
- ⬜ **Nothing modelled yet.** The next clicks are the boring, safe bookkeeping ones — *before* any AI:
  - **Step 1** (`scripts/01_build_manifest.py`): build a plain table of every slide in the data
    (the `slide_id` column). ← the immediate next action
  - **Step 2** (`scripts/02_attach_labels.py`): fill in the one-word label per slide (close the
    "label gap").
  - **Step 3 onward:** embed → mean-pool check → attention-MIL → heatmap → validation.

We are at the very start of the conveyor belt, and we climb it one short script at a time.
