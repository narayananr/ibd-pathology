# IBD H&E — Inflamed vs Healed · Detailed Notes (companion to `index.html`)

This is the long-form companion to the slide deck (`slides/index.html`). Each section
below maps **1:1 to a slide** and explains it in more depth than fits on the slide — the
biology, the engineering, and *why* we made each choice. Read the slide for the gist; read
here for the full explanation.

> **Maintenance:** whenever we add a slide to `index.html`, we add a matching `## Slide N`
> section here. The deck and these notes stay in lockstep — together they are the project's
> running logbook. Convert any relative dates to absolute when writing.

**Legend used throughout:** <span>🌸</span> pink in H&E = cytoplasm / connective tissue /
secretions · 🟣 purple dots = cell nuclei. A dense cloud of dark dots = a crowd of cells =
(usually) inflammation.

---

## Slide 1 — Title

**What it is:** the cover. Names the project and the one-line architecture.

**Detail.** The project takes H&E whole-slide images (WSIs) of colonic biopsies from IBD
patients and produces a per-slide **inflamed-vs-healed region map**, plus an honest
validation number. The defining design choice — stated up front so every later decision
traces back to it — is that we build on a **frozen** pathology foundation model. "Frozen"
means we do not train or fine-tune the big image encoder at all; we only train a small model
on top of the numbers it produces. This keeps v1 small, cheap, and reproducible.

**Takeaway:** one frozen encoder, one small trainable head, built up slowly.

---

## Slide 2 — The Challenge

**What it is:** why this problem is non-trivial.

**Detail — the four hard parts.**

1. **Slides are enormous.** A single WSI (`.ndpi` format here) is a gigapixel image, often
   1–4 GB on disk. No neural network ingests that whole. The standard move is to chop the
   slide into thousands of small **tiles** (e.g. 256×256 pixels) and work tile-by-tile.

2. **Labels are slide-level, not regional.** Our dataset (IBDColEpi) tells us whether a
   *whole slide* is **active** (disease present) or **inactive** — it does **not** tell us
   which tiles are inflamed. "Active" specifically means a pathologist found intraepithelial
   granulocytes (neutrophils) *somewhere* on that slide.

3. **Disease is focal.** This is the crux. In an "active" slide, the inflamed tissue may be a
   small minority — most tiles still look perfectly healthy. So if we naively copy the
   slide's "active" label onto every tile, we mislabel the majority of tiles. This is called
   *label noise*, and it's why a naive approach underperforms.

4. **It leaks if you're careless.** Tiles cut from the same slide are highly correlated
   (same patient, same staining batch, same scanner). If you randomly split *tiles* into
   train/test, near-duplicate tiles land on both sides and the model "cheats," giving a
   flattering but fake score. **Fix: always split by slide (or patient), never by tile.**

**Takeaway:** giant images + weak (slide-level) + focal labels + leakage risk. Every design
choice on the next slides is a response to one of these.

---

## Slide 3 — Our Approach

**What it is:** the end-to-end pipeline, and which part is *ours*.

**Detail — the five stations.**

| Station | What happens | Who does it |
|---|---|---|
| 🧫 Whole slide | the raw `.ndpi` scan | input |
| ✂️ Tile it | find tissue (skip blank glass), cut into 256px tiles at 20× | **TRIDENT** (off-the-shelf) |
| 🧠 Embed tiles | run each tile through a pretrained pathology model → a vector of numbers ("embedding") | **frozen foundation model** (off-the-shelf) |
| 🎓 Light head | a small classifier reads the embeddings and scores inflamed vs healed | **we build this** |
| 🗺️ Stitch heatmap | paint each tile's score back onto a slide thumbnail | **we build this** |

**Why "frozen embeddings"?** A foundation model (e.g. H-optimus, UNI, Virchow) has already
seen millions of pathology images and learned a rich, general representation of tissue. An
embedding is its compact description of a tile. Training such a model needs huge data and
GPUs; *using* it as a fixed feature extractor needs neither. By freezing it we get most of
the benefit for a tiny fraction of the cost — and every future upgrade (better scoring,
cell analysis, etc.) reuses the *same* cached embeddings.

**Why "smallest useful slice" first?** We deliberately ship: embeddings → light head →
heatmap → honest validation. Things like stain normalization, cell/crypt segmentation, and
clinical indices (Geboes/Nancy/Robarts) are **deferred** — each is a later add-on onto the
same embeddings, not a prerequisite.

**Takeaway:** stations 1–3 are solved by tools; we own only the small, cheap end.

### Deep dive (added 2026-06-28) — TRIDENT, the encoder, the heatmap

**What exactly is TRIDENT?** A single command-line tool from the Mahmood Lab (Harvard) that
automates stations 1–3. Point it at a folder of WSIs and it (1) **segments tissue** —
finds the tissue and ignores the blank glass that fills most of a slide, so we don't embed
empty space; (2) **tiles/patches** the tissue into a regular grid (our recipe: 256×256 px at
20× magnification), recording each tile's (x, y) coordinate; (3) **extracts features** by
running every tile through a chosen foundation model. Output per slide: one `.h5` file with
`features` (N×D: N tiles, each a D-number vector) and `coords` (N×2, tile positions), plus a
thumbnail and a tissue-contour GeoJSON. We use it instead of hand-rolled OpenSlide tiling
because WSIs are tricky to handle (magnification levels, coordinate systems, `.ndpi` format,
tissue detection) and TRIDENT also lets us swap encoders with one flag.

**The foundation model, concretely.** A large Vision Transformer pretrained (self-supervised,
i.e. without labels) on millions of histopathology tiles — it has effectively "seen" a huge
amount of tissue and learned to represent it. Candidates in TRIDENT: **H-optimus-1 / H0-mini**
(open weights, our starting default), **UNI2 / Virchow2** (excellent but gated on HuggingFace).
We choose by linear-probe performance on *our* IBD active/inactive slides — not on oncology
leaderboards.

**What we do with the embeddings.** Each tile becomes an **embedding**: a vector of ~1,000–1,500
numbers — a "fingerprint" / coordinate in a high-dimensional tissue space where look-alike
tiles land near each other. Inflamed tiles cluster apart from calm tiles even though the model
was never taught about IBD. We **freeze** the model and treat embeddings as fixed inputs, then
train a tiny **head**: (v1a) mean-pool a slide's tile embeddings → logistic regression → active/
inactive (signal check); (v1b) attention-MIL → per-tile scores. The head has few parameters,
trains in seconds–minutes on CPU, and needs little data — which is exactly why this approach
suits CPU-only hardware.

**What the heatmap looks like, and why it's useful.** Take the slide thumbnail; for each tile
we have P(active) ∈ [0,1]; colour each tile by that value (cool blue/green = healed → hot red =
inflamed) and overlay it semi-transparently on the thumbnail. Quiet tissue tints cool, inflamed
hotspots glow red. (`ibd_inflamed_probe.py` already does this with a turbo colormap + colorbar.)
It's useful because it (1) **localizes** the focal disease — *where*, not just *whether*;
(2) **builds trust** — a pathologist can verify the model fires on real cryptitis/abscesses, not
artifacts; (3) **exposes failures** (folds, blur, edges); and (4) **seeds severity scoring**
later (how much active area → Nancy/Geboes).

---

## Slide 4 — The Biology We're Detecting

**What it is:** the single visual concept the whole model is chasing.

**Detail.** The clinical line between our two classes is **neutrophil activity in the
epithelium**:

- **Active / inflamed** = neutrophils (the body's first-responder white blood cells) have
  left the bloodstream and invaded the gut lining: *cryptitis* (in the crypt wall) →
  *crypt abscess* (in the crypt lumen) → *erosion/ulcer* (surface breakdown).
- **Inactive / healed (quiescent)** = no active neutrophil injury. Importantly, the tissue
  may still show **chronic** changes from past disease (distorted/branched crypts,
  scarring) — that's allowed. We classify by **activity**, not by whether the tissue is
  pristine.

**Reading H&E.** H&E = Hematoxylin (stains nuclei 🟣 blue/purple) + Eosin (stains
cytoplasm/proteins 🌸 pink). Because every cell's nucleus goes dark, *cell density reads as
darkness*: a quiet area is mostly pink with sparse dots; an inflamed area is peppered with
dark dots. Training your eye to read "dot density" is 80% of reading these images.

**Takeaway:** we are teaching a model to recognize "neutrophils where they shouldn't be."

---

## Slide 5 — Image A · Healed / inactive (normal colon)

**Image:** `images/inactive_normal_colon.jpg` — normal colonic mucosa (Wikimedia Commons, CC-BY-SA).

**What to look at.**
- **Architecture is orderly:** crypts (the mucus-making glands) look like **test tubes
  stood neatly in a rack** — long, straight, parallel, evenly spaced, all reaching down to
  the muscle layer (the pink band at the bottom).
- **Crypt cross-sections are clean rings** with pale centres (goblet cells full of mucus).
- The **lamina propria** (tissue between crypts) is relatively quiet — some scattered
  immune cells are *normal* here, but it isn't dense or pus-filled.

**Why it's our reference.** This is the "calm" template. Everything we call "active" is a
*departure* from this orderliness. For the model, tiles that look like this should score
**low** (healed).

---

## Slide 6 — Image B · Active (cryptitis, intermediate magnification)

**Image:** `images/active_cryptitis_intermed.jpg` (Wikimedia Commons, CC-BY-SA, Nephron).

**What to look at.**
- The tidy rack is **gone** — crypt architecture is irregular and distorted.
- The spaces between crypts are **stuffed with dark inflammatory nuclei** (high dot density).
- The **surface** (top edge) is ragged/damaged rather than smooth.

**Why it matters.** This is a "mid-zoom" view of an actively inflamed region — the overall
*texture* (busy, dark, disordered) is exactly the kind of signal a tile-level model can pick
up even without resolving individual neutrophils. Tiles like this should score **high**.

---

## Slide 7 — Image C · Crypt abscess (the hallmark)

**Image:** `images/active_crypt_abscess.jpg` (Wikimedia Commons, CC-BY-SA).

**What a crypt abscess is, built up step by step.**
1. A **crypt** is a tiny well/gland in the gut lining. Cut **lengthwise** it's a tube; cut
   **across** it's a **ring** (a donut): a circle of wall cells around a hollow centre
   (the lumen). This image is a crypt cut across.
2. A **neutrophil** is a first-responder white blood cell that normally stays in the blood.
   Its nucleus is multi-lobed, so a crowd of them looks like a messy cluster of small dark
   blobs; a pile of dead ones is **pus**.
3. **Progression:** healthy (no neutrophils in the lining) → **cryptitis** (neutrophils in
   the crypt *wall*) → **crypt abscess** (neutrophils spilled into the crypt *centre*).

**What to look at:** the **ring** is the crypt wall; the **centre is packed** with
neutrophils + pink debris — that filled centre *is* the abscess. In a healthy crypt that
centre would be an empty pale hole (compare Image A).

**Why we care most about this one.** The dataset's "active" label exists precisely because a
pathologist found features like this. It's the **loudest, cleanest** version of our target
signal, so it doubles as a **sanity check**: if our model can't fire on tiles like this,
something is broken and nothing downstream will work.

---

## Slide 8 — Image D · Cryptitis up close (high magnification)

**Image:** `images/active_cryptitis_high.jpg` (Wikimedia Commons, CC-BY-SA, Nephron).

**What to look at.**
- Neutrophils **squeezing into and between** the epithelial cells that line the crypt — this
  *infiltration of the lining* is the literal definition of cryptitis.
- The surface at the top is **eroded/frayed**.

**Why include both C and D.** C shows the *result* (pus in the lumen); D shows the *act*
(neutrophils breaching the epithelium). Together they define "neutrophils invading the
epithelium" at two stages — the exact thing our classes hinge on.

---

## Slides 8A–8C — How to read the histology (annotated + zoom) — added 2026-06-30

A self-contained **"learn to read the slide"** mini-section, six slides, kicker
*"How to read the histology · N / 3"*. Each of the three states (normal → cryptitis → crypt
abscess) gets a **labelled overview** then a **zoom** that names the actual cells. Built with
matplotlib leader-line callouts over the same Wikimedia/Nephron H&E images; colour code is
consistent across all six: **green = epithelium**, **red = active / neutrophil**,
**blue = lamina propria**, **purple = goblet**, **grey = other / structural**.

**Why it exists.** Before trusting a model that calls tissue "active", you should be able to
read the feature yourself. The section teaches the eye the exact cue the labels hinge on — a
neutrophil's multi-lobed nucleus crowding the epithelium — by walking from the calm baseline
to the loudest active lesion.

- **8A — Normal mucosa, your baseline** (`histo_normal_annotated.png`): orderly crypts, pale
  goblet cells, a quiet lamina propria with no neutrophils. The "healed" template.
- **8A-ZOOM — into a single crypt** (`histo_normal_zoom_annotated.png`): crypt = one gland;
  tidy basal epithelial nuclei; abundant goblet vacuoles; lumen; only a few lymphocytes /
  plasma cells in the lamina propria; muscularis mucosae band.
- **8B — Cryptitis** (`histo_cryptitis_annotated.png`): dense infiltrate, neutrophils invading
  the crypt wall (= the definition of active), depleted goblets, eroded surface.
- **8B-ZOOM — the inflamed pattern** (`histo_cryptitis_zoom_annotated.png`): the *pattern*-level
  read — a carpet of small dark nuclei (vs a few dots in normal), depleted goblets, leaked red
  blood cells (bright salmon, no nucleus), and small dark cells crowding the epithelium. The
  source is soft, so we deliberately **do not pinpoint one neutrophil here** — that's the next
  slide's job.
- **8C — Crypt abscess** (`histo_abscess_annotated.png`): lumen full of neutrophils = abscess;
  neutrophil vs lymphocyte/plasma cell contrast; gland wall = epithelium.
- **8C-ZOOM — meet a neutrophil** (`histo_abscess_zoom_annotated.png`): the crisp,
  cell-resolving image — a single **neutrophil with a multi-lobed (segmented) nucleus**, the
  lumen packed with them (= crypt abscess), the epithelial gland wall ringing it, and a salmon
  **red blood cell (no nucleus)** as the classic look-alike not to miscount.

**Figure source:** `scripts/make_histology_zoom_figs.py` regenerates the two zoom figures from
the source jpgs (crop boxes are hard-coded for reproducibility).

---

## Slide 9 — The Insight That Shapes Everything

**What it is:** how focal disease dictates our modeling choice.

**Detail.** Because an "active" slide is *mostly calm tissue with a few guilty spots*
(Slide 2, point 3), we cannot trust the slide label as a per-tile label. Two consequences:

- **Use Multiple-Instance Learning (attention-MIL).** Instead of labeling tiles, we treat
  the whole slide as a "bag" of tiles with one bag-level label (active/inactive). An
  attention-MIL model (e.g. ABMIL / CLAM) learns to *find* the few tiles that justify the
  label and weights them — and that learned attention **is** our region heatmap. We'll
  cross-check our numbers against the Owkin IMILIA reference implementation on this data.
- **Validate grouped by slide/patient.** Held-out evaluation must keep all of a slide's
  tiles on one side of the split. Reported metrics: AUROC + agreement (e.g. kappa), always
  leave-slides-out.

**Build order for the head (smallest first):**
(a) mean-pool a slide's tile embeddings → logistic regression on active/inactive — a quick
"does the signal even exist?" check; (b) attention-MIL for localization; (c) optional: draw
a few QuPath polygons on a handful of slides and run the included linear probe.

**Takeaway:** focal + weak labels ⇒ MIL, not naive tile classification; and slide-grouped
validation is non-negotiable.

---

## Slide 10 — Where We Are & What's Next

**Done so far (as of 2026-06-28).**
- Built the mental model: pipeline = frozen embeddings → light head → heatmap (see Slides 11–12 / `APPROACH.md`).
- Learned the target biology and looked at active vs healed H&E — textbook images *and* real dataset tiles.
- Stood up this logbook (deck + these notes).
- **Environment ready** — project `.venv` with the light core (numpy, pandas, pillow, tifffile, scikit-learn, matplotlib, h5py).
- **Data downloaded & MD5-verified** — the pre-tiled HE patch set (`patch-dataset-HE.zip`, 8.76 GB).
- **Step 1 done** — `scripts/01_build_manifest.py` (+ `src/ibdpath/{paths,manifest}.py`) builds
  `artifacts/patch_manifest.csv`: 6,322 tiles · 140 slides, all 512px, every tile has a mask. Verified
  in code that the authors' Train/Val split leaks by slide (80 shared), Test is disjoint. 16 tests pass.
- **Step 2 done** — found the labels in the annotation/WSI filenames (`ID-{id}_HE_{active|inactive}`),
  derived all 140 via a ~1 MB central-directory tail-read of `TIFF-annotations.zip` (cross-checked vs
  Owkin IMILIA, zero disagreements) → committed `metadata/slide_labels.csv` (54 active / 86 inactive).
  `scripts/02_attach_labels.py` joins them onto the manifest → `patch_manifest_labeled.csv`
  (`label` + `slide_target` + `patient_id` per tile). 10 tests pass. See Slide 14.
- **Step 3 done** — built `.venv-embed` (Py 3.12 + torch/timm) and embedded all 6,322 tiles with frozen
  **H-optimus-0** on the Mac GPU (MPS) → `artifacts/embeddings/hoptimus0/<slide>.npy` (1,536-dim, L2-norm).
- **Step 4 done** — mean-pool + logistic regression, leave-patients-out: **AUROC 0.984** (κ 0.87). Verified
  honest (permutation→0.52, tile-count confound 0.66, best single dim 0.87). See Slide 15.

**Next steps.**
1. **Step 5 — attention-MIL** on the embeddings → per-tile attention = the inflamed/healed heatmap (the *where*).
2. **Step 6 — honest validation**: leave-patients-out AUROC + agreement, and log where the map fails.

---

## Slide 11 — The Whole Pipeline, End to End

**What it is:** a single consolidated view of the entire journey — from one raw slide to a
trustworthy map — pulling together the pieces introduced on Slides 2–9. A recap/reference slide.
The full prose version lives in [`APPROACH.md`](../APPROACH.md) at the repo root (added 2026-06-28).

**Detail — what goes in, and the five stages.**

The **input** is deliberately tiny: **one giant slide** (a gigapixel `.ndpi`, 1–4 GB) **+ one word**
(`active` or `inactive`). Everything else is the bridge to a coloured map.

1. **Tile (TRIDENT).** The slide is too big to feed to any model, so TRIDENT finds the tissue
   (ignoring blank glass), cuts it into ~1,000 small **tiles** (256×256 px), and *records each
   tile's (x, y) position* — we need those positions later to paint the map back in the right place.
2. **Embed (frozen foundation model).** Each tile is passed through a big pretrained model that
   turns it into an **embedding** — ~1,500 numbers, a "fingerprint" where look-alike tiles land near
   each other. We never train this model ("frozen"); we just use it as a fixed picture→numbers
   translator, and we **cache** the numbers so we never recompute them.
3. **Score (we build).** A tiny **head** reads each fingerprint and outputs `P(active)` for that tile.
4. **Stitch (we build).** Paint each tile's score onto the slide thumbnail — cool = healed, red =
   inflamed — giving the heatmap.
5. **Validate (we build).** Test on slides the model has never seen, **grouped by slide**, and report
   AUROC + agreement.

**Why it's split this way.** Stages 1–2 are an off-the-shelf, one-time cost — the pretraining is
already done and the embeddings are cached. We *own* only Stages 3–5, which are small, cheap, and run
comfortably on CPU. That is the entire "smallest useful slice" philosophy in one picture.

**Takeaway:** off-the-shelf tools carry the slide all the way to numbers; we own only the small,
cheap scoring → mapping → validation end, and we build it one script at a time.

---

## Slide 12 — The Three Modeling Steps (and why we never hand-label tiles)

**What it is:** how the "light head" from Slide 11 is actually built — three steps of rising
sophistication — plus the clarification that resolves the most common point of confusion: **we never
manually label individual tiles.**

**Detail — first, two kinds of label.** It is easy to blur these together; keep them apart:

- **Slide label** = *one word per slide* (`active`/`inactive`). This is the **only** label we ever
  need. It already exists — the dataset stored it in the WSI filename (`ID-X_Y.ndpi`) — so getting it
  is a spreadsheet *lookup*, not annotation. (Recovering this one word is the entire "label gap.")
- **Tile label** = saying which *individual tiles* are inflamed. We **do not** create these by hand.

**The three steps:**

- **(a) Mean-pool baseline.** Average a slide's ~1,000 embeddings into one vector, give it the slide's
  one word, fit logistic regression. Clean (no per-tile guessing) and fast — a "does the signal even
  exist?" gate (held-out AUROC ≳ 0.70 ⇒ proceed). Weakness: averaging *dilutes* a focal hotspot and
  gives no map. Uses **slide labels only**.
- **(b) Attention-MIL** *(the deliverable)*. Keep all tiles separate; the model learns a **weight** per
  tile (how much it counts toward the slide verdict) and votes with a weighted average. Trained on
  **slide labels only** — the model *teaches itself* which few tiles justify an "active" call, and
  those learned weights **are the heatmap**. This matches the data's own definition ("active if ≥1
  spot is inflamed"), which is exactly the MIL setup. Cross-checked against Owkin **IMILIA**.
- **(c) QuPath polygons** *(optional, last)*. Hand-draw a few inflamed/healed regions on a handful of
  slides → a small set of *clean, human-verified* tile labels, used as an honest yardstick for the
  heatmap. The **only** place a human marks anything; optional; it is what `ibd_inflamed_probe.py`
  already does.

**The analogy.** Imagine a box of 1,000 photos and you are told only *"this box contains at least one
fire-drill photo"* — never which ones. Attention-MIL learns, from that box-level statement alone, to
spot the fire-drill photos. Hand it ~1,000 tiles + the single word "active," and it figures out by
itself which tiles earned the word; those tiles become the map.

**Takeaway:** the only label we ever fetch is one word per slide — the model invents everything
tile-level on its own. Hand-drawn polygons are an optional, last, take-it-or-leave-it check, not a
prerequisite.

---

## Slide 13 — The Dataset (IBDColEpi) and its train/test groups

**What it is:** a reference slide describing exactly what data we're standing on, added 2026-06-28
after we looked inside the patch zip for Step 1.

**Where it comes from.** **IBDColEpi** = 140 H&E + 111 CD3-stained WSIs of colonic mucosa, collected
2007–2018 at NTNU / St. Olavs (Trondheim University Hospital, Norway), from patients with confirmed
IBD plus healthy controls. It was published (DataverseNO `doi:10.18710/TLA01U`) for an *epithelium
segmentation* paper — which is why every tile also ships with a pixel-level epithelium mask (DEFERRED
scope for us; we ignore the masks).

**What we actually downloaded.** Not the gigabyte WSIs — the **pre-tiled HE patch set**
(`patch-dataset-HE.zip`, 8.76 GB). Inside: **6,322 H&E tiles** (512×512 px, stored at downsample
`d=4` from a 2048px level-0 footprint), from **140 distinct HE slides**, each tile paired with a mask.
Filenames carry only `<id>_HE [d=,x=,y=,w=,h=]` — the position, but **not** the active/inactive label.

**The labels.** Slide-level **active vs inactive** ("active" = intraepithelial granulocytes in ≥1
spot). That word is stored in the *WSI* filename (`ID-X_Y.ndpi`), not the patch names — the "label
gap" Step 2 closes.

**Their three groups (measured from the zip):**

| Author's split | Tiles | Slides | What it is |
|---|---|---|---|
| Trainset | 4,973 | 104 | training WSIs |
| Validationset | 154 | 80 (a subset of the 104 train slides) | a random **3% of the *training tiles*** |
| Testset | 1,195 | 36 | fully held-out WSIs |
| **Total (distinct)** | **6,322** | **140** | = 104 train + 36 test |

**The crucial structure — and why we don't reuse their split for validation.** The authors first
split *WSIs* into 104 (train+val) and 36 (test). Then they peeled a random **3% of *tiles*** off the
training set to form "Validationset." So **Validation tiles come from the same slides as Training
tiles** — Train and Val overlap at the slide level. That was fine for their per-pixel segmentation
model, but for our **slide-level** inflamed/healed task it would *leak*: a held-out built that way
shares patients/slides/stain-batches with training and flatters the score (exactly Slide 2's warning).

So our rule:
- Keep the `split` column only as **provenance**, never as our evaluation boundary.
- Run our **own leave-slides-out** validation (GroupKFold by `slide_id`) across all 140 slides, and/or
  treat the **36 Testset slides as a clean external test** (they're the one slide-disjoint group).
- Caveat: the README notes two biopsies (one active, one inactive area) can come from the **same
  patient** under different IDs, but the anonymized IDs don't expose that link — so grouping by
  `slide_id` is the best we can do. (Lower risk than tile-level leakage, since those are different
  tissue with different labels.)

**Takeaway:** 6,322 HE tiles from 140 slides; labels are slide-level active/inactive (fetched in
Step 2); the authors' Train/Val split leaks by slide, so we validate our own way — leave-slides-out,
with the 36 Test slides as the clean held-out set.

---

## Slide 14 — The Labels: where they live & the 54/86 split

**What it is:** the result of the Step-2 investigation — where the active/inactive label actually
lives, how we recovered all 140 cheaply, and the class balance. Added 2026-06-28.

**Where the label lives.** Not in the patch files (they carry only `<ID>_HE`). The token is in the
dataset's **WSI/annotation filenames**: `ID-{id}_HE_{active|inactive}.tiff` (and the matching `.ndpi`).
"Active" = intraepithelial granulocytes in ≥1 spot (Slide 4).

**How we got all 140 without a big download.** A zip keeps a small *central directory*
(table-of-contents) at its very end. We fetched just the **last ~1 MB** of HuggingFace's
`TIFF-annotations.zip` (1.6 GB) with an HTTP byte-range request, parsed that central directory, and
read all 251 annotation filenames (140 HE + 111 CD3) — labels included. We then **cross-checked**
against Owkin IMILIA's GitHub `ibdcolepi_tiling_coords/` directory names (132 of our slides):
**zero disagreements**. Two independent sources agree → we trust the labels.

**The committed table.** Saved to `metadata/slide_labels.csv` — a small, version-controlled file
(deliberately *not* under `data/`, which `.gitignore` treats as disposable bulk). Its generation bakes
in assertions (must be 140 rows, must be 54/86, must match the manifest's slide_ids), so it can't
silently drift.

**Distribution.** 140 slides = **54 active / 86 inactive** (39% positive). Tiles: 2,819 from active
slides, 3,503 from inactive. **139 distinct patients** — slide `116` has two biopsies (`116_HE` and
`116_2_HE`); we group by `patient_id` for validation so they never straddle the split.

**Step 2's output.** `scripts/02_attach_labels.py` joins the labels onto the manifest →
`artifacts/patch_manifest_labeled.csv`: every tile gains `label`, `slide_target`, `patient_id`.

> ⚠️ **`slide_target` is the SLIDE's label copied onto each tile** — the bag label used by the
> mean-pool baseline and attention-MIL. It is **not** a per-tile truth: an active slide is mostly calm
> tiles. We never train a tile classifier on it directly (the focal-disease trap from Slide 9). The
> `slide_` prefix is a deliberate reminder.

**Takeaway:** labels recovered, verified twice, committed, and joined — 54 active / 86 inactive across
140 slides / 139 patients. Step 2 done; next is embeddings (Step 3).

---

## Slide 14B — The foundation model: H-optimus-0 (Step 3)

**What it is.** H-optimus-0 (Bioptimus) — a **1.1-billion-parameter Vision Transformer** pre-trained
**self-supervised** on a large H&E corpus (no labels). It has already learned general histology
structure; we treat that knowledge as fixed. Output width **1,536**; input **224×224** tiles (≈0.5 mpp);
open weights, **Apache-2.0** (gated on HuggingFace).

**How we use it — frozen.** Tile → encoder → one 1,536-d embedding, **inference only**, weights never
updated. Embeddings are cached to disk keyed by (slide, encoder, mag, patch) and reused by every head, so
we pay the ~42-min compute **once** (Step 3). This is the literal meaning of the project's "no encoder
training" rule.

**Why this is the core dependency.** The costly pre-training did the hard visual representation learning, so each
of our heads is tiny (logistic regression in Step 4, a small attention-MIL in Step 5) and trains in
seconds on a laptop GPU. **Swappable** by design: `H0-mini` / `UNI2` / `Virchow2` fit the same slot —
choose by a linear probe on held-out **IBD** slides, not oncology leaderboards.

**Why a dedicated slide (added later).** The deck jumped from labels straight to "embeddings computed";
a reader needs to know *what* did the embedding and what "frozen" means before the AUROC is presented. This slide
fills that gap and states the **core architecture decision** explicitly: frozen encoder + light heads,
every future upgrade = another head on the same embeddings.

---

## Slide 15 — Steps 3–4: Embeddings, then the signal check (AUROC 0.98)

**What it is:** the first *results* slide — we turned tiles into embeddings (Step 3) and confirmed the
foundation model's features actually separate active from inactive (Step 4). Added 2026-06-29.

**Step 3 — embeddings (the one heavy step).** Built a second env (`.venv-embed`, Python 3.12 +
torch/timm) because the 3.14 env has no torch wheels. Loaded **H-optimus-0** (frozen, 1.1B-param ViT,
Apache-2.0, gated) and ran every tile through it on the **Mac GPU via MPS** — resized to 224px,
H-optimus normalization, one forward pass each, then L2-normalized. Cached at
`artifacts/embeddings/hoptimus0/<slide>.npy` (6,322 tiles → 1,536-dim each; ~42 min, once). The
encoder is never trained — forward passes only (see Slides 11–12 on frozen vs fine-tuning).

**Step 4 — baseline signal check.** Mean-pool each slide's tile embeddings into one 1,536-vector, fit
logistic regression (class-weighted for the 54/86 imbalance), evaluate **leave-patients-out**
(GroupKFold by `patient_id`). Result: **AUROC 0.984**, Cohen's κ 0.87, accuracy 94%, per-fold all
≥0.96 — far above the 0.70 "signal exists" bar. Out-of-fold predictions saved to
`artifacts/baseline_oof_predictions.csv`.

**We didn't trust 0.98 blindly — three adversarial checks:**
- **Permutation test:** shuffle the labels → AUROC collapses to **0.52** (chance). The model
  needs the real labels → no pipeline leak or bug.
- **Tile-count confound:** active slides are slightly bigger (52 vs 41 tiles/slide), and tile-count
  *alone* gives AUROC 0.66 — a mild confound, but nowhere near 0.98, so the embeddings carry real
  morphology beyond mere size.
- **Single-feature ceiling:** the best individual embedding dimension already separates at 0.87 — the
  signal is near-linearly encoded in the frozen features.

**Caveat / what's next.** This is **slide-level** ("whether"), and mean-pooling gives **no map**. The
actual deliverable — *where* the inflammation is — is Step 5 (attention-MIL), which keeps tiles
separate and reads attention as the heatmap.

**Takeaway:** frozen H-optimus-0 embeddings separate active vs inactive at AUROC ≈ 0.98 (verified
honest). The signal is unambiguous; now we localize it.

---

## Slide 16 — Evidence: clean separation (the three-panel plot)

**What it is:** the evidence behind the 0.98, so we don't take it on faith. `artifacts/baseline_evidence.png`.

- **PCA (left):** the slide-mean embeddings form two clouds (active red / inactive green) separating along
  PC1, with a mixed seam; the 9 misclassified slides all sit in that seam (no wild outliers).
- **Score histogram (middle):** out-of-fold P(active) is strongly **bimodal** — ~77 inactive at ≈0, ~46
  active at ≈1. The model is confident *and* correct on the large majority (mean P: active 0.94, inactive
  0.07); only 4 slides land in the 0.3–0.7 uncertain band.
- **ROC (right):** sharp elbow, AUROC 0.984. With the permutation test collapsing to 0.52, this is honest.

**Takeaway:** the high AUROC reflects a clean, bimodal separation — not a lucky threshold.

---

## Slide 17 — Focal disease, shown (slide 17_HE)

**What it is:** the focal-disease thesis made visible — and the answer to "what about a small focus of
inflammation among normal?" `artifacts/focal_example.png` (and `slides/images/focal_17HE.png`).

We apply the baseline's linear "active direction" to **each tile** (an illustrative localizer — the
*proper* one is attention-MIL) to get per-tile P(active). For `17_HE` (labeled **active**, but the
mean-pool baseline wrongly called it inactive): the heatmap shows **one hot red focus** in otherwise
cool/normal mucosa; only ~11% of tiles score high. The hottest tiles (P≈0.80–0.86) are dense, dark,
architecturally disordered; the coolest (P≈0.17) are textbook orderly crypt rings.

**Why it matters:** this is *exactly* the case mean-pooling fails on (the focus is diluted in the
average), and the kind of slide attention-MIL was *meant* to catch. **Honest update (Step 5): MIL does
not catch it either** — `17_HE` stays missed (MIL P=0.02). Genuinely focal disease (~11% of tiles
inflamed) is the hard, open case for both heads; see the Slide 22 spectrum. The figure is still a working
**preview of the Step-5 heatmap**, produced from the same frozen embeddings.

**Caveat:** tiles show dense inflammation/architecture; confirming **neutrophils** specifically needs
one more zoom level — the morphology supports "active focus" but a pathologist would verify.

---

## Slide 18 — Where the model disagrees with the pathologist (failure modes)

**What it is:** the 9/140 out-of-fold disagreements, reconstructed from their tiles.
`artifacts/review/errors_grid.png`. Honest validation means *showing* the errors.

- **3 active → called inactive** (`105`, `17`, `64`): **focal** active disease the mean diluted. We
  *expected* attention-MIL to recover these — **it does not** (Step-5 result): all three stay missed, and
  MIL actually misses **more** active slides than the baseline (6 vs 3). Focal disease is unsolved in v1.
- **6 inactive → called active** (`9`, `140`, `59`, `98`, `58`, `16`): **dense/chronic** tissue (lymphoid
  aggregates, muscle/stroma, architectural change) or **tiny biopsies** — the encoder reads "abnormal" as
  "active," but active requires **neutrophils**. E.g. `132_HE` is correctly inactive (slide P=0.00) yet its
  hottest tile is 0.81 — a benign dense focus, not real activity. A genuine, harder confusion that MIL may
  not fully fix.

**The review tool.** A two-level gallery (`artifacts/review/index.html`): every slide's reconstructed
thumbnail + true label + model call, sorted by P(active), disagreements flagged; **click any card** to open
its tiles sorted by per-tile P(active) and hunt for neutrophils at the cell level. Built because a coarse
thumbnail shows *architecture*, but "active" is a *microscopic* (neutrophil) feature.

**Takeaway:** the errors are sensible and informative — they motivate MIL (focal misses) and flag a real
limitation (dense ≠ active).

---

## Slide 19 — Attention-MIL explained (Step 5)

**What it is:** the method behind the actual deliverable (the heatmap). NB the original hope — that it
would *beat* the mean-pool baseline on focal slides — **did not pan out** (it doesn't; see Slides 20/22).
The benefit is localization, not a better score.

**The setup.** Each slide = a **bag** of its tile embeddings (N × 1,536) + **one** label
(active/inactive). We have no per-tile labels. (This is "multiple-instance learning": the bag is
labeled, the instances aren't.)

**The model (gated attention-MIL, à la Ilse et al. 2018 / ABMIL).** Operates on the **frozen cached
embeddings** — the encoder is never touched, so the trainable part is tiny and fast (CPU/MPS):
1. (optional) a small linear layer projects each tile embedding `hᵢ`.
2. an **attention network** scores each tile, e.g. `eᵢ = wᵀ·tanh(V·hᵢ)`; a **softmax over the bag**
   turns the scores into weights `aᵢ` that **sum to 1**.
3. the slide is summarized as a **weighted average** `z = Σ aᵢ·hᵢ` (guilty tiles dominate).
4. a classifier reads `z` → `P(active)`. Trained **end-to-end on slide labels** by backprop through the
   attention + classifier only.

**Why the attention map = the heatmap.** To correctly call a focal active slide "active," the model is
*forced* to concentrate `aᵢ` on the few inflamed tiles — there's no other way to push `z` toward the
active region. So painting each tile by its `aᵢ` gives the inflamed/healed map, as a by-product of
getting the slide call right.

**vs the pooling alternatives.**
- **Mean-pool** (Step 4) = `aᵢ = 1/N`, every tile equal → focal signal diluted (missed `17_HE`).
- **Max-pool** = `aᵢ` one-hot on the single top tile → brittle, fooled by one benign hot tile (`132_HE`).
- **Attention** = learned, soft selection → sharper than mean, more robust than max.

**Plan.** Train ABMIL on the embeddings, evaluate **leave-patients-out** (GroupKFold by patient), report
AUROC + agreement, and compare to the 0.98 baseline — the point isn't a higher slide-level number, it's
the **localization**. (We hoped it would also recover focal misses; empirically it did not.) Cross-check
behaviour against Owkin IMILIA on this same data.

**Caveat.** MIL can over-attend to dense-but-benign tiles (lymphoid aggregates, muscle — the `132_HE`
trap), so honest validation and eyeballing the attention maps matter.

**Takeaway:** keep the tiles separate, let the model learn which few justify the label; that learned
attention is both the classifier's evidence and our heatmap. **→ Built and run — results on Slides 20–21.**

---

## Slide 20 — Attention-MIL result: the heatmap is real (Step 5)

**What we built.** `src/ibdpath/mil.py` — a small **gated attention-MIL** head (Ilse et al. 2018) on the
frozen embeddings: `proj(1536→128) → gated attention aᵢ → z = Σ aᵢ·hᵢ → linear classifier`. Trained in
the heavy env (`.venv-embed`), one bag at a time, class-weighted BCE, Adam, leave-patients-out
(GroupKFold by patient). Runner: `scripts/05_mil_head.py`; tests: `tests/test_mil.py` (6, all green).

**The numbers (leave-patients-out, 140 slides).**
- **AUROC 0.976** · κ 0.83 · accuracy 92% · per-fold AUROC `[1.0, 1.0, 0.967, 0.988, 0.953]`.
- vs the Step-4 baseline **0.984** → **96% call agreement**. Be honest: MIL is a **hair *below*** the
  baseline, not above, and it **misses more active slides (6 vs 3)** — including the focal ones.
  So the classification is **not** improved. **The benefit is the *where*** — a real per-tile heatmap
  the mean-pool baseline simply cannot produce.

**Reading the heatmap.** Colour = each tile's attention as a multiple of its **fair (uniform) share**
`1/N`, on a **fixed scale** (8× = full colour) so panels are directly comparable. Attention for each
slide comes from the **fold model that never saw it** (so the maps are leak-free, like the AUROC). The
**deck's primary figure is `mil_active_vs_calm.png`** — `137_HE` (active, one red hot-spot) beside
`144_HE` (inactive, none), with a legend; the clearest one-look explainer. `mil_heatmaps.png` is the
extended 3-panel version that also shows the `132_HE` trap.
- **`137_HE` (inflamed, called right, P=1.00):** one tile glows red — **38×** its fair share — and
  **91%** of all attention sits on just 3 of 47 tiles. ⚠️ But attention marks the model's *evidence*, not
  the disease's *extent*: by the per-tile localizer ~**72%** of this slide is actually inflamed (Slide 22),
  so "needle in a haystack" oversells it — the model simply only *needed* one blatant tile to be sure.
- **`144_HE` (clean inactive, P≈0):** no tile exceeds ~2× fair share → uniformly pale. Honest contrast:
  an inactive slide should *not* light up.
- **`132_HE` (the Step-4 trap, P≈0):** gated attention **spread out** (max 3× fair share) — it did **not**
  fixate on the dense-but-benign tile that the naive linear localizer scored 0.81. The sigmoid gate helps here: it can *suppress* a tile that merely looks busy.

**Important honesty note (the big one).** MIL **does not rescue focal disease** — it's *worse* there.
Active slides missed: **baseline** {105, 17, 64} (3); **MIL** {105, 15, 1, 17, 64, 83} (6). The
focal `17_HE` (11% inflamed) and `105_HE` (47%) stay missed by both. This is exactly why we did **not**
use `17_HE` as the demo, and why the headline is `137_HE` (a slide MIL got right). The honest one-liner:
**MIL buys the heatmap, not better accuracy.** Full extent picture on Slide 22.

**Artifacts:** `mil_oof_predictions.csv` (per-slide out-of-fold P), `mil_attention/<slide>.npy` (honest
per-tile attention for all 140), `mil_active_vs_calm.png` (deck primary), `mil_heatmaps.png` (3-panel).

---

## Slide 21 — Zoom on the focus: the tile attention picked really is inflamed (Step 5)

**What it shows (`artifacts/mil_137_hottest_tiles.png`).** The 3 tiles attention weighted most on `137_HE`,
pulled straight from the slide and viewed at tile resolution:
- **#1 (38× fair share):** a **dense inflammatory infiltrate at a denuded / eroded** mucosal surface, with
  scattered extravasated red cells — the textbook gross look of active disease.
- **#2 (4×):** crypts ringed by inflammatory cells.
- **#3 (1× = essentially ignored):** comparatively **orderly** crypts.

**Why this matters.** The model was trained on the **slide** label only — it was never told which tiles
contain disease. Yet its attention concentrated on exactly the eroded, infiltrated region. That is the
core idea of the project working end-to-end: *slide-level supervision → tile-level localization*, on
frozen foundation-model embeddings, with no per-tile annotation.

**Caveat on the claim.** At this magnification we can describe "dense infiltrate + erosion" confidently;
we **cannot** resolve individual intra-epithelial neutrophils by eye (the formal definition of "active").
The slide's *label* says active; the attention says *where the inflammation is densest*. Cell-level
confirmation (CellViT/HoVer-Net neutrophil detection) is **deferred scope** — a later upgrade on the same
tiles, not part of v1.

**The v1 deliverable is now complete in substance:** a validated, leak-free, per-tile inflamed-vs-healed
map on the slide. **Step 6** just locks down the honest validation reporting across all 140 slides.

---

## Slide 22 — Disease is a spectrum, and the focal end is the honest limit (Step 5)

**What it shows (`artifacts/disease_spectrum.png`, made by `scripts/make_disease_spectrum.py`).** Four
slides on ONE shared colour scale, coloured by the **per-tile localizer P(inflamed)** = *how inflamed each
tile looks* (this is *disease extent*, distinct from the attention map's *where the model focused*):
`17_HE` focal (11% inflamed) → `137_HE` patchy (72%) → `142_HE` diffuse (100%) → `144_HE` calm (5%).

**The two facts this slide makes honest.**
1. **Most active disease is broad, not focal.** Across the 54 active slides the median is **82%** of tiles
   inflamed; 38/54 are >70%; only **one** (`17_HE`) is under 20%. So the easy-looking diffuse slides are
   the common case, and both heads get them right.
2. **Focal disease is the unsolved part of v1.** `17_HE`/`105_HE` are missed by the baseline *and* by
   attention-MIL; MIL even misses **more** active slides overall (6 vs 3). We name this rather than hide it.

**Two maps, two questions (don't conflate them).** *Attention* (Slide 20) answers "which tiles did the
model lean on to make the call" — it concentrates even when disease is broad. *Per-tile P(inflamed)*
(here) answers "how much of the slide is inflamed" — the quantity a severity index (Nancy/Robarts) needs.
Both fall out of the same frozen embeddings; the extent map uses the illustrative slide-mean localizer
(the proper per-tile-supervised version is **deferred** scope).

**Bottom line for v1.** We deliver a real, leak-free inflamed-vs-healed map + an honest AUROC. We do **not**
deliver a solver for focal disease — that's a named open problem (more labelled data, or a head with some
per-tile supervision) for a later iteration.

---

## Slide 22B — Step 6: honest validation

What it shows (`artifacts/validation_report.png` + `validation_report.md`, from `scripts/06_validate.py`):
the leave-patients-out story for **both** heads, computed from their saved out-of-fold predictions (no new
model).
- **AUROC + 95% CI** (patient bootstrap, n=2000): baseline **0.984** (0.96–1.00), MIL **0.976** (0.95–0.99).
  Resampling whole *patients* keeps the interval honest about how few independent cases there are (139).
- **Confusion @0.5:** baseline sens 94% / spec 93% (FN 3); MIL sens 89% / spec 94% (FN 6) — MIL trades a
  little sensitivity for the heatmap.
- **Leakage check (the important one):** shuffle the labels and re-run the baseline CV → AUROC collapses to
  ~**0.46** (chance) over 25 shuffles. The 0.98 needs the real labels, so the pipeline isn't memorizing/leaking.
- **Head agreement:** 96%.

Helpers in `src/ibdpath/validate.py` (`bootstrap_auroc_ci`, `classification_metrics`, `call_agreement`);
`tests/test_validate.py` (5). This is a reporting step — it does not change any model.

---

## Slide 23 — Summary & what's next (v1 recap)

The honest one-slide close. Four cards: **(1) What we built** — one frozen FM (H-optimus-0) → embeddings
→ two light heads, no encoder training (Steps 1–5, each tested). **(2) Numbers** — baseline AUROC 0.984,
attention-MIL 0.976 (no better); leave-patients-out always grouped by patient; label-shuffle → 0.52 (no
leak). **(3) Deliverable** — a per-tile inflamed-vs-healed heatmap learned from slide labels only. **(4)
What we did NOT solve** — focal disease (`17_HE`, `105_HE`) missed by both heads (MIL misses more, 6 vs 3);
"abnormal ≠ neutrophils" false positives; cell-level confirmation deferred.

**Next:** Step 6 is **done** (validation locked in — Slide 22B). What remains are the **deferred** upgrades
(cell/neutrophil detection, stain norm, Geboes/Nancy/Robarts scoring) — each a new head on the **same**
frozen embeddings, per the core architecture decision.

---

## Slide 23B — Planned next: segmentation + cell typing (deferred)

A "what's next" slide — **not built** (deferred scope), shown so the planned chapter is visible. The upgrade
turns the per-tile black-box score into a **typed cell map**:
- **A — epithelium segmentation** as a light head on the **frozen H-optimus patch tokens** (16×16×1,536 per
  tile → light decoder → 224² mask). The dataset's `Labels_tif` masks are ground truth → **Dice,
  leave-patients-out**, vs a U-Net baseline. Region-level → rides the frozen FM (on-brand).
- **B — cell detection + typing** with a **dedicated** model (CellViT/HoVer-Net). Two honest gates:
  **resolution** (cell models want 40×; our 512-px patches are ~1–2 µm/px) and **neutrophil specificity**
  (PanNuke lumps "inflammatory"; neutrophils need a MoNuSAC-trained model). Instance detection is too fine
  for 14-px patch tokens, so B is *not* a light head on H-optimus — the pipeline is intentionally asymmetric.
- **Join** (cell ∩ compartment) → per-tile status (cryptitis/abscess/…) → Geboes/Nancy/Robarts roll-up.

First slice = `137_HE` active focus vs `144_HE` control. Detail: `docs/segmentation-and-cell-typing-plan.md`
(master) + `docs/cell-quantification-plan.md` (cell-typing deep-dive). **Planning only — confirm scope before building.**

---

## Slide 23C — Segmentation, first result (Component A, started)

The first build of the deferred segmentation work — **Gate 1 passed**. Epithelium segmentation as a *light
head on the frozen model*: tile → H-optimus `forward_features` → **256 patch tokens** (16×16×1,536) → a
per-patch **logistic probe** → P(epithelium) → upsample 16×16 → 512 → mask. No encoder training.

- **Held-out patients** (4 patients, 179 tiles; 12 slides total, patient split): patch-level **AUROC 0.995**,
  **Dice 0.846** on epithelium-containing tiles (per-tile examples 0.86–0.93).
- **Honesty:** the all-tiles Dice **0.771** is *inflated* — 21 empty-mask tiles score Dice 1.0 trivially; we
  headline the epithelium-tile number (0.846). Boundaries are blocky because the probe predicts at 16×16
  patch resolution → a **conv decoder** (or higher-res tiles) is the refinement.
- **Why it matters:** this is the **compartment layer** for cell-typing — it cleanly separates epithelium from
  the inflammatory lamina propria (visible in the active-focus overlay), which is what "intraepithelial
  neutrophil" needs.

Code: `src/ibdpath/epithelium.py` (mask loader + overlay, 4 tests) + `scripts/exp_epithelium_seg.py`
(experimental). Figure: `epithelium_seg_demo.png`. Still **deferred / experimental**; next = conv decoder,
then Component B (cell typing) + the join.

---

## Slide 24 — References & credits

A compact credits slide (full version in **`REFERENCES.md`** at the repo root). Covers: the **IBDColEpi**
dataset (CC0; Pettersen et al. 2021, `doi:10.18710/TLA01U`), the **H-optimus-0** foundation model
(Bioptimus, Apache-2.0), the **attention-MIL** method (Ilse et al. 2018), the software libraries actually
used (PyTorch, timm, scikit-learn, NumPy, pandas, Matplotlib, Pillow), the **Owkin IMILIA** reference
implementation, recommended-but-unused WSI tools (TRIDENT, OpenSlide, QuPath), and the teaching figures.

---

### Image credits
Example H&E teaching images (cryptitis / crypt-abscess) are from **Wikimedia Commons** (CC-BY-SA; by
*Nephron*) — purely illustrative, **not** from IBDColEpi. All **result** figures (heatmaps, spectrum,
evidence, overview) use real **IBDColEpi** tiles (CC0). Full citations with DOIs: **`REFERENCES.md`**.
