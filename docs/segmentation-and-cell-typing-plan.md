# Plan — compartment segmentation + cell typing (per-tile typed cell map)

> **Status: planning only.** Both pieces are in the project's **deferred** scope (cell / compartment
> segmentation). Nothing here is built. This is the master plan; the cell-typing component's deep-dive (the
> two feasibility gates, model choices) lives in [`cell-quantification-plan.md`](cell-quantification-plan.md).

## Why and what

Today, per tile we output one black-box activity score. To get **interpretable, clinically-aligned**
quantities — the substrate of Geboes / Nancy / Robarts — we need to know, per tile, **which cells** are
present and **where** they sit. That is two models plus a join:

```
        ┌─ A. compartment segmentation ─→ epithelium mask         (WHERE)
tile ───┤
        └─ B. cell detection + typing ──→ nuclei: location + type (WHO)
                         │
              join: each cell ∩ compartment
                         ↓
   per-tile cell-type-by-compartment counts → inflammation status → roll-up to Geboes/Nancy/Robarts
```

The clinically meaningful signal is the **combination**: a *neutrophil inside the epithelium* = cryptitis;
*filling a crypt lumen* = crypt abscess. Neither model alone gives that.

---

## Component A — compartment (epithelium) segmentation  ✅ has ground truth

> **✅ First slice DONE (Gate 1 passed).** A per-patch **logistic probe** on the frozen H-optimus patch
> tokens, patient-split on 12 slides → **held-out patch-level AUROC 0.995**, **Dice 0.846** on
> epithelium-containing tiles (per-tile 0.86–0.93). Works on the existing 512-px patches — no re-tiling
> needed for segmentation. Boundaries are 16×16-blocky → a **conv decoder** is the next refinement.
> Code: `src/ibdpath/epithelium.py` + `scripts/exp_epithelium_seg.py`; figure `epithelium_seg_demo.png`;
> deck Slide 23C.

The dataset ships pixel-level **epithelium masks** (`Labels_tif`) aligned to each tile — real labels to train
**and** validate on. (For tiles we already have, we can just *read* the mask; we train a model only to predict
masks on **new/unannotated** tissue, which is the actual task and the dataset's intended use.)

**On-brand architecture — segmentation as another light head on the frozen encoder:**

- **Encoder (frozen):** H-optimus-0 → **patch tokens**, a 16×16×1,536 dense feature map per 224×224 tile.
  (We currently cache only the CLS vector, so the embed step must also dump the patch tokens.)
- **Decoder (the only trained part):** start with a **linear** probe per patch token → 16×16 logits →
  bilinear upsample to 224×224; then try a small **conv / FPN** decoder for sharper boundaries.
- **Loss:** Dice + BCE on the masks.
- **Validation:** **Dice / IoU**, **leave-patients-out** (group by `patient_id`). **Baseline to beat:** a
  small **U-Net** trained from scratch — does the frozen FM actually help segmentation, not just classification?

**Steps:** (1) mask loader `src/ibdpath/epithelium.py` + overlay sanity check; (2) dense patch-token
extraction from frozen H-optimus; (3) light decoder + Dice/BCE, leave-patients-out; (4) Dice/IoU + vs U-Net;
(5) overlays.

**Risks:** 14-px patches → coarse boundaries (mitigate: conv decoder or higher-res tiles); patch tokens are
~256× the CLS cache (compute/storage); mask label noise.

---

## Component B — cell detection + typing  ⚠️ no ground truth here

A **dedicated instance model** — **HoVer-Net** or **CellViT/CellViT++** — detects every nucleus and assigns a
**type**. Full detail (the two gates, model options, first-slice steps) is in
[`cell-quantification-plan.md`](cell-quantification-plan.md); in brief:

- **Resolution gate:** cell models want ~40× (≈0.25 µm/px); our 512-px patches are ~1–2 µm/px → a neutrophil
  is only ~5–10 px. Likely need to **re-tile the `.ndpi` WSIs at 40×** for the demo slides.
- **Neutrophil-specificity gate:** PanNuke models give a lumped *"inflammatory"* class (not neutrophil-
  specific); true neutrophils need a **MoNuSAC**-trained model (classes incl. **Neutrophil**). Start with
  "inflammatory cell" as a proxy; don't over-claim "neutrophil".

**Honest architecture note — why this one is *not* a light head on H-optimus:** cell detection is
**instance-level at high resolution**; H-optimus's 14-px patch tokens are too coarse to localize ~10-px
nuclei. So Component B needs its own model. (CellViT++ *can* use a pathology-FM encoder, which keeps it partly
on-brand, but it's still gated by resolution.) The pipeline is intentionally **asymmetric**: A rides the
frozen FM, B is a dedicated model.

---

## The join → per-tile output

For each detected cell, test its centroid against the epithelium mask → a structured per-tile readout:

- `intraepithelial_neutrophils` (→ cryptitis), `crypt_lumen_neutrophils` (→ abscess),
  `lamina_propria_inflammatory_density`, `epithelial_cell_count`, neutrophil-to-lymphocyte ratio …
- a graded per-tile **status** (quiescent → mild → moderate → severe active), then a slide-level
  **Geboes / Nancy / Robarts** roll-up.

## Validation (asymmetric — and honest about it)

| Piece | Ground truth? | How we validate |
|---|---|---|
| Epithelium segmentation | **Yes** (dataset masks) | Dice / IoU, leave-patients-out, vs a U-Net baseline |
| Cell typing | **No** | qualitative overlays + correlation with v1's per-tile attention |
| Combined status | partial | does intraepithelial-inflammatory density rank `137_HE` (active focus) ≫ `144_HE` (healed)? |

## First slice (smallest useful)

`137_HE` active focus vs `144_HE` control, a handful of tiles: run A + B + the join, render the **typed-cell
overlay with the epithelium outline**, and check the active/healed contrast. **Not** the full index pipeline,
not all 140 slides.

**Deliverables:** `src/ibdpath/{epithelium,cells}.py` + `scripts/exp_seg_celltype.py` (clearly
*experimental / deferred*); a Dice number for segmentation; an overlay figure; a go/no-go note.

## Decision gates (resolve before coding)

1. Do the existing 512-px patches segment epithelium well enough, or do we extract patch tokens at higher res?
2. Do they have enough resolution for cell detection, or must we re-tile the WSIs at 40×? (most likely blocker)
3. Which cell model gives **neutrophils** specifically (MoNuSAC-trained), and does it transfer to colon H&E?

## Out of scope for this slice

Full per-tile graded status; crypt/gland segmentation; Geboes/Nancy/Robarts roll-up; training/fine-tuning a
cell model; running all 140 slides. See [`../ROADMAP.md`](../ROADMAP.md).
