# Plan — cell type & inflammation status per tile (first slice)

> **Status: planning only.** This crosses into the project's **deferred** scope (cell/compartment
> segmentation). Nothing here is built yet. The plan deliberately scopes the *smallest useful slice*, not
> the full Geboes/Nancy/Robarts pipeline — consistent with "build the smallest useful first slice".

## Goal of the first slice

On a **handful of slides** (not all 140), show that we can:

1. detect and classify nuclei in our tiles,
2. count **inflammatory cells inside the epithelium** (the substrate of "active" disease), using the
   dataset's epithelium masks, and
3. overlay the detections and produce one interpretable per-tile number (intraepithelial inflammatory-cell
   density),

then sanity-check that the active focus (`137_HE`) lights up and a healed control (`144_HE`) does not — and
that this agrees with v1's attention heatmap.

We are **not** building Geboes scoring, training a cell model, or running all slides in this slice.

## ⚠️ The two gates to resolve *before* coding

These decide whether the first slice is even feasible on the current data; resolve them first.

1. **Resolution.** Cell-detection models expect ~**40× (≈0.25 µm/px)**. Our pre-tiled patches are 512 px
   stored from a 2048-px level-0 crop (~4× downsample) → roughly **1–2 µm/px (~5–10×)**. A neutrophil
   (~10 µm) is then only ~5–10 px across — **marginal** for reliable single-cell detection.
   - **Decision:** test detection on the existing 512 px patches first. If quality is poor, **re-tile the
     WSIs (`.ndpi`) at 40×** for the few demo slides (needs the WSI files + OpenSlide/TRIDENT). This is the
     most likely blocker.
2. **Neutrophil specificity.** The common pretrained models classify **PanNuke** classes
   (Neoplastic / **Inflammatory** / Connective / Dead / Epithelial) — "Inflammatory" lumps neutrophils +
   lymphocytes + plasma cells, so it is **not neutrophil-specific**.
   - **Decision:** start with "inflammatory cell in epithelium" as a useful proxy (PanNuke model), and note
     that true neutrophil specificity needs a **MoNuSAC**-trained model (classes: Epithelial / Lymphocyte /
     Macrophage / **Neutrophil**) or a fine-tune. Don't over-claim "neutrophils" until the model actually
     distinguishes them.

## Prerequisites

- **Tools / weights:** **CellViT/CellViT++** or **HoVer-Net** with pretrained weights (PanNuke for a 5-class
  proxy; MoNuSAC if we want neutrophils). Check weight licences before committing them — likely **link, not
  vendor** (as with H-optimus).
- **Env:** a heavy env with torch + the chosen package (extend `.venv-embed` or a new `.venv-cells`).
- **Epithelium masks:** the dataset's `Labels_tif` are pixel-level epithelium masks aligned to each
  `Images_tif` tile (already identified). Build a small loader to read mask + tile from the patch zip.
  *(Using these masks is itself deferred scope — confirming as part of this plan.)*

## Steps

1. **Mask loader** — `src/ibdpath/epithelium.py`: read a tile's `Labels_tif` mask (0/1 epithelium) from the
   patch zip, aligned to the `Images_tif` tile. Unit-test on one known tile.
2. **Cell detector wrapper** — `src/ibdpath/cells.py`: load the frozen cell model, run one tile → list of
   nuclei `{x, y, type, prob}`. Keep it model-agnostic (CellViT or HoVer-Net behind one interface).
3. **Per-tile features** — for a tile: total cells by type; **cells whose centroid falls inside the
   epithelium mask**; intraepithelial inflammatory-cell **count** and **density** (per mm² or per epithelial
   area).
4. **Overlay figure** — draw nuclei on the tile (colour by type), outline the epithelium, flag the
   intraepithelial inflammatory cells. Render for: the `137_HE` hottest tile (active focus), a mid tile, and
   a `144_HE` tile (healed control).
5. **Sanity check** — does intraepithelial inflammatory density rank active-focus ≫ healed control? Does it
   correlate with v1's per-tile attention / P(active) on the same tiles? (A scatter of the two.)
6. **Experimental script** — `scripts/exp_cell_quant.py` (clearly marked *experimental / deferred*), writing
   `artifacts/cell_quant_demo.png` + a tiny CSV of per-tile counts. **No deck/README change until it works.**

## Deliverables

- `artifacts/cell_quant_demo.png` — overlays for active-focus vs healed-control tiles.
- A short results note (what the model found, did it transfer, resolution verdict).
- A go/no-go recommendation for the next slice (more slides → per-tile status → index roll-up).

## Validation (honest)

- **Qualitative first:** by-eye on the overlays + correlation with v1 attention. We have **no per-cell ground
  truth**, so the first slice is a feasibility check, not a measured accuracy.
- **Quantitative later:** needs annotated cells (or expert review) on a few regions to report neutrophil
  detection precision/recall before any clinical claim.

## Risks

- **Resolution mismatch** (Gate 1) — the most likely reason the first slice needs WSIs re-tiled at 40×.
- **Domain transfer** — PanNuke/MoNuSAC are mostly not colon-IBD; detections may be noisy on our stain/site.
- **Neutrophil vs other inflammatory cells** (Gate 2) — proxy only until a neutrophil-aware model is used.
- **Compute** — cell models are heavier per tile than the embedding pass; keep the first slice to a few tiles.

## Effort & decision gates

- **~½ day** to wire one detector + mask loader and render overlays on a few tiles (if the 512 px patches
  are usable).
- **+~1 day** if Gate 1 forces re-tiling the WSIs at 40× (OpenSlide/TRIDENT + the `.ndpi` files).
- **Decision gate after the demo:** only proceed to per-tile *status* and *index roll-up* if the detector
  transfers and the active/healed contrast is clear. Otherwise, fix detection (model/magnification) first.

## Out of scope for this slice (later upgrades)

Per-tile graded status; crypt/gland segmentation; Geboes / Nancy / Robarts roll-up; training/fine-tuning a
cell model; running all 140 slides. See `ROADMAP.md`.
