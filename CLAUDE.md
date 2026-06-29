# CLAUDE.md — IBD H&E Inflamed/Healed Pipeline

## What this project is
A computational-pathology pipeline that takes H&E whole-slide images (WSIs) of colonic
biopsies from IBD patients and produces an **inflamed-vs-healed region map** per slide,
plus an honest validation number. Long-term target: Geboes / Nancy / Robarts indices and a
histologic-remission call. We are building the **smallest useful first slice**, not the full
platform. When in doubt, do less.

## Core architecture decision
One **frozen** pathology foundation model encodes tiles into embeddings; everything is built
on top of those frozen embeddings. **No encoder training** — the heavy lifting is pretrained.
v1 = frozen FM embeddings → a light head → stitched heatmap. That is the whole first slice.

## v1 scope (build this, in order)
1. Ingest → tissue-segment → tile → embed WSIs — use **TRIDENT** (do NOT hand-roll OpenSlide tiling).
2. A light head on the frozen embeddings that scores inflamed vs healed.
3. Stitch per-tile scores into a heatmap overlay on the slide thumbnail.
4. Honest validation: leave-slides-out, report AUROC + agreement.

## Explicitly DEFERRED — do NOT build yet
Stain normalization, cell/compartment segmentation (CellViT++/HoVer-Net), crypt/gland
analysis, spatial graphs, and the Geboes/Nancy/Robarts scoring heads. Each is a later upgrade
onto the SAME embeddings. **If a task drifts into any of these, stop and confirm scope first.**

## Dataset — IBDColEpi (primary, public)
140 H&E + 111 CD3 WSIs of colonic mucosa, active/inactive IBD + healthy controls, from
NTNU / St. Olavs (Trondheim). Format `.ndpi` (OpenSlide/TRIDENT read it directly). All
epithelium is annotated at pixel level.
- **KEY label semantics:** labels are **slide-level** active vs inactive, where "active" =
  intraepithelial granulocytes (neutrophils) present. That IS our inflamed/healed line.
  Disease is focal, so most epithelium even in an "active" slide can look inactive →
  naive slide→tile labels are noisy; prefer MIL or self-drawn region polygons.
- Download: DataverseNO `doi:10.18710/TLA01U` · HuggingFace `andreped/IBDColEpi` ·
  Kaggle `henrikpe/251-he-cd3-wsis-annotated-epithelium-ibdcolepi`. A pre-tiled HE patch set
  exists in the Dataverse for a no-WSI day-1 test.
- **Reference implementation on this exact data:** Owkin IMILIA, github.com/owkin/imilia
  (attention-MIL inflammation prediction). Use it to sanity-check our numbers.

Because labels are slide-level, the head order is:
- (a) baseline: mean-pool slide embeddings → logistic regression on active/inactive (AUROC sanity check)
- (b) localization: attention-MIL (ABMIL/CLAM) on slide labels → attention map = region heatmap
- (c) optional: draw a few QuPath polygons on a handful of slides → run the included linear probe

## What's already in the repo
- `ibd_inflamed_probe.py` — tested glue. Reads TRIDENT feature `.h5` + QuPath GeoJSON polygons,
  labels tiles by polygon membership, trains a leave-slides-out logistic probe, and stitches a
  P(inflamed) heatmap over the thumbnail. Use for path (c); adapt its loaders for (a) and (b).

## Tooling
- **TRIDENT** (`mahmoodlab/trident`): WSI seg→tile→embed, 18+ foundation models, one CLI.
  Recipe: HEST segmenter, 20x, 256px tiles, no overlap. Outputs per-slide feature `.h5`
  (`features` (N,D), `coords` (N,2) top-left @ level 0) + QuPath-openable contour GeoJSON.
- **Foundation model:** pick by linear-probe on held-out IBD slides, NOT oncology leaderboards.
  Default to H-optimus-1 / H0-mini (open) to start; UNI2 or Virchow2 are strong but gated on HF.
- **QuPath:** view slides, draw/inspect annotations, export GeoJSON.

## Environment
- Python 3.10+. GPU strongly preferred for TRIDENT feature extraction (Otsu seg + small runs
  work on CPU).
- OpenSlide needs the **system** library in addition to the Python binding:
  `apt-get install openslide-tools` (Debian/Ubuntu) or `brew install openslide` (macOS).
- TRIDENT installs separately (it pulls torch/timm); match the CUDA build to your GPU.

## Conventions / guardrails
- Validation is **always grouped by slide (or patient)** — never a random tile split, it leaks badly.
- Define classes by histologic **ACTIVITY**, not normality: active = neutrophils in epithelium/
  lamina propria, cryptitis, crypt abscess, erosion/ulcer; healed/quiescent = no active
  inflammation (chronic architectural change is allowed).
- Cache embeddings to disk keyed by `(slide, encoder, mag, patch_size)`; never re-embed needlessly.
- Don't reach for a DEFERRED component to "improve" a result — confirm scope first.

## How to run (current)
```bash
# TRIDENT (after install)
python run_batch_of_slides.py --task seg    --wsi_dir ./wsis --job_dir ./trident_processed --segmenter hest --gpus 0
python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
python run_batch_of_slides.py --task feat   --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder hoptimus1 --mag 20 --patch_size 256
# Probe + heatmap (path c)
python ibd_inflamed_probe.py --feature_dir trident_processed/20x_256px_0px_overlap/features_hoptimus1 --annotation_dir annotations --thumbnail_dir trident_processed/thumbnails
```
(Confirm the exact `--patch_encoder` key from `run_batch_of_slides.py --help`; names get added over time.)
