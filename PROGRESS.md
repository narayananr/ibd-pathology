# PROGRESS — resume here

> **Status as of 2026-06-29:** **Steps 1–5 done.** Built the manifest (6,322 tiles · 140 slides),
> attached labels (54 active / 86 inactive), embedded every tile with frozen **H-optimus-0** on the
> Mac GPU (`artifacts/embeddings/`), ran the baseline (**leave-patients-out AUROC 0.984**), and trained
> the **attention-MIL head**: **leave-patients-out AUROC 0.976** — a hair *below* the baseline, and it
> **misses more active slides (6 vs 3)**, so it does **not** improve classification. **Its only real win is
> the heatmap** (real per-tile attention; on `137_HE` 91% of attention on 3 tiles, hottest 38× fair share;
> the `132_HE` trap held). **Honest open problem: focal disease** (`17_HE`, `105_HE`) is missed by both
> heads. All 38 tests green.
> **Next action = Step 6 (`scripts/06_validate.py`): lock the honest validation reporting across all 140 slides.**

This file is the single "where are we / how do I continue" page. Pair it with the visual
logbook in [`slides/index.html`](slides/index.html) (the story) and [`slides/NOTES.md`](slides/NOTES.md)
(the deep-dive). Project rules live in [`CLAUDE.md`](CLAUDE.md); the ordered build plan in
[`BUILD_PLAN.md`](BUILD_PLAN.md); the plain-language end-to-end walkthrough in
[`APPROACH.md`](APPROACH.md).

---

## ▶️ How to resume in VSCode

1. Open this folder (`ibd-pathology/`) in VSCode.
2. **Select the Python interpreter** = the project venv:
   `Cmd+Shift+P` → "Python: Select Interpreter" → choose `./.venv/bin/python` (Python 3.14.4).
3. In a terminal, activate it if you want to run things by hand:
   ```bash
   source .venv/bin/activate      # then just `python ...`
   # (or call it directly without activating: .venv/bin/python ...)
   ```
4. Sanity check the env:
   ```bash
   .venv/bin/python -c "import numpy, pandas, PIL, tifffile, sklearn, matplotlib, h5py; print('env OK')"
   ```

---

## ✅ What's done

- **Understood the problem & biology** — active = neutrophils in epithelium (cryptitis →
  crypt abscess → erosion); inactive = orderly/quiet crypts. See the deck + NOTES.
- **Looked at real data** — opened textbook H&E examples *and* 4 genuine dataset tiles.
  Observation: `110_HE` tile looked actively inflamed (dense dark cells between crypts);
  `108_HE` looked calm (tidy crypt rings). Confirms the signal is visible by eye.
- **Environment (Step 0)** — created `.venv` (Python 3.14.4) and installed the **light core**:
  `numpy pandas pillow tifffile scikit-learn matplotlib h5py`. All import cleanly.
- **Downloaded the data** — `patch-dataset-HE.zip` (8.76 GB), **MD5 verified** = `c29af65c4a7e7bb4723e324657f22ccc`.
- **Decided the code format** — numbered scripts in `scripts/` + a small reusable `src/ibdpath/` module.
- **Step 1 — manifest built** — `scripts/01_build_manifest.py` parses the patch zip (no extraction)
  into `artifacts/patch_manifest.csv`: **6,322 tiles · 140 slides**, all 512px, every tile has a mask.
  Confirmed *in code* that the authors' **Train/Val split leaks by slide** (80 shared) while **Test is
  disjoint** → we validate leave-slides-out ourselves. Logic in `src/ibdpath/{paths,manifest}.py`;
  **16 unit/integration tests** (`tests/test_manifest.py`) all green.
- **Step 2 — labels found & attached** — the active/inactive token lives in the WSI/annotation
  filenames (`ID-{id}_HE_{active|inactive}.tiff/.ndpi`), absent from the patch set. Derived all **140**
  via a ~1 MB central-directory tail-read of HF `TIFF-annotations.zip` (no full download), cross-checked
  against Owkin IMILIA (132 overlap, **zero disagreements**) → committed `metadata/slide_labels.csv`
  (**54 active / 86 inactive**). `scripts/02_attach_labels.py` joins them onto the manifest →
  `artifacts/patch_manifest_labeled.csv` (adds `label`, `slide_target`, `patient_id`); **10 tests** green.
- **Step 3 — tiles embedded** — built `.venv-embed` (Py 3.12 + torch/timm; the 3.14 env has no torch),
  loaded frozen **H-optimus-0** (1.1B ViT, Apache-2.0, gated) and embedded all 6,322 tiles on the Mac
  GPU (MPS) → `artifacts/embeddings/hoptimus0/<slide>.npy` (1,536-dim, L2-normalized; ~42 min, cached).
- **Step 4 — baseline signal check** — mean-pool per slide → logistic regression, leave-patients-out:
  **AUROC 0.984**, κ 0.87, acc 94%. Verified honest: label-permutation → 0.52, tile-count confound 0.66,
  best single dim 0.87. Code in `src/ibdpath/{embed,baseline}.py`; 6 tests. → proceed to MIL.
- **Step 4 evidence & review tooling (2026-06-29)** — verified 0.98 honestly: `artifacts/baseline_evidence.png`
  (PCA/hist/ROC); tile-level t-SNE (`tile_embedding_map.png`, `extremes_separation.png`) shows an inflammation
  gradient + focal overlap (96% of P>0.7 tiles are from active slides; active slides ~45% inflamed-tiles vs
  inactive 1.3%). Built a **two-level review gallery** (`scripts/make_review_gallery.py` + `src/ibdpath/mosaic.py`):
  `artifacts/review/index.html` — reconstructed slide thumbnails + true/call, **click → per-slide tile montage**
  sorted by an *illustrative* per-tile localizer (the linear "active direction", **no scaler** — proper one is MIL).
  9/140 errors: 3 focal-active misses (MIL's job), 6 inactive→active (dense/chronic ≠ neutrophils; e.g. `132_HE`
  correctly inactive but hottest tile 0.81 = benign lymphoid/muscle). Deck **Slides 16–19** + NOTES capture all this.
- **Step 5 — attention-MIL (the *where*)** — `src/ibdpath/mil.py` = a small **gated ABMIL** head
  (`proj 1536→128 → gated attention → weighted pool → linear`) on the frozen embeddings; trained
  leave-patients-out in `.venv-embed`. **AUROC 0.976** (κ 0.83, acc 92%, per-fold all ≥0.95) — a hair
  *below* the 0.984 baseline, and it **misses MORE active slides (6 vs 3)**: baseline misses {105,17,64},
  MIL misses {105,15,1,17,64,83}. So **MIL does not improve classification and does NOT rescue focal
  disease** — `17_HE`(11% inflamed) & `105_HE`(47%) missed by both. **Its only real win is localization:**
  honest per-tile attention (fold model that never saw the slide) → real heatmaps. `137_HE` puts **91%** of
  attention on 3/47 tiles (hottest **38×** fair share = dense infiltrate at an eroded surface), `144_HE`
  stays diffuse, the **`132_HE` trap held**. ⚠️ Two distinct maps: *attention* (model's evidence) vs
  *per-tile P(inflamed)* (disease **extent**) — disease is a spectrum (median active slide 82% inflamed;
  `142_HE` 100%; `17_HE` 11%). Deck **Slides 17–22** corrected to say all this honestly; `tests/test_mil.py`
  (6) green → 38 tests total.

---

## 📦 Where things are

| Path | What it is |
|---|---|
| `.venv/` | project virtual environment (Python 3.14.4) — *git-ignored* |
| `data/ibdcolepi/patch-dataset-HE.zip` | the 8.76 GB HE patch set — *git-ignored* |
| `data/ibdcolepi/00_README.txt` | official dataset README (label semantics, splits) |
| `slides/index.html` | the running slide deck / build log (26 slides; open in a browser) |
| `slides/overview.html` | **standalone 5-slide overview** for a cold audience (problem→approach→results); `scripts/make_overview_figs.py` builds its problem figure |
| `slides/NOTES.md` | detailed per-slide notes |
| `slides/images/` | example H&E images used in the deck |
| `ibd_inflamed_probe.py` | pre-existing path-(c) probe (TRIDENT features + QuPath polygons → heatmap) |
| `APPROACH.md` | plain-language end-to-end walkthrough of the whole pipeline (deck Slides 11–12 in prose) |
| `REFERENCES.md` | citations & attribution — IBDColEpi (CC0), H-optimus-0, attention-MIL, libraries, tools |
| `README.md` | GitHub front page — what/results/how-to-run/data-download/limitations; **edit LICENSE copyright name before pushing** |
| `LICENSE` | MIT (code only); third-party data/asset terms noted at the bottom |
| `src/ibdpath/` | reusable module — `paths.py`, `manifest.py` (zip parsing), `labels.py` (labels + join) |
| `scripts/01_build_manifest.py` | Step 1 runner → writes `artifacts/patch_manifest.csv` |
| `tests/test_manifest.py` | 16 tests — `.venv/bin/python -m unittest tests.test_manifest` |
| `artifacts/patch_manifest.csv` | **Step 1 output** — one row per tile (6,322) — *git-ignored* |
| `metadata/slide_labels.csv` | **curated** slide → active/inactive (140 rows) — committed; provenance in `source` col |
| `scripts/02_attach_labels.py` | Step 2 runner → writes `artifacts/patch_manifest_labeled.csv` |
| `tests/test_labels.py` | 10 tests — `.venv/bin/python -m unittest tests.test_labels` |
| `artifacts/patch_manifest_labeled.csv` | **Step 2 output** — tiles + `label`/`slide_target`/`patient_id` — *git-ignored* |
| `.venv-embed/` | heavy env (Py 3.12 + torch/timm) for embedding/MIL — *git-ignored* |
| `src/ibdpath/embed.py` | embedding cache layer (paths, save/load, L2-normalize) — torch-free |
| `scripts/03_embed_tiles.py` | Step 3 runner (heavy env) → `artifacts/embeddings/hoptimus0/` |
| `artifacts/embeddings/hoptimus0/` | **Step 3 output** — per-slide `(n_tiles, 1536)` `.npy` + `meta.json` — *git-ignored* |
| `src/ibdpath/baseline.py` + `scripts/04_baseline_clf.py` | Step 4 — mean-pool + logreg, leave-patients-out AUROC |
| `artifacts/baseline_oof_predictions.csv` | **Step 4 output** — per-slide out-of-fold `p_active` — *git-ignored* |
| `src/ibdpath/mosaic.py` + `scripts/make_review_gallery.py` | reconstruct slide thumbnails (no WSI thumbs ship) + two-level review gallery |
| `artifacts/review/` | gallery `index.html` + per-slide `thumbs/` & tile `details/` — *git-ignored* |
| `slides/images/{baseline_evidence,focal_17HE,errors_grid}.png` | evidence figures embedded in deck Slides 16–18 |
| `src/ibdpath/mil.py` | **Step 5** — gated attention-MIL head (torch) + train/predict loop; runs in `.venv-embed` |
| `scripts/05_mil_head.py` | Step 5 runner — leave-patients-out train + AUROC + honest attention + demo heatmaps |
| `tests/test_mil.py` | 6 tests (heavy env) — `.venv-embed/bin/python -m unittest tests.test_mil` |
| `src/ibdpath/mosaic.py` → `heatmap_overlay()` | per-tile heat overlay on the reconstructed thumbnail (fixed `vmax` scale) |
| `artifacts/mil_oof_predictions.csv` | **Step 5 output** — per-slide out-of-fold `p_active` (MIL) — *git-ignored* |
| `artifacts/mil_attention/<slide>.npy` | **Step 5 output** — honest per-tile attention, all 140 slides — *git-ignored* |
| `slides/images/mil_{active_vs_calm,heatmaps,137_hottest_tiles}.png` | Step-5 figures (deck Slides 20–21; `active_vs_calm` is the primary) |
| `scripts/make_disease_spectrum.py` → `slides/images/disease_spectrum.png` | extent figure (focal→patchy→diffuse→calm, per-tile P) for deck Slide 22 |

### What's inside the patch zip (25,303 entries)
```
Trainset/ Validationset/ Testset/        # the 3 splits (Test has 36 distinct HE slides)
  └── Images_tif/    <slide>_HE [d=4,x=..,y=..,w=2048,h=2048].tif   # 512x512 H&E RGB tiles
  └── Labels_tif/    <same name>.tif                                # EPITHELIUM masks (class 0/1)
  └── Images_mibImg/ , Labels_mibCat/                               # same data in MIB tool format (ignore)
```
- The `.tif` H&E tiles are **512×512** (the `w=2048` in the name is the level-0 crop; stored downsampled).
- `Labels_tif` are **epithelium segmentation masks** (look black raw because values are 0/1, not brightness).
  ⚠️ Epithelium/compartment segmentation is **DEFERRED scope** — do not build on these masks without
  confirming scope first (see CLAUDE.md).

---

## ✅ Resolved at Step 2 — the label gap

Our target labels are **slide-level active vs inactive**. The token is in the **annotation/WSI
filename** as `ID-{id}_HE_{active|inactive}.tiff/.ndpi` — absent from the patch set (patches carry only
`<ID>_HE`). **Resolved (2026-06-28):** read all 140 labels from the **central directory** of HF
`TIFF-annotations.zip` via a ~1 MB HTTP range tail-read (no full download), cross-checked against Owkin
IMILIA's `ibdcolepi_tiling_coords/` dir names (132 overlap, zero disagreements). Committed to
`metadata/slide_labels.csv` (54 active / 86 inactive). `116`+`116_2` = same patient → group by
`patient_id` for validation. **Derivation method recorded in project memory** in case it must be regenerated.

---

## 🗂️ Planned code structure (to create from Step 1)

```
src/ibdpath/
  __init__.py
  paths.py        # one place for all data/output paths
  manifest.py     # parse patch filenames -> tidy records
  io.py           # load a tile (and mask) as arrays
scripts/
  01_build_manifest.py   # -> artifacts/patch_manifest.csv
  02_attach_labels.py    # -> adds active/inactive per slide
  03_embed_tiles.py      # frozen foundation model -> embeddings cache
  04_baseline_clf.py     # mean-pool + logistic regression, grouped by slide
  05_mil_head.py         # attention-MIL + heatmap
  06_validate.py         # leave-slides-out CV, honest metrics
artifacts/               # generated tables/caches/maps — git-ignored
```

---

## 🧭 Roadmap & current position

| Step | Script | Goal | Status |
|---|---|---|---|
| 0 | — | venv + light deps | ✅ done |
| 1 | `scripts/01_build_manifest.py` | read patch zip, parse filenames → `artifacts/patch_manifest.csv` (split, slide_id, x, y, w, h, has_mask) | ✅ done (+16 tests) |
| 2 | `scripts/02_attach_labels.py` | map slide_id → active/inactive → `patch_manifest_labeled.csv` | ✅ done (+10 tests) |
| 3 | `scripts/03_embed_tiles.py` | frozen H-optimus-0 → embeddings on Mac MPS (`.venv-embed`) | ✅ done |
| 4 | `scripts/04_baseline_clf.py` | mean-pool → logreg, leave-patients-out → **AUROC 0.984** | ✅ done (+6 tests) |
| 5 | `scripts/05_mil_head.py` | attention-MIL → per-tile scores → stitched heatmap → **AUROC 0.976** | ✅ done (+6 tests) |
| **6** | `scripts/06_validate.py` | leave-slides-out CV, AUROC + agreement | **⬅️ NEXT** |

### Step 5 — ✅ done (attention-MIL)
Built `src/ibdpath/mil.py` (gated ABMIL, `proj 1536→128 → gated attention → weighted pool → linear`) +
`scripts/05_mil_head.py` + `tests/test_mil.py` (6). Leave-patients-out **AUROC 0.976** (96% agreement with
the 0.984 baseline = parity by design). Two design fixes made the heatmap honest: (1) the **demo slide is
picked by attention concentration** among slides MIL got *right* (`137_HE`), not the focal *miss* `17_HE`
(which MIL also misses, P=0.02); (2) the heat colour is **attention ÷ uniform fair-share on a fixed `vmax`
scale**, so a diffuse inactive slide stays pale instead of being stretched to full red. The `132_HE` trap
held (gated attention stayed diffuse, max 3× fair share). Outputs: `mil_oof_predictions.csv`,
`mil_attention/`, `mil_heatmaps.png`, `mil_137_hottest_tiles.png`.

### Step 6 plan (NEXT — honest validation)
- `scripts/06_validate.py`: consolidate the leave-patients-out story for **both** heads (baseline + MIL) in
  one honest report — AUROC (with a CI via per-patient bootstrap), sensitivity/specificity at the chosen
  threshold, confusion matrix, the label-permutation null (≈0.5), and the head-vs-head agreement. Save a
  small `artifacts/validation_report.{md,png}` and a deck slide. No new modelling — this is the "how good
  is it, really, and how do we know it's not leaking" wrap for v1.
- Optional sanity cross-check vs **Owkin IMILIA** numbers on this same data (reference, not a target).

### Step 1 spec — ✅ done
Implemented in `src/ibdpath/manifest.py` + `scripts/01_build_manifest.py`; the CSV also carries a few
columns beyond the original spec: `downsample`, `stored_px`, `image_path`, `mask_path`. Original spec:
`scripts/01_build_manifest.py` should:
- read `data/ibdcolepi/patch-dataset-HE.zip` **without fully extracting** (use Python `zipfile`),
- for every `*/Images_tif/*.tif` entry, parse: `split` (Train/Val/Test), `slide_id` (e.g. `107_HE`),
  and the `[d=,x=,y=,w=,h=]` crop fields via regex,
- record whether a matching `Labels_tif` mask exists,
- write a tidy table to `artifacts/patch_manifest.csv` and print a quick summary
  (counts per split, #slides, tiles/slide).
- No heavy deps — `zipfile` + `re` + `pandas`. Runs in seconds on CPU.

---

## 🧱 Guardrails (don't forget)
- Validation is **always grouped by slide/patient**, never a random tile split.
- Classes are by **activity** (neutrophils in epithelium), not normality.
- **DEFERRED** (confirm scope before touching): stain normalization, cell/compartment seg
  (the epithelium masks), crypt/gland analysis, spatial graphs, Geboes/Nancy/Robarts heads.
- ⚠️ **Python 3.14 caveat:** PyTorch/TRIDENT may lack 3.14 wheels at Step 3 — plan to make a
  separate Py 3.11/3.12 env just for embedding if so.
